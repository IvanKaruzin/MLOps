"""Короткий LoRA-ран из ДЗ 5 с логированием на каждом шаге оптимизатора."""
import gc
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import save_json
from src.data import batches, load_split
from src.memory import PeakMemory
from src.runtime import memory_metric, resolve_device, resolve_dtype, set_seed, synchronize
from src.sampling import sample_split
from src.training_view import build_training_view, fingerprint
from src.train import supervised_loss


def prepare_data(params):
    """Select complete-source strata without rewriting HW4 tensors or raw data."""
    splits = {name: load_split(params["data"][name]) for name in ("train", "val")}
    for blob in splits.values():
        if blob.get("model") != params["model"]["name"]:
            raise ValueError("Тензоры токенизированы другой моделью")
    train_ids = {example["id"] for example in splits["train"]["examples"]}
    val_ids = {example["id"] for example in splits["val"]["examples"]}
    if train_ids & val_ids:
        raise ValueError("Полные train и val содержат пересекающиеся id")
    if splits["train"]["pad_token_id"] != splits["val"]["pad_token_id"]:
        raise ValueError("pad_token_id train и val различаются")
    selected, proof = {}, {}
    for name in ("train", "val"):
        selected[name], proof[name] = sample_split(
            splits[name], params["data"][f"source_{name}"], params["train"][f"{name}_limit"],
            params["sampling"]["seed"], name, params["sampling"]["method"])
    return selected["train"], selected["val"], splits["train"]["pad_token_id"], proof


def processed_summary(identities, selected_records):
    """Describe actually consumed examples separately from the sampled pool."""
    by_id = {record["id"]: record for record in selected_records}
    records = [by_id[identity] for identity in identities]
    return {
        "ids": list(identities), "records": records, "count": len(records),
        "unique_count": len(set(identities)),
        "strata_counts": dict(sorted(Counter(record["stratum"] for record in records).items())),
        "year_counts": dict(sorted(Counter(str(record["release_year"]) for record in records).items())),
        "genre_cardinality_counts": dict(sorted(Counter(str(record["genre_cardinality"])
                                                        for record in records).items())),
    }


def expected_sampling_manifest(params):
    """Reconstruct the planned pool and exact consumption from immutable inputs."""
    _, _, _, sampling = prepare_data(params)
    config = params["train"]
    identities = sampling["train"]["ids"]
    per_step = config["batch_size"] * config["grad_accum"]
    steps_per_epoch = math.ceil(len(identities) / per_step)
    full_epochs, remaining = divmod(config["max_steps"], steps_per_epoch)
    consumed = identities * full_epochs + identities[:remaining * per_step]
    sampling["processed_train"] = processed_summary(consumed, sampling["train"]["selected_records"])
    sampling["epochs_started"] = math.ceil(config["max_steps"] / steps_per_epoch)
    return sampling


def prepare_training_data(params):
    examples, val, pad, sampling = prepare_data(params)
    examples, val, view = build_training_view(params, examples, val)
    return examples, val, pad, sampling, view


@torch.no_grad()
def evaluate(model, examples, pad_id, device, response_format="source"):
    was_training = model.training
    model.eval()
    try:
        total, tokens = 0.0, 0
        for batch in batches(examples, 1, pad_id, False, 0):
            batch = {k: v.to(device) for k, v in batch.items()}
            n = int((batch["labels"][:, 1:] != -100).sum())
            if n:
                loss = supervised_loss(model, batch) if response_format == "json_genres" else model(**batch).loss
                total += float(loss) * n
                tokens += n
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError("В val нет токенов ответа под loss")
    value = total / tokens
    if not math.isfinite(value):
        raise ValueError("val loss не конечен")
    return value


def train(params, candidate, out, log, expected_view_sha256=None):
    device = resolve_device(params["model"]["device"])
    with PeakMemory(device) as memory:
        result, details, answers = _train(params, candidate, out, log, memory, expected_view_sha256)
    result["peak_memory_mb"] = memory.used / 1048576
    details["memory_sample_interval_ms"] = 10 if device.type == "mps" else 0
    return result, details, answers


def _train(params, candidate, out, log, memory, expected_view_sha256=None):
    t, m = params["train"], params["model"]
    set_seed(t["seed"])
    device, dtype = resolve_device(m["device"]), resolve_dtype(m["dtype"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    examples, val, pad, sampling, view = prepare_training_data(params)
    view_sha = fingerprint(view)
    if expected_view_sha256 is not None and view_sha != expected_view_sha256:
        raise ValueError("Actual training view differs from the parent preflight; refusing model load")
    response_format = view["response_format"]
    loss_implementation = "supervised_suffix_same_objective" if response_format == "json_genres" else "full_forward"
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], dtype=dtype).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Не удалось закрепить commit базовой модели Hugging Face")
    tok = AutoTokenizer.from_pretrained(m["name"], revision=revision)
    if tok.pad_token_id != pad:
        raise ValueError("pad_token_id токенизатора не совпадает с тензорами ДЗ4")
    if t["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=candidate["r"], lora_alpha=candidate["r"] * params["lora"]["alpha_over_r"],
        lora_dropout=params["lora"]["dropout"], target_modules=candidate["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=candidate["lr"])
    synchronize(device)
    start = time.perf_counter()
    base = evaluate(model, val, pad, device, response_format)
    log({"val_loss": base}, 0)
    memory.checkpoint()
    step, epoch = 0, 0
    processed_ids = []
    while step < t["max_steps"]:
        # The sampler already determines one shuffled order, shared by every
        # candidate. Keeping it here makes the recorded manifest executable.
        stream = list(batches(examples, t["batch_size"], pad, False, 0))
        for pos in range(0, len(stream), t["grad_accum"]):
            group = stream[pos:pos + t["grad_accum"]]
            # Последняя неполная группа тоже даёт шаг; нормируем на токены ответа.
            counts = [int((b["labels"][:, 1:] != -100).sum()) for b in group]
            n_tokens = sum(counts)
            if not n_tokens:
                raise ValueError("В train-батче нет токенов ответа")
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro_index, (batch, count) in enumerate(zip(group, counts)):
                if not count:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                unweighted = supervised_loss(model, batch) if response_format == "json_genres" else model(**batch).loss
                loss = unweighted * (count / n_tokens)
                if not torch.isfinite(loss):
                    raise ValueError("train loss не конечен")
                loss.backward()
                first = (pos + micro_index) * t["batch_size"]
                processed_ids.extend(example["id"] for example in examples[first:first + t["batch_size"]])
                loss_sum += float(loss.detach())
                memory.checkpoint()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            memory.checkpoint()
            step += 1
            metrics = {"train_loss": loss_sum}
            if step % t["eval_every"] == 0 or step == t["max_steps"]:
                metrics["val_loss"] = evaluate(model, val, pad, device, response_format)
                print(f"{candidate['name']} step={step} val={metrics['val_loss']:.4f}", flush=True)
            log(metrics, step)
            if step >= t["max_steps"]:
                break
        epoch += 1
    synchronize(device)
    seconds = time.perf_counter() - start
    out = Path(out)
    sampling["processed_train"] = processed_summary(processed_ids, sampling["train"]["selected_records"])
    sampling["epochs_started"] = epoch
    sampling_sha = hashlib.sha256(json.dumps(sampling, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save_json(out / "sampling.json", sampling)
    save_json(out / "training-view.json", view)
    (out.parent / "training-view.json.gz").write_bytes(gzip.compress(
        json.dumps({"train": examples, "val": val}, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(), mtime=0))
    model.save_pretrained(out)
    tok.save_pretrained(out)
    manifest = {"model": {**m, "revision": revision}, "generation": params["generation"],
                "candidate": candidate, "sampling_sha256": sampling_sha,
                "training_view_sha256": view_sha, "response_format": response_format,
                "loss_implementation": loss_implementation}
    save_json(out / "manifest.json", manifest)
    # Генерация тем же кодом, который затем загрузится из реестра.
    from src.adapter_model import AdapterModel
    import pandas as pd
    wrapper = AdapterModel()
    wrapper.model, wrapper.tokenizer, wrapper.device, wrapper.manifest = model.eval(), tok, device, manifest
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    prompts = params["generation"]["prompts"]
    answers = wrapper.predict(None, pd.DataFrame({"prompt": prompts}))
    processed_counts = Counter(processed_ids)
    results = {"base_val_loss": base, "final_val_loss": metrics["val_loss"],
        "seconds": seconds, "steps": step,
        "processed_examples": len(processed_ids), "processed_unique_examples": len(set(processed_ids)),
        "train_pool_rows": len(examples), "val_rows": len(val),
        "processed_tokens": sum(sum(example["attention_mask"]) * processed_counts[example["id"]]
                                for example in examples),
        "processed_supervised_tokens": sum(sum(label != -100 for label in example["labels"][1:])
                                           * processed_counts[example["id"]] for example in examples),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_mb": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1048576}
    details = {"base_revision": revision, "device": str(device), "memory_metric": memory_metric(device),
        "train_rows": len(examples), "val_rows": len(val), "sampling_sha256": sampling_sha,
        "training_view_sha256": view_sha, "response_format": view["response_format"],
        "loss_implementation": loss_implementation,
        "sampling_method": params["sampling"]["method"], "sampling_seed": params["sampling"]["seed"]}
    del wrapper, model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return results, details, answers
