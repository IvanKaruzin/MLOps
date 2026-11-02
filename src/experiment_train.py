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
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import save_json
from src.data import batches, load_split
from src.memory import PeakMemory
from src.runtime import memory_metric, resolve_device, resolve_dtype, set_seed, synchronize
from src.sampling import sample_split, epoch_order
from src.training_view import auxiliary_metadata, build_training_view, fingerprint, load_auxiliary
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
    auxiliary, auxiliary_proof = load_auxiliary(params)
    if auxiliary_proof:
        proof["auxiliary"] = auxiliary_proof
        for name in ("train", "val"):
            proof[name]["original_ids"] = list(proof[name]["ids"])
            proof[name]["original_selected_rows"] = len(selected[name])
            selected[name].extend({"id": row["id"], "_auxiliary": True} for row in auxiliary[name])
            proof[name]["ids"].extend(row["id"] for row in auxiliary[name])
            proof[name]["selected_records"].extend(auxiliary_metadata(row) for row in auxiliary[name])
            proof[name]["selected_rows"] = len(selected[name])
            proof[name]["auxiliary_rows"] = len(auxiliary[name])
    return selected["train"], selected["val"], splits["train"]["pad_token_id"], proof


def processed_summary(identities, selected_records):
    """Describe actually consumed examples separately from the sampled pool."""
    by_id = {record["id"]: record for record in selected_records}
    records = [by_id[identity] for identity in identities]
    return {
        "ids": list(identities), "records": records, "count": len(records),
        "unique_count": len(set(identities)),
        "strata_counts": dict(sorted(Counter(record["stratum"] for record in records).items())),
        "year_counts": dict(sorted(Counter(str(record["release_year"]) for record in records
                                            if record["release_year"] is not None).items())),
        "genre_cardinality_counts": dict(sorted(Counter(str(record["genre_cardinality"])
                                                        for record in records).items())),
        **({"kind_counts": dict(sorted(Counter(record.get("kind", "genre") for record in records).items())),
            "source_counts": dict(sorted(Counter(record.get("source", "original") for record in records).items()))}
           if any("source" in record for record in records) else {}),
    }


def ordered_epoch(examples, params, epoch):
    return (epoch_order(examples, params["sampling"]["seed"], epoch)
            if params["sampling"]["method"] == "full_shuffle" else list(examples))


def check_epoch_budget(params, pool_rows):
    config = params["train"]
    steps_per_epoch = math.ceil(pool_rows / (config["batch_size"] * config["grad_accum"]))
    if "epochs" in config and config["max_steps"] != config["epochs"] * steps_per_epoch:
        raise ValueError("max_steps must equal epochs times the complete actual training-pool budget")
    return steps_per_epoch


def expected_sampling_manifest(params):
    """Reconstruct the planned pool and exact consumption from immutable inputs."""
    _, _, _, sampling = prepare_data(params)
    config = params["train"]
    identities = sampling["train"]["ids"]
    per_step = config["batch_size"] * config["grad_accum"]
    steps_per_epoch = check_epoch_budget(params, len(identities))
    consumed, epoch = [], 0
    for first_step in range(0, config["max_steps"], steps_per_epoch):
        steps = min(steps_per_epoch, config["max_steps"] - first_step)
        consumed.extend(ordered_epoch(identities, params, epoch)[:steps * per_step])
        epoch += 1
    sampling["processed_train"] = processed_summary(consumed, sampling["train"]["selected_records"])
    sampling["epochs_started"] = epoch
    return sampling


def prepare_training_data(params):
    examples, val, pad, sampling = prepare_data(params)
    examples, val, view = build_training_view(params, examples, val)
    return examples, val, pad, sampling, view


@torch.no_grad()
def evaluate(model, examples, pad_id, device, response_format="source", batch_size=1):
    was_training = model.training
    model.eval()
    try:
        total, tokens = 0.0, 0
        for batch in batches(examples, batch_size, pad_id, False, 0):
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


def learning_rate_multiplier(step, config):
    warmup = math.ceil(config["max_steps"] * config.get("warmup_ratio", 0.0))
    if step < warmup:
        return (step + 1) / warmup
    if config.get("lr_schedule", "constant") == "constant":
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, config["max_steps"] - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class AdapterCheckpoint:
    """Keep adapter weights on CPU; select only on the original validation NLL."""
    def __init__(self):
        self.loss, self.step, self.state = float("inf"), None, None

    def consider(self, model, step, loss, force=False):
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Checkpoint validation loss must be finite and nonnegative")
        if force or loss < self.loss:
            self.loss, self.step = loss, step
            self.state = {key: value.detach().cpu().clone()
                          for key, value in get_peft_model_state_dict(model).items()}

    def restore(self, model):
        if self.state is None:
            raise ValueError("No trained checkpoint was selected")
        set_peft_model_state_dict(model, self.state)


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
    check_epoch_budget(params, len(examples))
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
    model.config.use_cache = False
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=candidate["r"], lora_alpha=candidate["r"] * params["lora"]["alpha_over_r"],
        lora_dropout=params["lora"]["dropout"], target_modules=candidate["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=candidate["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: learning_rate_multiplier(index, t))
    original_val_ids = set(sampling.get("val", {}).get("original_ids", [row["id"] for row in val]))
    primary_val = [row for row in val if row["id"] in original_val_ids]
    auxiliary_val = [row for row in val if row["id"] not in original_val_ids]
    if not primary_val:
        raise ValueError("The original validation split must remain nonempty")
    eval_batch_size = t.get("eval_batch_size", 1)
    def validation():
        values = {"val_loss": evaluate(model, primary_val, pad, device, response_format, eval_batch_size)}
        if auxiliary_val:
            values["aux_val_loss"] = evaluate(model, auxiliary_val, pad, device, response_format, eval_batch_size)
        return values
    synchronize(device)
    start = time.perf_counter()
    initial = validation()
    base = initial["val_loss"]
    log(initial, 0)
    checkpoints = [{"step": 0, **initial}]
    best = AdapterCheckpoint()
    memory.checkpoint()
    step, epoch = 0, 0
    processed_ids = []
    while step < t["max_steps"]:
        epoch_examples = ordered_epoch(examples, params, epoch)
        stream = list(batches(epoch_examples, t["batch_size"], pad, False, 0))
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
                processed_ids.extend(example["id"] for example in epoch_examples[first:first + t["batch_size"]])
                loss_sum += float(loss.detach())
                memory.checkpoint()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            actual_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            memory.checkpoint()
            step += 1
            metrics = {"train_loss": loss_sum, "learning_rate": actual_lr}
            if step % t["eval_every"] == 0 or step == t["max_steps"]:
                metrics.update(validation())
                checkpoints.append({"step": step, **{key: value for key, value in metrics.items() if key != "train_loss"}})
                best.consider(model, step, metrics["val_loss"], force=t.get("checkpoint_selection", "best_val") == "last")
                print(f"{candidate['name']} step={step} val={metrics['val_loss']:.4f}", flush=True)
            log(metrics, step)
            if step >= t["max_steps"]:
                break
        epoch += 1
    end_val_loss = metrics["val_loss"]
    best.restore(model)
    selected_metrics = validation()
    if not math.isclose(selected_metrics["val_loss"], best.loss, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Restored checkpoint does not reproduce its original validation loss")
    log({"selected_val_loss": selected_metrics["val_loss"], "selected_step": best.step}, step)
    synchronize(device)
    seconds = time.perf_counter() - start
    out = Path(out)
    sampling["processed_train"] = processed_summary(processed_ids, sampling["train"]["selected_records"])
    sampling["epochs_started"] = epoch
    if sampling != expected_sampling_manifest(params):
        raise ValueError("Actual processed ids/epochs differ from the complete planned sampling manifest")
    sampling_sha = hashlib.sha256(json.dumps(sampling, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save_json(out / "sampling.json", sampling)
    save_json(out / "training-view.json", view)
    (out.parent / "training-view.json.gz").write_bytes(gzip.compress(
        json.dumps({"train": examples, "val": val}, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(), mtime=0))
    model.save_pretrained(out)
    tok.save_pretrained(out)
    weights = out / "adapter_model.safetensors"
    checkpoint = {"policy": t.get("checkpoint_selection", "best_val"), "metric": "original_val_token_weighted_nll",
                  "primary_val_rows": len(primary_val), "auxiliary_val_rows": len(auxiliary_val),
                  "primary_val_ids_sha256": fingerprint([row["id"] for row in primary_val]),
                  "selected_step": best.step, "training_steps": step, "selected_val_loss": selected_metrics["val_loss"],
                  "end_val_loss": end_val_loss, "checkpoints": checkpoints,
                  "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    save_json(out / "checkpoint.json", checkpoint)
    manifest = {"model": {**m, "revision": revision}, "generation": params["generation"],
                "candidate": candidate, "sampling_sha256": sampling_sha,
                "training_view_sha256": view_sha, "response_format": response_format,
                "loss_implementation": loss_implementation, "checkpoint": checkpoint}
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
    results = {"base_val_loss": base, "final_val_loss": selected_metrics["val_loss"],
        "end_val_loss": end_val_loss, "selected_step": best.step, "primary_val_rows": len(primary_val),
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
        "sampling_method": params["sampling"]["method"], "sampling_seed": params["sampling"]["seed"],
        "checkpoint_policy": t.get("checkpoint_selection", "best_val"), "selected_adapter_sha256": checkpoint["adapter_sha256"]}
    if auxiliary_val:
        results["aux_val_loss"] = selected_metrics["aux_val_loss"]
    del wrapper, model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return results, details, answers
