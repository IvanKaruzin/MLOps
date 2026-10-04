"""LoRA training with token-weighted evaluation and gradient accumulation."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, batches, load_split
from src.memory import PeakMemory
from src.runtime import (
    allocated_bytes, memory_metric, reset_peak_memory, resolve_device,
    resolve_dtype, set_seed, synchronize,
)

TRAIN_CODE = ("src/train.py", "src/data.py", "src/runtime.py", "src/config.py", "src/collate.py", "src/memory.py", "uv.lock")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants", "tokenize", "generate")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dependency_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}


def inputs_fingerprint(params: dict) -> str:
    """Bind artifacts to the code, configuration and actual input tensors."""
    digest = hashlib.sha256()
    for name in TRAIN_CODE:
        digest.update(name.encode())
        digest.update((PROJECT_ROOT / name).read_bytes())
    digest.update(json.dumps({key: params.get(key) for key in TRAIN_PARAMS}, sort_keys=True).encode())
    digest.update(json.dumps(dependency_versions(), sort_keys=True).encode())
    for key in ("train", "val"):
        path = Path(params["data"][key])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        digest.update(key.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    if type(freeze_first) is not int or not 0 <= freeze_first < n_layers:
        raise ValueError(f"freeze_first должен быть от 0 до {n_layers - 1}")
    cfg = params["lora"]
    if cfg.get("modules_to_save"):
        raise ValueError("modules_to_save сохраняет полные веса; здесь разрешены только LoRA-матрицы")
    targets = cfg["target_modules"]
    if not isinstance(targets, list) or not targets or any("embed" in x or "lm_head" in x for x in targets):
        raise ValueError("target_modules: нужен список проекций attention/MLP без словаря")
    return LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        target_modules=targets, modules_to_save=None, task_type="CAUSAL_LM",
        layers_to_transform=list(range(freeze_first, n_layers)),
    )


def supervised_tokens(batch: dict) -> int:
    # Causal LM shifts logits left and labels right; labels[:, 0] never enters loss.
    return int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum().item())


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int, peak_tracker=None) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
            batch = {key: value.to(device) for key, value in batch.items()}
            tokens = supervised_tokens(batch)
            if not tokens:
                continue
            output = model(**batch)
            if peak_tracker is not None:
                peak_tracker.sample()
            loss = float(output.loss.item())
            del output
            if not math.isfinite(loss):
                raise RuntimeError("Validation loss is not finite")
            total += loss * tokens
            count += tokens
    finally:
        model.train(was_training)
        if device.type == "mps":
            torch.mps.empty_cache()
    if not count:
        raise ValueError("Validation split has no shifted supervised tokens")
    return total / count


def validate_training_config(cfg: dict, max_steps) -> None:
    for key in ("epochs", "batch_size", "grad_accum", "eval_every", "eval_batch_size"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError(f"train.{key} должен быть положительным целым")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps должен быть положительным целым или null")
    for key in ("lr", "max_grad_norm"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"train.{key} должен быть положительным конечным")
    if not 0 <= cfg["warmup_ratio"] <= 1 or cfg["weight_decay"] < 0:
        raise ValueError("Некорректный warmup_ratio или weight_decay")


def run_training(model, examples, val_examples, pad_id, device, cfg, optimizer, scheduler,
                 total_steps: int) -> dict:
    with PeakMemory(device) as peak_tracker:
        result = _run_training(model, examples, val_examples, pad_id, device, cfg,
                               optimizer, scheduler, total_steps, peak_tracker)
    result["peak_memory_mb"] = round(peak_tracker.used / 1048576, 1)
    result["peak_rss_mb"] = round(peak_tracker.rss / 1048576, 1)
    return result


def _run_training(model, examples, val_examples, pad_id, device, cfg, optimizer, scheduler,
                  total_steps: int, peak_tracker) -> dict:
    """Average gradients over supervised tokens, including the final short group."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    reset_peak_memory(device)
    peak = allocated_bytes(device)
    curve_train, curve_val = [], []
    eval_seconds = 0.0

    def evaluate_at(step):
        nonlocal eval_seconds, peak
        synchronize(device)
        started = time.perf_counter()
        loss = evaluate(model, val_examples, pad_id, device, cfg["eval_batch_size"], peak_tracker)
        synchronize(device)
        eval_seconds += time.perf_counter() - started
        peak = max(peak, allocated_bytes(device))
        curve_val.append([step, round(loss, 6)])
        print(f"  шаг {step}/{total_steps}: val {loss:.6f}", flush=True)
        return loss

    synchronize(device)
    started = time.perf_counter()
    base_val = evaluate_at(0)
    model.train()
    step, processed_tokens, processed_supervised, processed_examples = 0, 0, 0, 0
    for epoch in range(cfg["epochs"]):
        micro_count = math.ceil(len(examples) / cfg["batch_size"])
        group_tokens, group_loss, group_micro = 0, 0.0, 0
        for micro_index, batch in enumerate(batches(
            examples, cfg["batch_size"], pad_id, shuffle=True, seed=cfg["seed"] + epoch
        ), 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            tokens = supervised_tokens(batch)
            if not tokens:
                raise ValueError("Training batch has no shifted supervised tokens")
            loss = model(**batch).loss
            peak_tracker.sample()
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise RuntimeError(f"Training loss is not finite before optimizer step {step + 1}")
            # Summed NLL first; divide gradients once by the complete group's token count.
            (loss * tokens).backward()
            peak_tracker.sample()
            group_tokens += tokens
            group_loss += value * tokens
            group_micro += 1
            processed_tokens += int(batch["attention_mask"].sum().item())
            processed_supervised += tokens
            processed_examples += batch["input_ids"].shape[0]
            peak = max(peak, allocated_bytes(device))
            if group_micro < cfg["grad_accum"] and micro_index < micro_count:
                continue
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(group_tokens)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            peak_tracker.sample()
            peak = max(peak, allocated_bytes(device))
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(group_loss / group_tokens, 6)])
            print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.6f}", flush=True)
            group_tokens, group_loss, group_micro = 0, 0.0, 0
            if step % cfg["eval_every"] == 0 or step == total_steps:
                evaluate_at(step)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    if step != total_steps:
        raise RuntimeError(f"Training finished at {step} steps, expected {total_steps}")
    synchronize(device)
    seconds = max(0.0, time.perf_counter() - started - eval_seconds)
    return {
        "steps": step, "base_val_loss": base_val, "final_val_loss": curve_val[-1][1],
        "diverged": False, "curve_train": curve_train, "curve_val": curve_val,
        "seconds": round(seconds, 3), "eval_seconds": round(eval_seconds, 3),
        "seconds_per_step": round(seconds / step, 3),
        "processed_tokens": processed_tokens, "processed_supervised_tokens": processed_supervised,
        "processed_examples": processed_examples,
        "train_tokens_per_sec": round(processed_tokens / seconds, 1) if seconds else 0,
        "peak_memory_mb": round(peak / 1048576, 1), "memory_metric": memory_metric(device),
    }


def dir_size_mb(path: Path) -> float:
    return round(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1048576, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all_layers")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", help="smoke outputs: adapter_<variant>/ and metrics/")
    parser.add_argument("--val-limit", type=int)
    args = parser.parse_args()
    if (args.max_steps is not None or args.val_limit is not None) and not args.out:
        raise ValueError("CLI smoke limits require --out to preserve production artifacts")
    params = load_params()
    variants = {variant["name"]: variant for variant in params["variants"]}
    if args.variant not in variants:
        raise ValueError(f"Unknown variant {args.variant!r}: {sorted(variants)}")
    cfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps")
    validate_training_config(cfg, max_steps)
    if args.val_limit is not None and args.val_limit <= 0:
        raise ValueError("val-limit должен быть положительным")
    device, dtype = resolve_device(params["model"]["device"]), resolve_dtype(params["model"]["dtype"])
    # This precedes base-model loading AND the random initialization of LoRA A.
    set_seed(cfg["seed"])
    fingerprint = inputs_fingerprint(params)
    train_blob, val_blob = load_split(params["data"]["train"]), load_split(params["data"]["val"])
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    tokenizer.padding_side = params["tokenize"]["padding_side"]
    for blob in (train_blob, val_blob):
        if blob.get("model") != params["model"]["name"] or blob["pad_token_id"] != tokenizer.pad_token_id:
            raise ValueError("Tokenized input model/pad_token_id does not match tokenizer")
        if any(token >= len(tokenizer) for example in blob["examples"] for token in example["input_ids"]):
            raise ValueError("Input token is outside the tokenizer vocabulary")
    examples = train_blob["examples"]
    val_examples = val_blob["examples"][:args.val_limit] if args.val_limit else val_blob["examples"]
    model = AutoModelForCausalLM.from_pretrained(params["model"]["name"], dtype=dtype).to(device)
    base_revision = getattr(model.config, "_commit_hash", None)
    n_layers = model.config.num_hidden_layers
    freeze_first = variants[args.variant]["freeze_first"]
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, n_layers, freeze_first))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    micro_per_epoch = math.ceil(len(examples) / cfg["batch_size"])
    total_steps = math.ceil(micro_per_epoch / cfg["grad_accum"]) * cfg["epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)
    print(f"[{args.variant}] {device}, trainable {trainable:,}/{total:,} ({trainable / total:.3%}), steps {total_steps}", flush=True)
    metrics = run_training(model, examples, val_examples, train_blob["pad_token_id"],
                           device, cfg, optimizer, scheduler, total_steps)
    if fingerprint != inputs_fingerprint(params):
        raise RuntimeError("Code or tokenized inputs changed while training; refusing stale artifacts")
    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "training_metadata.json").write_text(json.dumps({
        "model": params["model"], "tokenize": params["tokenize"],
        "generate": params["generate"], "inputs_fingerprint": fingerprint,
        "dependency_versions": dependency_versions(), "base_model_revision": base_revision,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics.update({
        "variant": args.variant, "freeze_first": freeze_first, "model": params["model"]["name"],
        "device": device.type, "dtype": params["model"]["dtype"], "seed": cfg["seed"], "lr": cfg["lr"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"], "planned_steps": total_steps,
        "trainable_params": trainable, "total_params": total, "trainable_share": round(trainable / total, 6),
        "adapter_dir": str(adapter_dir), "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": fingerprint, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "validation_examples": len(val_examples), "validation_limited": args.val_limit is not None,
        "dependency_versions": dependency_versions(), "base_model_revision": base_revision,
        "memory_sample_interval_ms": 10 if device.type == "mps" else None,
    })
    metrics_dir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {metrics['steps']} steps, {metrics['seconds']} s, "
          f"peak {metrics['peak_memory_mb']} MB, adapter {metrics['adapter_size_mb']} MB -> {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
