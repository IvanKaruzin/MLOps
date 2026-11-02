"""Общий контракт эксперимента: конфиг, версии входов и подключение."""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import mlflow
import yaml


def load_params():
    params = yaml.safe_load(Path(os.environ.get("HW6_PARAMS", "conf/hw67.yaml")).read_text(encoding="utf-8"))
    validate_params(params)
    return params


def validate_params(params):
    train = params["train"]
    for key in ("max_steps", "train_limit", "val_limit", "batch_size", "grad_accum", "eval_every"):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"train.{key}: нужно положительное целое число")
    if type(train["seed"]) is not int:
        raise ValueError("train.seed: нужно целое число")
    if train.get("response_format", "source") not in ("source", "json_genres"):
        raise ValueError("train.response_format: допустимы source и json_genres")
    if train.get("response_format") == "json_genres" and (type(train.get("view_max_seq_len")) is not int
                                                          or train["view_max_seq_len"] <= 0):
        raise ValueError("JSON training view требует положительный view_max_seq_len")
    sampling = params["sampling"]
    if sampling.get("method") not in ("stratified_shuffle", "full_shuffle") or type(sampling.get("seed")) is not int or sampling["seed"] < 0:
        raise ValueError("Нужны поддерживаемый sampling.method и неотрицательный целочисленный sampling.seed")
    for key in ("eval_batch_size", "epochs", "worker_timeout_seconds"):
        if key in train and (type(train[key]) is not int or train[key] <= 0):
            raise ValueError(f"train.{key}: нужно положительное целое число")
    if train.get("worker_timeout_seconds", 21600) > 172800:
        raise ValueError("Таймаут worker не должен превышать двое суток")
    if train.get("lr_schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("train.lr_schedule: допустимы constant и cosine")
    if (isinstance(train.get("warmup_ratio", 0.0), bool)
            or not isinstance(train.get("warmup_ratio", 0.0), (int, float))
            or not math.isfinite(train.get("warmup_ratio", 0.0)) or not 0 <= train.get("warmup_ratio", 0.0) < 1):
        raise ValueError("train.warmup_ratio: нужно конечное число от нуля до единицы")
    if train.get("checkpoint_selection", "best_val") not in ("best_val", "last"):
        raise ValueError("train.checkpoint_selection: допустимы best_val и last")
    for key in ("train", "val", "source_train", "source_val", "tokenization_params"):
        if not isinstance(params["data"].get(key), str) or not params["data"][key]:
            raise ValueError(f"data.{key}: нужен путь к файлу")
    if params["data"]["source_train"] == params["data"]["source_val"]:
        raise ValueError("Исходные train и val должны различаться")
    if "auxiliary" in params["data"] and (not isinstance(params["data"]["auxiliary"], str)
                                           or not params["data"]["auxiliary"]):
        raise ValueError("data.auxiliary: нужен путь к отдельному training/dev bundle")
    candidates = params["sweep"]
    if len(candidates) < 6 or len({item["name"] for item in candidates}) != len(candidates):
        raise ValueError("Нужно минимум шесть кандидатов с уникальными именами")
    for item in candidates:
        if type(item["r"]) is not int or item["r"] <= 0 or not math.isfinite(item["lr"]) or item["lr"] <= 0:
            raise ValueError("Ранг и learning rate должны быть положительными")
        modules = item["target_modules"]
        if not isinstance(modules, list) or not modules or len(modules) != len(set(modules)):
            raise ValueError("Нужны разные именованные target_modules")
    configurations = {(item["r"], item["lr"], tuple(item["target_modules"])) for item in candidates}
    if len(configurations) != len(candidates) or any(len({item[i] for item in configurations}) < 2 for i in range(3)):
        raise ValueError("Каждая конфигурация уникальна; варьируются r, lr и target_modules")
    revision = params["model"]["revision"]
    if not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("До sweep закрепите точную commit-ревизию модели")


def connect(params):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", params["tracking"]["uri"]))
    mlflow.set_experiment(params["tracking"]["experiment"])


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def code_digest():
    h = hashlib.sha256()
    for p in sorted(Path("src").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(b"uv.lock")
    h.update(Path("uv.lock").read_bytes())
    return h.hexdigest()


def provenance(params):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hashes = {f"source_{split}_sha256": digest(params["data"][f"source_{split}"])
                     for split in ("train", "val")}
    source_hashes["tokenization_params_sha256"] = digest(params["data"]["tokenization_params"])
    if params["data"].get("auxiliary"):
        source_hashes["auxiliary_sha256"] = digest(params["data"]["auxiliary"])
    return {
        "git_sha": sha,
        "git_dirty": str(bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())),
        "code_sha256": code_digest(),
        "data_train_sha256": digest(params["data"]["train"]),
        "data_val_sha256": digest(params["data"]["val"]),
        **source_hashes,
        "config_sha256": hashlib.sha256(json.dumps(
            {"params": params, "source_hashes": source_hashes}, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }


def verify_provenance(params, expected):
    current = provenance(params)
    for key in ("code_sha256", "config_sha256", "data_train_sha256", "data_val_sha256"):
        if current[key] != expected.get(key):
            raise ValueError(f"Входы изменились во время sweep: {key}")


def save_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
