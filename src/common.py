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
    return {
        "git_sha": sha,
        "git_dirty": str(bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())),
        "code_sha256": code_digest(),
        "data_train_sha256": digest(params["data"]["train"]),
        "data_val_sha256": digest(params["data"]["val"]),
        "config_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
    }


def save_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
