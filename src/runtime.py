"""Device selection, random state and accelerator memory measurements."""

import os
import random
import sys

import numpy as np
import psutil
import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Неподдерживаемое устройство {name!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS недоступен")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"model.dtype = {name!r}, допустимо: {sorted(DTYPES)}")
    return DTYPES[name]


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def allocated_bytes(device: torch.device) -> int:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    # ru_maxrss records transient CPU peaks between Python measurements.
    if sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    return int(psutil.Process().memory_info().peak_wset)


def memory_metric(device: torch.device) -> str:
    return {"mps": "torch.mps.driver_allocated_memory (sampled)",
            "cuda": "torch.cuda.max_memory_allocated"}.get(device.type, "process peak RSS")
