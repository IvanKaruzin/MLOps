"""Bounded CPU work, verified simulator artifacts and local prefix cache."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

from hw11.metrics import Metrics


@dataclass(frozen=True)
class Settings:
    pod_name: str = "local"
    model_version: str = "A"
    artifact_path: str = str(Path(__file__).with_name("artifacts") / "model-a.json")
    artifact_sha256: str = "0cddfaca828bf0d8e66df2f716bf81c3abec759ea9806c5a83dc1579ca8cd9fa"
    warmup_seconds: float = 20
    never_ready: bool = False
    broken_probes: bool = False
    max_running: int = 1
    max_waiting: int = 32
    queue_timeout_seconds: float = 30
    cache_capacity: int = 16
    hit_iterations: int = 75000
    miss_iterations: int = 700000
    chunk_iterations: int = 25000

    def __post_init__(self):
        for name in ("warmup_seconds", "queue_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or value > 600:
                raise ValueError(f"Invalid {name}")
        for name in ("max_running", "max_waiting", "cache_capacity", "hit_iterations", "miss_iterations", "chunk_iterations"):
            value = getattr(self, name)
            minimum = 0 if name in ("max_waiting", "cache_capacity") else 1
            maximum = 100000000 if name.endswith("iterations") else 1024
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid {name}")
        if self.model_version not in ("A", "B"):
            raise ValueError("Expected model version A or B")
        if len(self.artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.artifact_sha256):
            raise ValueError("Expected SHA-256")

    @classmethod
    def from_env(cls):
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = os.environ.get(name.upper())
            if raw is None:
                continue
            if field.type is bool:
                if raw.lower() not in ("true", "false"):
                    raise ValueError(f"Invalid {name}")
                values[name] = raw.lower() == "true"
            else:
                values[name] = field.type(raw)
        return cls(**values)


class AdmissionError(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class Lease:
    """Hold capacity until physical executor work ends, even after disconnect."""

    def __init__(self, backend, queue_seconds):
        self.backend = backend
        self.queue_seconds = queue_seconds
        self.pending = None
        self.closed = False
        self.released = False

    async def compute(self, iterations):
        loop = asyncio.get_running_loop()
        self.pending = loop.run_in_executor(self.backend.executor, self.backend.cpu_work, iterations)
        return await asyncio.shield(self.pending)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pending is not None and not self.pending.done():
            self.pending.add_done_callback(lambda _: self.release())
        else:
            self.release()

    def release(self):
        if self.released:
            return
        self.released = True
        self.backend.running -= 1
        self.backend.metrics.running.dec()
        self.backend.slots.release()


class Backend:
    def __init__(self, settings):
        self.settings = settings
        self.metrics = Metrics()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_running, thread_name_prefix="hw11-cpu")
        self.slots = asyncio.Semaphore(settings.max_running)
        self.running = 0
        self.waiting = 0
        self.cache = OrderedDict()
        self.warmup_finished = False
        self.draining = False
        self.error = None
        self.loaded_version = None
        self.loaded_sha256 = None

    @staticmethod
    def cpu_work(iterations):
        # OpenSSL performs real CPU work and releases the GIL; the event loop stays responsive.
        return hashlib.pbkdf2_hmac("sha256", b"hw11-simulated-prefix", b"fixed-cpu-workload", iterations)

    async def load(self):
        try:
            raw = Path(self.settings.artifact_path).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            artifact = json.loads(raw)
            if (digest != self.settings.artifact_sha256 or artifact.get("model_version") != self.settings.model_version
                    or artifact.get("backend_mode") != "simulated" or artifact.get("schema_version") != 1):
                raise ValueError("Artifact verification failed")
            self.loaded_version = artifact["model_version"]
            self.loaded_sha256 = digest
            await asyncio.sleep(self.settings.warmup_seconds)
            self.warmup_finished = not self.settings.never_ready
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = type(exc).__name__

    @property
    def ready(self):
        return self.warmup_finished and not self.draining

    async def acquire(self):
        if not self.warmup_finished:
            raise AdmissionError(503, "backend_not_ready")
        queued = time.monotonic()
        # No await between observing capacity and registering as a waiter.
        waiting = self.slots.locked()
        if waiting and self.waiting >= self.settings.max_waiting:
            raise AdmissionError(429, "queue_full")
        if waiting:
            self.waiting += 1
            self.metrics.waiting.inc()
        try:
            if waiting:
                await asyncio.wait_for(self.slots.acquire(), timeout=max(.001, self.settings.queue_timeout_seconds))
            else:
                await self.slots.acquire()
        except TimeoutError as exc:
            raise AdmissionError(429, "queue_timeout") from exc
        finally:
            if waiting:
                self.waiting -= 1
                self.metrics.waiting.dec()
        self.running += 1
        self.metrics.running.inc()
        delay = time.monotonic() - queued
        self.metrics.queue.observe(delay)
        return Lease(self, delay)

    def prefix(self, prompt, messages):
        source = messages[0].content if messages else prompt[:128]
        return hashlib.sha256(source.encode()).hexdigest()

    async def prepare(self, lease, prompt, messages):
        key = self.prefix(prompt, messages)
        hit = key in self.cache
        if hit:
            self.cache.move_to_end(key)
        self.metrics.cache.labels("hit" if hit else "miss").inc()
        await lease.compute(self.settings.hit_iterations if hit else self.settings.miss_iterations)
        if self.settings.cache_capacity:
            self.cache[key] = None
            self.cache.move_to_end(key)
            while len(self.cache) > self.settings.cache_capacity:
                self.cache.popitem(last=False)
        return hit

    def debug(self):
        return dict(ready=self.ready, warmup_finished=self.warmup_finished, draining=self.draining,
                    running=self.running, waiting=self.waiting, cache_size=len(self.cache),
                    model_version=self.loaded_version, artifact_sha256=self.loaded_sha256, error=self.error,
                    backend_mode="simulated", pod=self.settings.pod_name)

    async def close(self):
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=False)
