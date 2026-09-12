"""Пиковая память устройства и RSS; общий замер для ДЗ-1 и ДЗ-2."""

import gc
import sys
import threading

import torch

try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None


def resolve_device(params: dict) -> torch.device:
    """Развернуть device: auto в конкретное устройство — ровно один раз.

    Строка «auto» уходит в device_map и включает диспетчер accelerate,
    который для шага обучения только мешает. Решаем здесь и передаём дальше
    уже конкретное имя.
    """
    name = params["model"]["device"]
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_rss() -> tuple[int, str]:
    """Пик RSS процесса в байтах И метка источника метрики.

    Метка возвращается не для красоты: «пик 1001 МБ» без указания, чем это
    снято, — не результат, а повод для спора. Тем более что RSS и память
    ускорителя — разные величины (см. PeakMemory ниже).

    Три ОС меряют по-разному:

    * macOS и Linux — `resource.getrusage(RUSAGE_SELF).ru_maxrss`, high-water
      mark процесса; на macOS он в байтах, на Linux в килобайтах;
    * Windows — `psutil.Process().memory_info().peak_wset`: модуля `resource`
      там нет вовсе. Обратное тоже верно — поля `peak_wset` нет на macOS и
      Linux, и код, написанный только под него, у соседа падает.

    Если недоступно ничего — исключение. Тихий ноль хуже отсутствия числа:
    ноль попадает в отчёт и его выдают за результат.
    """
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if sys.platform == "darwin" else peak * 1024), "ru_maxrss"
    if psutil is not None and hasattr(psutil.Process().memory_info(), "peak_wset"):
        return int(psutil.Process().memory_info().peak_wset), "peak_wset"
    raise RuntimeError(
        f"нечем снять пик RSS на платформе {sys.platform}: модуля resource нет, "
        "а psutil не установлен либо не отдаёт peak_wset. Выполните uv sync."
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def device_allocated_bytes(device: torch.device) -> int:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return peak_rss()[0]


def device_metric_source(device: torch.device) -> str:
    if device.type == "mps":
        return "torch.mps.driver_allocated_memory"
    if device.type == "cuda":
        return "torch.cuda.max_memory_allocated"
    return peak_rss()[1]


class PeakMemory:
    """Максимум MPS с интервалом 10 мс; CUDA/RSS используют high-water mark.

    MPS включает кэш аллокатора и буферы MPSGraph. Выборочный максимум
    может пропустить короткие всплески между замерами; sample() позволяет
    дополнительно измерить память на границах forward/backward/step.
    """

    def __init__(self, device: torch.device, interval: float = 0.01):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.device = torch.device(device)
        self.interval = interval
        self.used = 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.error = None

    def sample(self) -> None:
        used = device_allocated_bytes(self.device)
        with self.lock:
            self.used = max(self.used, used)

    def checkpoint(self) -> None:
        synchronize(self.device)
        self.sample()

    def sample_loop(self) -> None:
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except Exception as exc:
            self.error = exc

    def __enter__(self) -> "PeakMemory":
        gc.collect()
        synchronize(self.device)
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        if self.device.type == "mps":
            self.thread = threading.Thread(target=self.sample_loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.checkpoint()
        finally:
            self.stop.set()
            if self.thread is not None:
                self.thread.join()
            self.rss, self.rss_source = peak_rss()
        if self.error is not None and exc_type is None:
            raise RuntimeError("MPS sampler failed") from self.error
        return False

    def result(self) -> dict:
        accelerator = self.device.type in ("mps", "cuda")
        return {
            "peak_mb": round(self.used / 1024 ** 2, 1),
            "peak_device_mb": round(self.used / 1024 ** 2, 1),
            "peak_rss_mb": round(self.rss / 1024 ** 2, 1),
            "metric": f"аллокатор {self.device.type}" if accelerator else "RSS процесса",
            "metric_source": device_metric_source(self.device),
            "rss_source": self.rss_source,
            "sample_interval_ms": self.interval * 1000 if self.device.type == "mps" else None,
        }
