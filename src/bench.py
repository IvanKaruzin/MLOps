"""Замер производительности машины на выбранной модели.

Три числа меряются РАЗДЕЛЬНО — смешивать их бессмысленно:
  * время загрузки модели  — разовая стоимость старта;
  * tokens/sec             — скорость генерации, только после прогрева;
  * пик памяти устройства и RSS — отдельные метрики с именами источников.
"""

import json
import statistics
import time
from pathlib import Path

from src.config import load_params
from src.memory import PeakMemory, resolve_device, synchronize
from src.model import generate, load_model, set_seed


def main() -> None:
    params = load_params()
    prompt = params["bench"]["prompt"]
    set_seed(params["generate"]["seed"])

    device = resolve_device(params)
    params["model"]["device"] = str(device)
    with PeakMemory(device) as peak:
        t0 = time.perf_counter()
        tokenizer, model = load_model(params)
        synchronize(device)
        load_time = time.perf_counter() - t0

        for _ in range(params["bench"]["warmup_runs"]):
            generate(tokenizer, model, params, prompt)
            synchronize(device)

        speeds = []
        for _ in range(params["bench"]["measure_runs"]):
            synchronize(device)
            t0 = time.perf_counter()
            _, n_tokens = generate(tokenizer, model, params, prompt)
            synchronize(device)
            elapsed = time.perf_counter() - t0
            speeds.append(n_tokens / elapsed)

    # Медиана устойчивее среднего к одиночному выбросу.
    report = {
        "model": params["model"]["name"],
        "device": str(device),
        "dtype": params["model"]["dtype"],
        "load_time_sec": round(load_time, 2),
        "tokens_per_sec": round(statistics.median(speeds), 2),
        "tokens_per_sec_all": [round(s, 2) for s in speeds],
        **peak.result(),
        "weights_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2, 1),
    }

    assert report["peak_mb"] >= report["weights_mb"]

    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
