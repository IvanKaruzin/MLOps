"""Кривые train и val обоих LoRA-вариантов → docs/curves.png."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_params  # noqa: E402

COLORS = {"all_layers": "#7e22ce", "freeze14": "#0e7490"}


def validate_run(run: dict) -> None:
    """Пустые или устаревшие кривые не должны выглядеть как готовый отчёт."""
    for name in ("curve_train", "curve_val"):
        curve = run.get(name)
        if not curve:
            raise ValueError(f"{run['variant']}: нет {name} — сначала выполните обучение")
        last_step = -1
        for step, loss in curve:
            if step <= last_step or not math.isfinite(loss) or loss < 0:
                raise ValueError(f"{run['variant']}: некорректная точка {name}: {[step, loss]}")
            last_step = step
    if run["curve_val"][0][0] != 0:
        raise ValueError(f"{run['variant']}: нет val loss до первого шага")
    if run["curve_train"][-1][0] != run["curve_val"][-1][0]:
        raise ValueError(f"{run['variant']}: нет val loss на последнем шаге обучения")


def plot_runs(runs: list[dict]):
    for run in runs:
        validate_run(run)
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for run in runs:
        color = COLORS.get(run["variant"], "#444444")
        label = f"{run['variant']} ({run['trainable_share']:.2%} параметров, {run['seconds']:.0f} с)"
        ax_train.plot(*zip(*run["curve_train"]), color=color, alpha=0.85, label=label)
        ax_val.plot(*zip(*run["curve_val"]), color=color, marker="o", label=run["variant"])
    ax_train.set_title("train loss (по шагам оптимизатора)")
    ax_val.set_title("val loss (шаг 0 — базовая модель)")
    for axis in (ax_train, ax_val):
        axis.set_xlabel("шаг")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    ax_train.set_ylabel("loss на токен ответа")
    fig.tight_layout()
    return fig


def main() -> None:
    params = load_params()
    metrics_dir = Path(params["paths"]["metrics"])
    from src.train import inputs_fingerprint

    fingerprint = inputs_fingerprint(params)
    runs = []
    # Не включаем случайные smoke-файлы: отчёт относится к двум вариантам конфига.
    for variant in params["variants"]:
        path = metrics_dir / f"train_{variant['name']}.json"
        if not path.exists():
            raise SystemExit(f"нет {path} — сначала make train")
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("variant") != variant["name"] or run.get("inputs_fingerprint") != fingerprint:
            raise SystemExit(f"{path}: метрики не соответствуют текущему коду/конфигу — повторите make train")
        runs.append(run)
    figure = plot_runs(runs)
    output = Path(params["paths"]["curves"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print(f"-> {output}")


if __name__ == "__main__":
    main()
