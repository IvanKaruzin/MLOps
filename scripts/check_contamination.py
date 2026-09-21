#!/usr/bin/env python3
"""Проверка контаминации train/test как отдельный запускаемый гейт.

Стадия split уже проверяет себя, но проверка обязана существовать отдельно:
сплит могли собрать руками, получить от соседа или откатить через dvc checkout.
Возвращает 1 при любом пересечении — годится для CI.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402
from src.contamination import is_clean, report  # noqa: E402
from src.schema import iter_examples  # noqa: E402


def main() -> int:
    started = time.perf_counter()
    params = load_params()
    paths = params["paths"]
    nd = params["clean"]["near_dup"]

    train = list(iter_examples(paths["train"]))
    test = list(iter_examples(paths["test"]))
    rep = report(
        train,
        test,
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )
    metrics = {
        "version": params["collect"]["version"],
        "train_rows": len(train),
        "test_rows": len(test),
        "threshold": params["contamination"]["threshold"],
        "shingle_words": nd["shingle_words"],
        "num_perm": nd["num_perm"],
        **rep,
        "passed": is_clean(rep),
        "seconds": round(time.perf_counter() - started, 2),
    }
    metrics_path = Path(paths["metrics_contamination"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Сначала перезаписываем метрику, затем возвращаем ненулевой код. Так при
    # падении CI не останется старый зелёный отчёт от предыдущего запуска.
    temporary_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(metrics_path)

    print(f"train: {len(train)} строк, test: {len(test)} строк")
    print(f"  пересечение по id:        {rep['id_overlap']}")
    print(f"  пересечение по тексту:    {rep['text_overlap']}")
    print(f"  пересечение по группам:   {rep['group_overlap']}")
    print(f"  near-dup пар train↔test:  {rep['near_dup_pairs']}")

    if metrics["passed"]:
        print("контаминации нет")
        return 0

    for kind, items in rep["examples"].items():
        if items:
            print(f"  примеры ({kind}): {items}")
    print("КОНТАМИНАЦИЯ: train и test пересекаются, метрики на test завышены")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
