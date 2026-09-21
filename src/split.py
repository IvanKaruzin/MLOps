"""Детерминированный групповой split по году выпуска фильма."""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.config import load_params
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group, normalize_text

SPLIT_NAMES = ("train", "val", "test")
_YEAR_GROUP = re.compile(r"release_year:\d{4}")


def validate_ratios(ratios: dict[str, float]) -> dict[str, float]:
    """Проверить и нормализовать контракт долей train/val/test."""
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(
            "split.ratios должен содержать ровно train, val и test; "
            f"получено {sorted(ratios)}"
        )
    normalized: dict[str, float] = {}
    for name in SPLIT_NAMES:
        value = ratios[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"split.ratios.{name} должен быть числом, получено {value!r}")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"split.ratios.{name} должен быть положительным конечным числом, "
                f"получено {value!r}"
            )
        normalized[name] = value
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"сумма split.ratios должна быть 1.0, получено {total:.12g}")
    return normalized


def genres(example: Example) -> set[str]:
    """Жанровые метки примера в канонической форме."""
    return {
        normalize_text(label)
        for label in example.assistant.split(",")
        if normalize_text(label)
    }


def _group_examples(examples: Sequence[Example]) -> dict[str, list[Example]]:
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        key = normalize_group(example.topic)
        if not _YEAR_GROUP.fullmatch(key):
            raise ValueError(
                "topic должен иметь вид release_year:YYYY; "
                f"для id={example.id!r} получено {example.topic!r}"
            )
        grouped[key].append(example)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError(
            "для положительных train/val/test нужно минимум три группы, "
            f"получено {len(grouped)}"
        )
    return {key: sorted(rows, key=lambda row: row.id) for key, rows in grouped.items()}


def _genre_counts(rows: Iterable[Example]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(genres(row))
    return counts


def _coverage_anchors(grouped: dict[str, list[Example]]) -> list[str]:
    """Выбрать детерминированный набор годов, покрывающий все жанры train.

    Это не надежда на удачный shuffle: даже жанр, встречающийся ровно в одном
    году, получает жёсткую гарантию оказаться в train.
    """
    group_genres = {
        key: set(_genre_counts(rows)) for key, rows in sorted(grouped.items())
    }
    uncovered = set().union(*group_genres.values())
    if not uncovered:
        raise ValueError("ни в одном примере нет жанровых меток")
    anchors: list[str] = []
    while uncovered:
        key = min(
            group_genres,
            key=lambda candidate: (
                -len(group_genres[candidate] & uncovered),
                len(grouped[candidate]),
                candidate,
            ),
        )
        newly_covered = group_genres[key] & uncovered
        if not newly_covered:
            raise ValueError(f"не удалось покрыть жанры train: {sorted(uncovered)}")
        anchors.append(key)
        uncovered -= newly_covered
    return anchors


def _distribution(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in sorted(counts.items())}


def _drift(
    buckets: dict[str, list[Example]],
) -> tuple[dict[str, object], dict[str, list[str]]]:
    counts = {name: _genre_counts(buckets[name]) for name in SPLIT_NAMES}
    distributions = {name: _distribution(counts[name]) for name in SPLIT_NAMES}
    all_genres = sorted(set().union(*(set(value) for value in counts.values())))
    train_genres = set(counts["train"])
    missing = {
        name: sorted(set(counts[name]) - train_genres) for name in ("val", "test")
    }

    comparisons: dict[str, dict[str, object]] = {}
    train_size = len(buckets["train"])
    for name in ("val", "test"):
        per_genre_delta = {
            label: round(
                counts[name].get(label, 0) / len(buckets[name])
                - counts["train"].get(label, 0) / train_size,
                6,
            )
            for label in all_genres
        }
        total_variation = 0.5 * sum(
            abs(distributions[name].get(label, 0.0) - distributions["train"].get(label, 0.0))
            for label in all_genres
        )
        largest_label = max(per_genre_delta, key=lambda label: abs(per_genre_delta[label]))
        comparisons[name] = {
            "total_variation": round(total_variation, 6),
            "max_prevalence_delta": round(abs(per_genre_delta[largest_label]), 6),
            "largest_delta_genre": largest_label,
            "prevalence_delta": per_genre_delta,
        }

    return (
        {
            "genre_counts": {
                name: dict(sorted(split_counts.items()))
                for name, split_counts in counts.items()
            },
            "comparisons_to_train": comparisons,
            "max_total_variation": max(
                comparison["total_variation"] for comparison in comparisons.values()
            ),
        },
        missing,
    )


def _candidate(
    grouped: dict[str, list[Example]],
    ratios: dict[str, float],
    seed: int,
    attempt: int,
    train_anchors: Sequence[str],
) -> dict[str, list[Example]]:
    """Построить один вариант, всегда назначая группу целиком."""
    anchor_set = set(train_anchors)
    group_keys = sorted(set(grouped) - anchor_set)
    random.Random(seed + attempt * 104_729).shuffle(group_keys)
    total_rows = sum(len(rows) for rows in grouped.values())
    targets = {name: total_rows * ratios[name] for name in SPLIT_NAMES}
    sizes = {
        "train": sum(len(grouped[key]) for key in train_anchors),
        "val": 0,
        "test": 0,
    }
    assigned: dict[str, list[str]] = {
        "train": list(train_anchors),
        "val": [],
        "test": [],
    }

    for position, key in enumerate(group_keys):
        groups_left = len(group_keys) - position
        empty = [name for name in SPLIT_NAMES if not assigned[name]]
        choices = empty if groups_left == len(empty) else list(SPLIT_NAMES)

        def row_error(name: str) -> tuple[float, int]:
            projected = dict(sizes)
            projected[name] += len(grouped[key])
            error = sum(
                ((projected[split] - targets[split]) / targets[split]) ** 2
                for split in SPLIT_NAMES
            )
            return error, SPLIT_NAMES.index(name)

        destination = min(choices, key=row_error)
        assigned[destination].append(key)
        sizes[destination] += len(grouped[key])

    return {
        name: [row for key in sorted(assigned[name]) for row in grouped[key]]
        for name in SPLIT_NAMES
    }


def _candidate_score(
    buckets: dict[str, list[Example]], ratios: dict[str, float]
) -> tuple[float, dict[str, object], dict[str, list[str]]]:
    total = sum(len(rows) for rows in buckets.values())
    ratio_error = sum(
        abs(len(buckets[name]) - total * ratios[name]) for name in SPLIT_NAMES
    ) / total
    drift, missing = _drift(buckets)
    missing_count = sum(len(labels) for labels in missing.values())
    drift_sum = sum(
        comparison["total_variation"]
        for comparison in drift["comparisons_to_train"].values()
    )
    # Покрытие — жёсткий инвариант. Среди допустимых вариантов сначала держим
    # близкие размеры, затем выбираем наименее сдвинутые распределения жанров.
    score = missing_count * 1_000.0 + ratio_error * 4.0 + drift_sum
    return score, drift, missing


def grouped_split(
    examples: Sequence[Example],
    ratios: dict[str, float],
    seed: int,
    search_attempts: int = 512,
) -> tuple[dict[str, list[Example]], dict[str, object]]:
    """Выбрать лучший из детерминированной последовательности назначений групп."""
    ratios = validate_ratios(ratios)
    if (
        isinstance(search_attempts, bool)
        or not isinstance(search_attempts, int)
        or search_attempts <= 0
    ):
        raise ValueError(
            "split.search_attempts должен быть положительным целым, "
            f"получено {search_attempts!r}"
        )
    grouped = _group_examples(examples)
    train_anchors = _coverage_anchors(grouped)
    if len(grouped) - len(train_anchors) < 2:
        raise ValueError(
            "после обязательного покрытия жанров в train не остаётся двух групп "
            "для val/test"
        )
    best: tuple[
        float,
        int,
        dict[str, list[Example]],
        dict[str, object],
        dict[str, list[str]],
    ] | None = None
    baseline: dict[str, object] | None = None
    for attempt in range(search_attempts):
        buckets = _candidate(grouped, ratios, seed, attempt, train_anchors)
        score, drift, missing = _candidate_score(buckets, ratios)
        if attempt == 0:
            baseline = {
                "score": round(score, 6),
                "sizes": {name: len(buckets[name]) for name in SPLIT_NAMES},
                "max_total_variation": drift["max_total_variation"],
                "missing_train_genres": missing,
            }
        current = (score, attempt, buckets, drift, missing)
        if best is None or current[:2] < best[:2]:
            best = current

    assert best is not None and baseline is not None
    score, attempt, buckets, drift, missing = best
    if any(missing.values()):
        raise ValueError(
            "не удалось обеспечить покрытие жанров в train: "
            + "; ".join(f"{name}={labels}" for name, labels in missing.items() if labels)
        )
    return buckets, {
        "search_attempts": search_attempts,
        "train_coverage_anchor_groups": train_anchors,
        "selected_attempt": attempt,
        "selected_score": round(score, 6),
        "baseline": baseline,
        "drift": drift,
        "genre_coverage": {
            "passed": True,
            "missing_train_genres": missing,
        },
    }


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if not examples:
        raise SystemExit("split: очищенный датасет пуст")
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    try:
        buckets, audit = grouped_split(
            examples,
            cfg["ratios"],
            cfg["seed"],
            cfg["search_attempts"],
        )
    except ValueError as exc:
        raise SystemExit(f"split: {exc}") from exc

    max_drift = float(audit["drift"]["max_total_variation"])
    max_allowed_drift = float(cfg["max_genre_total_variation"])
    drift_passed = max_drift <= max_allowed_drift
    audit["drift"]["max_allowed_total_variation"] = max_allowed_drift
    audit["drift"]["passed"] = drift_passed
    audit["drift"]["baseline_was_strong"] = (
        float(audit["baseline"]["max_total_variation"]) > max_allowed_drift
    )

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for example in rows:
                fh.write(dump(example) + "\n")

    total = len(examples)
    group_sets = {
        name: {normalize_group(example.topic) for example in rows}
        for name, rows in buckets.items()
    }
    group_overlap = {
        f"{left}_{right}": sorted(group_sets[left] & group_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(set().union(*group_sets.values())),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {name: len(groups) for name, groups in group_sets.items()},
        "ratios_target": validate_ratios(cfg["ratios"]),
        "ratios_actual": {
            name: round(len(rows) / total, 6) for name, rows in buckets.items()
        },
        "group_overlap": group_overlap,
        "optimization": audit,
        "passed": drift_passed and not any(group_overlap.values()),
        "seconds": round(time.perf_counter() - started, 2),
    }
    metrics_path = Path(paths["metrics_split"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not metrics["passed"]:
        reasons = []
        if not drift_passed:
            reasons.append(
                f"genre drift {max_drift:.4f} выше порога {max_allowed_drift:.4f}"
            )
        if any(group_overlap.values()):
            reasons.append("группы пересекаются между частями")
        raise SystemExit("split: " + "; ".join(reasons))

    print(
        "split: "
        + ", ".join(
            f"{name} {len(rows)} ({len(group_sets[name])} групп)"
            for name, rows in buckets.items()
        )
        + f"; max genre TV={max_drift:.4f} ({metrics['seconds']} с)"
    )


if __name__ == "__main__":
    main()
