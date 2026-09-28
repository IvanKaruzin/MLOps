"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def validate_params(params: dict) -> None:
    """Проверить общие инварианты конфигурации до запуска стадий."""
    clean_threshold = params["clean"]["near_dup"]["threshold"]
    contamination_threshold = params["contamination"]["threshold"]
    if clean_threshold != contamination_threshold:
        raise ValueError(
            "clean.near_dup.threshold и contamination.threshold должны "
            f"совпадать: {clean_threshold!r} != {contamination_threshold!r}"
        )

    data = params["data"]
    paths = params["paths"]
    for data_key, path_key in (("train_jsonl", "train"), ("val_jsonl", "val")):
        if data[data_key] != paths[path_key]:
            raise ValueError(
                f"data.{data_key} и paths.{path_key} должны совпадать: "
                f"{data[data_key]!r} != {paths[path_key]!r}"
            )
    tokenize = params["tokenize"]
    if tokenize["max_seq_len"] <= 0:
        raise ValueError("tokenize.max_seq_len должен быть положительным")
    if not 0 <= tokenize["truncated_warn_ratio"] <= 1:
        raise ValueError("tokenize.truncated_warn_ratio должен быть от 0 до 1")
    if tokenize["padding_side"] != "left":
        raise ValueError("для decoder-only tokenize.padding_side должен быть left")
    if params["train_estimate"]["tokens_per_sec"] <= 0:
        raise ValueError("train_estimate.tokens_per_sec должен быть положительным")


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_params(params)
    return params


def source_files(params: dict, *, require_exists: bool = True) -> list[Path]:
    """Вернуть файлы-источники для выбранной версии датасета.

    Версия хранится в ``params.yaml``, чтобы DVC мог учитывать её при
    инвалидации стадии ``collect``.
    """
    version = params["collect"]["version"]
    sources = params["collect"]["sources"]
    if version not in sources:
        raise SystemExit(
            f"collect.version = {version!r}, но в collect.sources "
            f"есть только {sorted(sources)}"
        )

    files = [Path(path) for path in sources[version]]
    missing = [path for path in files if not path.exists()]
    if require_exists and missing:
        raise SystemExit(
            "стадия collect не нашла файл-источник:\n  "
            + "\n  ".join(str(path) for path in missing)
            + "\n\nЗагрузите зафиксированный snapshot выбранного источника "
            "и проверьте collect.sources в params.yaml."
        )
    return files
