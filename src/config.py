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
