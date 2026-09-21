"""Нормализация текста и шинглы — общие для очистки, сплита и проверки контаминации.

Один модуль на все три стадии специально: если нормализация разъедется,
дедупликация и проверка контаминации начнут мерить разные вещи, и проверка
станет зелёной при реальном пересечении.
"""

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_DASHES = str.maketrans({"—": "-", "–": "-", "‑": "-", " ": " "})
_ANSWER_OPTION = re.compile(r"^\s*(?:\d+|[a-zа-я])\s*[.)]\s*(\S.*)$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Каноническая форма строки: NFKC, единые тире, схлопнутые пробелы, нижний регистр."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    return _SPACES.sub(" ", text).strip().lower()


def normalize_group(topic: str) -> str:
    """Каноническая форма названия темы — ключ группы для сплита.

    В источнике одна и та же тема встречается в нескольких написаниях:
    «... - 2025» и «... — 2025», плюс склейка алиасов через «|».
    Без нормализации это разные группы, и сплит по группам протекает.
    """
    return normalize_text(topic.split("|")[0])


def _canonicalize_answer_options(text: str) -> str:
    """Сделать порядок нумерованных вариантов несущественным для near-dup.

    Номер варианта не является частью вопроса: при перестановке ответов меняются
    и номера, хотя смысл примера остаётся тем же. Сортируем только блоки из двух
    и более соседних строк вида ``0. ...``/``1) ...``. Обычный прозаический
    текст (в частности, описания фильмов) не переставляется.
    """
    lines = unicodedata.normalize("NFKC", text).translate(_DASHES).splitlines()
    canonical: list[str] = []
    options: list[str] = []

    def flush_options() -> None:
        nonlocal options
        if len(options) >= 2:
            canonical.extend(sorted(options))
        else:
            canonical.extend(options)
        options = []

    for line in lines:
        match = _ANSWER_OPTION.match(line)
        if match:
            options.append(normalize_text(match.group(1)))
            continue
        flush_options()
        normalized = normalize_text(line)
        if normalized:
            canonical.append(normalized)
    flush_options()
    return "\n".join(canonical)


def shingles(text: str, size: int) -> set[str]:
    """Множество словных n-грамм — вход для MinHash."""
    if size <= 0:
        raise ValueError(f"размер шингла должен быть положительным, получено {size}")
    words = re.findall(r"\w+", _canonicalize_answer_options(text).lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}
