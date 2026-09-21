"""Проверка контаминации train/test — общий код для стадии split и для скрипта.

Три уровня, каждый ловит своё:
  1. id      — та же строка попала в оба сплита;
  2. текст   — тот же вопрос под другим id (копипаста источника);
  3. near-dup— парафраз: переставленные варианты ответа, другой порядок слов.
Случайный сплит валится обычно на третьем: первые два он проходит.
"""

from collections import defaultdict
from typing import Sequence

from src.dedup import cross_near_duplicates, jaccard
from src.schema import Example
from src.textnorm import normalize_group, normalize_text, shingles

DIAGNOSTIC_LIMIT = 3


def report(
    train: Sequence[Example],
    test: Sequence[Example],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> dict[str, object]:
    """Сводка пересечений train/test. Ноль по всем ключам — сплит честный."""
    train_ids = {ex.id for ex in train}
    test_ids = {ex.id for ex in test}
    id_overlap = sorted(train_ids & test_ids)

    train_texts = [ex.user for ex in train]
    test_texts = [ex.user for ex in test]
    train_by_text: dict[str, list[str]] = defaultdict(list)
    test_by_text: dict[str, list[str]] = defaultdict(list)
    for example in train:
        train_by_text[normalize_text(example.user)].append(example.id)
    for example in test:
        test_by_text[normalize_text(example.user)].append(example.id)
    text_overlap = sorted(set(train_by_text) & set(test_by_text))

    train_by_group: dict[str, list[str]] = defaultdict(list)
    test_by_group: dict[str, list[str]] = defaultdict(list)
    for example in train:
        train_by_group[normalize_group(example.topic)].append(example.id)
    for example in test:
        test_by_group[normalize_group(example.topic)].append(example.id)
    group_overlap = sorted(set(train_by_group) & set(test_by_group))

    pairs = cross_near_duplicates(
        train_texts, test_texts, shingle_words=shingle_words, num_perm=num_perm, threshold=threshold
    )

    return {
        "id_overlap": len(id_overlap),
        "text_overlap": len(text_overlap),
        "group_overlap": len(group_overlap),
        "near_dup_pairs": len(pairs),
        "examples": {
            "id": [{"id": value} for value in id_overlap[:DIAGNOSTIC_LIMIT]],
            "text": [
                {
                    "train_ids": sorted(train_by_text[value])[:DIAGNOSTIC_LIMIT],
                    "test_ids": sorted(test_by_text[value])[:DIAGNOSTIC_LIMIT],
                    "normalized_preview": value[:160],
                }
                for value in text_overlap[:DIAGNOSTIC_LIMIT]
            ],
            "group": [
                {
                    "group": value,
                    "train_ids": sorted(train_by_group[value])[:DIAGNOSTIC_LIMIT],
                    "test_ids": sorted(test_by_group[value])[:DIAGNOSTIC_LIMIT],
                }
                for value in group_overlap[:DIAGNOSTIC_LIMIT]
            ],
            "near_dup": [
                {
                    "train_id": train[i].id,
                    "test_id": test[j].id,
                    "jaccard": round(
                        jaccard(
                            shingles(train[i].user, shingle_words),
                            shingles(test[j].user, shingle_words),
                        ),
                        6,
                    ),
                    "train_preview": normalize_text(train[i].user)[:120],
                    "test_preview": normalize_text(test[j].user)[:120],
                }
                for i, j in pairs[:DIAGNOSTIC_LIMIT]
            ],
        },
    }


def is_clean(rep: dict) -> bool:
    return all(rep[k] == 0 for k in ("id_overlap", "text_overlap", "group_overlap", "near_dup_pairs"))
