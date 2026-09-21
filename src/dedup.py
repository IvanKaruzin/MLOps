"""Дедупликация: точная и near-duplicate.

Точная ловит буквальные повторы, near-dup — переформулировки и перестановки
вариантов ответа. В курсовом источнике буквальных повторов нет, а почти-дублей
несколько процентов: одна только точная дедупликация здесь не делает ничего.
"""

from typing import Sequence

from datasketch import MinHash, MinHashLSH

from src.textnorm import shingles


def build_minhash(text: str, shingle_words: int, num_perm: int) -> MinHash:
    if num_perm <= 0:
        raise ValueError(f"num_perm должен быть положительным, получено {num_perm}")
    # Фиксируем seed явно: одинаковый вход обязан давать одинаковые кандидаты
    # независимо от процесса и машины.
    mh = MinHash(num_perm=num_perm, seed=1)
    mh.update_batch([s.encode("utf-8") for s in shingles(text, shingle_words)])
    return mh


def jaccard(left: set[str], right: set[str]) -> float:
    """Фактический Jaccard двух множеств, без MinHash-аппроксимации."""
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _validate_threshold(threshold: float) -> None:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"threshold должен быть в интервале (0, 1], получено {threshold}"
        )


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    """Индексы повторных вхождений. Первое вхождение остаётся."""
    seen: set[str] = set()
    dupes: list[int] = []
    for i, key in enumerate(keys):
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


def near_duplicates(
    texts: Sequence[str], shingle_words: int, num_perm: int, threshold: float
) -> list[int]:
    """Индексы почти-дублей: жадный проход, первый представитель кластера остаётся.

    MinHash + LSH дают линейное время вместо O(n^2) полного попарного сравнения.
    """
    _validate_threshold(threshold)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    dupes: list[int] = []
    shingle_sets: dict[int, set[str]] = {}
    for i, text in enumerate(texts):
        current = shingles(text, shingle_words)
        mh = build_minhash(text, shingle_words, num_perm)
        # LSH только дешёво отбирает кандидатов. Решение об удалении принимает
        # реальный Jaccard; иначе MinHash-коллизия может удалить не-дубликат.
        candidates = sorted(int(key) for key in lsh.query(mh))
        if any(jaccard(current, shingle_sets[key]) >= threshold for key in candidates):
            dupes.append(i)
        else:
            lsh.insert(str(i), mh)
            shingle_sets[i] = current
    return dupes


def cross_near_duplicates(
    left: Sequence[str],
    right: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[tuple[int, int]]:
    """Пары (индекс в left, индекс в right) с Жаккаром выше порога.

    Используется проверкой контаминации: точное совпадение текстов ловит
    копипасту, а протекают обычно парафразы.
    """
    _validate_threshold(threshold)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    left_shingles: list[set[str]] = []
    for i, text in enumerate(left):
        left_shingles.append(shingles(text, shingle_words))
        lsh.insert(str(i), build_minhash(text, shingle_words, num_perm))
    pairs: list[tuple[int, int]] = []
    for j, text in enumerate(right):
        current = shingles(text, shingle_words)
        candidates = sorted(
            int(key)
            for key in lsh.query(build_minhash(text, shingle_words, num_perm))
        )
        for i in candidates:
            if jaccard(left_shingles[i], current) >= threshold:
                pairs.append((i, j))
    return pairs
