"""Диагностика потенциального выигрыша от packing.

Бины не являются готовым обучающим датасетом: у обычной причинной маски
каждый пример видит предыдущий. Для обучения на бинах понадобится отдельная
блочно-диагональная маска внимания.
"""


def pack_examples(examples: list[dict], max_seq_len: int) -> list[dict]:
    """Разложить примеры по длине для оценки числа бинов."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len должен быть положительным")
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i]["input_ids"]))
    bins: list[dict] = []
    for i in order:
        example = examples[i]
        size = len(example["input_ids"])
        if size > max_seq_len:
            raise ValueError(f"пример {i} длиннее max_seq_len")
        if not (size == len(example["attention_mask"]) == len(example["labels"])):
            raise ValueError(f"у примера {i} расходятся длины полей")
        for candidate in bins:
            if len(candidate["input_ids"]) + size <= max_seq_len:
                target = candidate
                break
        else:
            target = {"input_ids": [], "attention_mask": [], "labels": [], "seq_lens": []}
            bins.append(target)
        target["input_ids"].extend(example["input_ids"])
        target["attention_mask"].extend(example["attention_mask"])
        target["labels"].extend(example["labels"])
        target["seq_lens"].append(size)
    return bins


def packing_report(examples: list[dict], bins: list[dict], max_seq_len: int, batch_size: int) -> dict:
    """Теоретический выигрыш по шагам при размере батча ``batch_size``."""
    if max_seq_len <= 0 or batch_size <= 0:
        raise ValueError("max_seq_len и batch_size должны быть положительными")
    steps_before = -(-len(examples) // batch_size)
    steps_after = -(-len(bins) // batch_size)
    used = sum(len(bin_["input_ids"]) for bin_ in bins)
    return {
        "sequences_before": len(examples),
        "sequences_after": len(bins),
        "batch_size": batch_size,
        "steps_before": steps_before,
        "steps_after": steps_after,
        "steps_saved_ratio": round(1 - steps_after / steps_before, 4) if steps_before else 0.0,
        "fill_ratio": round(used / (len(bins) * max_seq_len), 4) if bins else 0.0,
        "max_examples_per_bin": max((len(bin_["seq_lens"]) for bin_ in bins), default=0),
        "diagnostic_only": True,
    }
