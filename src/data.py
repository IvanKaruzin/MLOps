"""Validated, safe tensor inputs from HW4; no second tokenization."""

import random
from pathlib import Path

import torch

from src.collate import LABEL_PAD_ID, DynamicPaddingCollator


def load_split(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Нет {p}: сначала выполните стадию tokenize ДЗ4")
    blob = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not isinstance(blob.get("examples"), list) or not blob["examples"]:
        raise ValueError(f"{p}: ожидается непустой список examples")
    pad_id = blob.get("pad_token_id")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError(f"{p}: некорректный pad_token_id")
    if blob.get("padding_side", "left") != "left":
        raise ValueError(f"{p}: ожидается левый паддинг из ДЗ4")
    for i, example in enumerate(blob["examples"]):
        if not isinstance(example, dict):
            raise ValueError(f"{p}: пример {i} не является словарём")
        sequences = [example.get(key) for key in ("input_ids", "attention_mask", "labels")]
        if any(not isinstance(seq, list) or any(type(x) is not int for x in seq) for seq in sequences):
            raise ValueError(f"{p}: пример {i}: нужны списки целых чисел")
        ids, attention, labels = sequences
        if len(ids) < 2 or len(ids) != len(attention) or len(ids) != len(labels):
            raise ValueError(f"{p}: пример {i}: некорректные длины")
        if any(x < 0 for x in ids) or any(x not in (0, 1) for x in attention):
            raise ValueError(f"{p}: пример {i}: некорректные токены или attention_mask")
        if labels[0] != LABEL_PAD_ID or not any(x != LABEL_PAD_ID for x in labels[1:]):
            raise ValueError(f"{p}: пример {i}: нет корректной causal-supervision")
        if any(label != LABEL_PAD_ID and (label != token or not mask)
               for token, mask, label in zip(ids, attention, labels)):
            raise ValueError(f"{p}: пример {i}: labels не соответствуют input_ids/маске")
    return blob


def pad_batch(features: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    return DynamicPaddingCollator(pad_id, padding_side="left")(features)


def batches(examples: list[dict], batch_size: int, pad_id: int, shuffle: bool, seed: int):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size должен быть положительным целым")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield pad_batch([examples[j] for j in order[i:i + batch_size]], pad_id)
