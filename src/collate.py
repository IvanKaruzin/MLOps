"""Динамический левый паддинг батча для decoder-only модели."""

import torch

LABEL_PAD_ID = -100


class DynamicPaddingCollator:
    """Собрать примеры в тензоры, маскируя паддинг в labels."""

    def __init__(self, pad_token_id: int, padding_side: str = "left") -> None:
        if padding_side not in ("left", "right"):
            raise ValueError(f"padding_side должен быть left или right, получено {padding_side!r}")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def _pad(self, seq: list[int], width: int, value: int) -> list[int]:
        padding = [value] * (width - len(seq))
        return padding + seq if self.padding_side == "left" else seq + padding

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("нельзя собрать пустой батч")
        width = max(len(feature["input_ids"]) for feature in features)
        for feature in features:
            if not (len(feature["input_ids"]) == len(feature["attention_mask"]) == len(feature["labels"])):
                raise ValueError("input_ids, attention_mask и labels должны быть одной длины")
        batch = {
            "input_ids": [self._pad(f["input_ids"], width, self.pad_token_id) for f in features],
            "attention_mask": [self._pad(f["attention_mask"], width, 0) for f in features],
            "labels": [self._pad(f["labels"], width, LABEL_PAD_ID) for f in features],
        }
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
