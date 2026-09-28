"""Focused checks for the HW4 tokenizer and batching contracts."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from transformers import AutoTokenizer

from src.collate import LABEL_PAD_ID, DynamicPaddingCollator
from src.config import load_params
from src.prompt import build_chat_text, prompt_token_len
from src.tokenize_data import encode_example, mask_prompt, process_split


class HW4TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_name = load_params()["model"]["name"]
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError as exc:
            raise unittest.SkipTest(f"cached tokenizer {model_name} is unavailable: {exc}")

    @staticmethod
    def params(enable_thinking: bool = False, max_seq_len: int = 1024) -> dict:
        return {
            "generate": {"enable_thinking": enable_thinking},
            "tokenize": {"max_seq_len": max_seq_len, "truncated_warn_ratio": 0.05},
            "packing": {"enabled": False},
        }

    @staticmethod
    def dialog(question: str = "Вопрос?", answer: str = "Короткий ответ.") -> list[dict]:
        return [
            {"role": "system", "content": "Отвечай кратко."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    def test_training_supervises_exact_answer_and_eos(self) -> None:
        messages = self.dialog()
        answer = messages[-1]["content"]
        eos = self.tokenizer.eos_token
        for thinking in (False, True):
            with self.subTest(enable_thinking=thinking):
                params = self.params(enable_thinking=thinking)
                raw = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=thinking,
                )
                self.assertTrue(raw.endswith(eos + "\n"))

                train_text = build_chat_text(
                    self.tokenizer, messages, params, add_generation_prompt=False
                )
                self.assertTrue(train_text.endswith(answer + eos))
                self.assertFalse(train_text.endswith(eos + "\n"))

                encoded = encode_example(
                    self.tokenizer,
                    {"id": "answer-only", "messages": messages},
                    params,
                    max_seq_len=1024,
                )
                supervised_ids = [
                    token_id
                    for token_id, label in zip(encoded["input_ids"], encoded["labels"])
                    if label != LABEL_PAD_ID
                ]
                self.assertTrue(supervised_ids)
                self.assertEqual(
                    self.tokenizer.decode(supervised_ids, skip_special_tokens=False),
                    answer + eos,
                )
                self.assertEqual(encoded["_meta"]["supervised"], len(supervised_ids))

    def test_inference_accepts_prompt_only_and_complete_dialog(self) -> None:
        messages = self.dialog()
        for thinking in (False, True):
            with self.subTest(enable_thinking=thinking):
                params = self.params(enable_thinking=thinking)
                from_complete = build_chat_text(
                    self.tokenizer, messages, params, add_generation_prompt=True
                )
                from_prompt = build_chat_text(
                    self.tokenizer, messages[:-1], params, add_generation_prompt=True
                )
                train = build_chat_text(
                    self.tokenizer, messages, params, add_generation_prompt=False
                )
                self.assertEqual(from_complete, from_prompt)
                self.assertTrue(train.encode("utf-8").startswith(from_prompt.encode("utf-8")))

    def test_truncation_drops_examples_without_supervision(self) -> None:
        short = {"id": "short", "topic": "unit", "messages": self.dialog("Кто?", "Я.")}
        long = {
            "id": "long",
            "topic": "unit",
            "messages": self.dialog("Очень длинный вопрос. " * 100, "Ответ."),
        }
        full_long = encode_example(self.tokenizer, long, self.params(), 10000)
        max_seq_len = full_long["_meta"]["prompt_len"]
        self.assertGreater(max_seq_len, 0)
        self.assertLess(
            len(encode_example(self.tokenizer, short, self.params(), 10000)["input_ids"]),
            max_seq_len,
        )
        params = self.params(max_seq_len=max_seq_len)
        clipped = encode_example(self.tokenizer, long, params, max_seq_len)
        self.assertTrue(clipped["_meta"]["truncated"])
        self.assertEqual(clipped["_meta"]["supervised"], 0)
        self.assertTrue(all(label == LABEL_PAD_ID for label in clipped["labels"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.jsonl"
            path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in (short, long))
                + "\n",
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                examples, stats, bins = process_split(self.tokenizer, "train", path, params)
            self.assertTrue(any(item.category is RuntimeWarning for item in caught))

        self.assertEqual([example["id"] for example in examples], ["short"])
        self.assertEqual(stats["examples_in"], 2)
        self.assertEqual(stats["examples_kept"], 1)
        self.assertEqual(stats["dropped_no_supervision"], 1)
        self.assertEqual(stats["truncated"], 1)
        self.assertEqual(stats["truncated_ratio"], 0.5)
        self.assertEqual(stats["supervised_tokens"], sum(
            label != LABEL_PAD_ID for label in examples[0]["labels"]
        ))
        self.assertEqual(bins, [])

    def test_bpe_boundary_masks_crossing_token(self) -> None:
        prompt_text, full_text = "Приве", "Привет, мир"
        encoded = self.tokenizer(
            full_text, add_special_tokens=False, return_offsets_mapping=True
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        n_prompt, used_fallback = prompt_token_len(
            self.tokenizer, prompt_text, input_ids, offsets
        )
        self.assertTrue(used_fallback)
        self.assertGreater(n_prompt, 0)
        self.assertLess(n_prompt, len(input_ids))
        self.assertLess(offsets[n_prompt - 1][0], len(prompt_text))
        self.assertGreater(offsets[n_prompt - 1][1], len(prompt_text))
        self.assertGreaterEqual(offsets[n_prompt][0], len(prompt_text))

        labels = mask_prompt(input_ids, n_prompt)
        self.assertEqual(labels, [LABEL_PAD_ID] * n_prompt + input_ids[n_prompt:])
        supervised = [token for token, label in zip(input_ids, labels) if label != LABEL_PAD_ID]
        self.assertEqual(
            self.tokenizer.decode(supervised), full_text[offsets[n_prompt][0]:]
        )


class HW4CollatorTests(unittest.TestCase):
    def test_dynamic_left_padding_masks_pad_positions(self) -> None:
        collator = DynamicPaddingCollator(pad_token_id=99)
        self.assertEqual(collator.padding_side, "left")
        batch = collator([
            {"input_ids": [10, 11], "attention_mask": [1, 1], "labels": [-100, 11]},
            {"input_ids": [20, 21, 22], "attention_mask": [1, 1, 1], "labels": [-100, 21, 22]},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[99, 10, 11], [20, 21, 22]])
        self.assertEqual(batch["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 11], [-100, 21, 22]])


if __name__ == "__main__":
    unittest.main()
