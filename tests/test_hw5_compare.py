"""Офлайн-проверки общего промпта, переносимого инференса и полных кривых."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from transformers import BatchEncoding

from src.compare import generate, inputs_fingerprint, load_adapter_tokenizer, render_comparison, validate_compare
from src.plot import plot_runs, validate_run
from src.prompt import build_chat_text


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    chat_template = "test-template"

    def __init__(self):
        self.encoded = []
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "<bos>" + "/".join(message["content"] for message in messages) + "<assistant>"

    def __call__(self, text, **kwargs):
        self.encoded.append((text, kwargs))
        return BatchEncoding({"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones(1, 2)})

    def decode(self, ids, **kwargs):
        if ids.tolist() != [12, 13]:
            raise AssertionError("декодирован промпт вместо одного ответа")
        return "  драма, триллер  "


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "train": {"seed": 67},
            "generate": {"enable_thinking": False},
            "model": {"name": "cached-base", "device": "cpu", "dtype": "float32"},
            "compare": {"max_new_tokens": 40, "system": "Только жанры.", "prompts": [str(i) for i in range(5)]},
        }

    def test_generation_uses_training_builder_without_duplicate_special_tokens(self):
        tokenizer = RecordingTokenizer()
        model = Mock()
        model.generate.return_value = torch.tensor([[10, 11, 12, 13]])
        question = "A detective searches for a missing child."
        answers = generate(model, tokenizer, [question], "Только жанры.", self.params, torch.device("cpu"))
        reference = build_chat_text(
            tokenizer,
            [{"role": "system", "content": "Только жанры."}, {"role": "user", "content": question}],
            self.params,
            add_generation_prompt=True,
        )
        self.assertEqual(tokenizer.encoded[0][0], reference)
        self.assertFalse(tokenizer.encoded[0][1]["add_special_tokens"])
        self.assertFalse(tokenizer.template_calls[0][1]["enable_thinking"])
        self.assertEqual(answers, ["драма, триллер"])
        model.eval.assert_called_once()
        generation = model.generate.call_args.kwargs
        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 1)

    def test_tokenizer_is_loaded_only_from_adapter_offline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            tokenizer = RecordingTokenizer()
            with patch("src.compare.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
                self.assertIs(load_adapter_tokenizer(adapter_dir), tokenizer)
            load.assert_called_once_with(adapter_dir, local_files_only=True)

    def test_adapter_without_template_or_pad_is_rejected(self):
        for attribute, value in (("chat_template", None), ("pad_token_id", None)):
            tokenizer = RecordingTokenizer()
            setattr(tokenizer, attribute, value)
            with self.subTest(attribute=attribute), patch(
                "src.compare.AutoTokenizer.from_pretrained", return_value=tokenizer
            ), self.assertRaises(ValueError):
                load_adapter_tokenizer(Path("adapter"))

    def test_exactly_five_distinct_questions_required(self):
        validate_compare(self.params)
        for prompts in (["one"] * 5, ["one"], ["0", "1", "2", "3", " "]):
            params = copy.deepcopy(self.params)
            params["compare"]["prompts"] = prompts
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                validate_compare(params)

    def test_report_rejects_silent_zip_truncation(self):
        with self.assertRaisesRegex(ValueError, "число ответов"):
            render_comparison({"prompts": ["one", "two"], "base": ["base"], "adapter": ["a", "b"]}, 40)

    def test_fingerprint_detects_prompt_and_weight_changes_but_not_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Path(tmpdir) / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"first-weights")
            first = inputs_fingerprint(self.params, adapter)
            moved = Path(tmpdir) / "moved"
            adapter.rename(moved)
            self.assertEqual(first, inputs_fingerprint(self.params, moved))
            params = copy.deepcopy(self.params)
            params["compare"]["prompts"][0] = "different-movie"
            self.assertNotEqual(first, inputs_fingerprint(params, moved))
            (moved / weights.name).write_bytes(b"second-weights")
            self.assertNotEqual(first, inputs_fingerprint(self.params, moved))


class CurveTests(unittest.TestCase):
    @staticmethod
    def run_metrics(variant="all_layers"):
        return {
            "variant": variant,
            "trainable_share": 0.0084,
            "seconds": 12.0,
            "curve_train": [[1, 1.0], [2, 0.5]],
            "curve_val": [[0, 1.2], [1, 0.8], [2, 0.6]],
        }

    def test_plot_contains_train_and_val_of_both_variants(self):
        figure = plot_runs([self.run_metrics(), self.run_metrics("freeze14")])
        try:
            for axis in figure.axes:
                self.assertEqual(len(axis.lines), 2)
                self.assertIsNotNone(axis.get_legend())
            self.assertEqual(list(figure.axes[1].lines[0].get_xdata()), [0, 1, 2])
            with tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / "curves.png"
                figure.savefig(output)
                self.assertGreater(output.stat().st_size, 1000)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_missing_base_final_or_nonfinite_loss_is_rejected(self):
        for field, curve in (
            ("curve_val", []),
            ("curve_val", [[1, 0.8], [2, 0.6]]),
            ("curve_val", [[0, 1.2], [1, 0.8]]),
            ("curve_train", [[1, float("nan")], [2, 0.5]]),
            ("curve_train", [[2, 1.0], [1, 0.5]]),
        ):
            run = self.run_metrics()
            run[field] = curve
            with self.subTest(field=field, curve=curve), self.assertRaises(ValueError):
                validate_run(run)


if __name__ == "__main__":
    unittest.main()
