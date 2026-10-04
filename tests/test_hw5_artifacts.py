"""Проверки против ложноположительного сравнения и короткого обучения."""

import copy
import unittest

from scripts.check_hw5_artifacts import validate_comparison, validate_training


class ArtifactChecksTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "model": {"name": "local-base"},
            "generate": {"enable_thinking": False},
            "compare": {"prompts": [str(i) for i in range(5)], "system": "жанры", "max_new_tokens": 40},
        }
        self.result = {
            "model": "local-base", "enable_thinking": False, "max_new_tokens": 40,
            "system": "жанры", "prompts": self.params["compare"]["prompts"],
            "base": ["драма"] * 5, "adapter": ["драма, триллер"] * 5, "inputs_fingerprint": "current",
        }

    def test_short_answers_are_rejected_instead_of_truncated_zip(self):
        validate_comparison(self.result, self.params, "current")
        for key in ("base", "adapter", "prompts"):
            result = copy.deepcopy(self.result)
            result[key] = result[key][:4]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "пять"):
                validate_comparison(result, self.params, "current")

    def test_stale_comparison_or_changed_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "устаревший"):
            validate_comparison(self.result, self.params, "new-code")
        result = copy.deepcopy(self.result)
        result["prompts"][0] = "other-question"
        with self.assertRaisesRegex(ValueError, "вопросы"):
            validate_comparison(result, self.params, "current")

    def test_cli_short_training_cannot_pass_full_epoch_check(self):
        params = {"train": {"max_steps": None, "eval_every": 10, "batch_size": 2, "grad_accum": 4, "epochs": 1}}
        metrics = {"variant": "all_layers", "inputs_fingerprint": "current", "validation_limited": False,
                   "validation_examples": 5, "steps": 3, "planned_steps": 3}
        with self.assertRaisesRegex(ValueError, "неполное обучение"):
            validate_training(metrics, params, {"name": "all_layers", "freeze_first": 0}, "current", [{}] * 100, 5)


if __name__ == "__main__":
    unittest.main()
