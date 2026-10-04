"""Fast training regressions without downloading or fitting a large model."""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.data import batches, load_split
from src.runtime import set_seed
from src.train import (
    TRAIN_CODE, evaluate, inputs_fingerprint, lora_config, main, run_training,
    validate_training_config,
)


def example(values):
    ids = [0] + list(values)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] + list(values)}


class TokenLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask, labels):
        targets = labels[:, 1:]
        targets = targets[targets != -100].float()
        return SimpleNamespace(loss=((self.weight - targets) ** 2).mean())


class HW5TrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"epochs": 1, "batch_size": 1, "grad_accum": 2,
                    "eval_batch_size": 2, "eval_every": 2, "seed": 42,
                    "lr": 0.001, "max_grad_norm": 100., "warmup_ratio": 0., "weight_decay": 0.}
        self.examples = [example([1]), example([2, 2, 2]), example([3, 3]),
                         example([1, 2]), example([3])]
        self.device = torch.device("cpu")

    def fit(self, model, cfg, examples, total_steps):
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        with contextlib.redirect_stdout(io.StringIO()):
            return run_training(model, examples, self.examples, 0, self.device,
                                cfg, optimizer, scheduler, total_steps)

    def test_validation_weighted_by_shifted_tokens_and_restores_mode(self):
        model = TokenLossModel().eval()
        records = [example([1]), example([3, 3, 3])]
        self.assertAlmostEqual(evaluate(model, records, 0, self.device, 1), (0.25 + 3 * 6.25) / 4)
        self.assertFalse(model.training)
        self.assertAlmostEqual(evaluate(model, records, 0, self.device, 2), 4.75)

    def test_accumulation_matches_complete_token_weighted_batch(self):
        first, second = TokenLossModel(), TokenLossModel()
        records = self.examples[:3]
        micro_cfg, full_cfg = copy.deepcopy(self.cfg), copy.deepcopy(self.cfg)
        micro_cfg["grad_accum"] = 3
        full_cfg.update(batch_size=3, grad_accum=1)
        micro_result = self.fit(first, micro_cfg, records, 1)
        full_result = self.fit(second, full_cfg, records, 1)
        self.assertTrue(torch.allclose(first.weight, second.weight, atol=1e-7, rtol=0))
        self.assertEqual(micro_result["curve_train"], full_result["curve_train"])
        self.assertEqual(micro_result["processed_supervised_tokens"], 6)

    def test_residual_accumulation_is_flushed_per_epoch_and_final_eval(self):
        cfg = {**self.cfg, "epochs": 2}
        result = self.fit(TokenLossModel(), cfg, self.examples, 6)
        self.assertEqual(result["steps"], 6)
        self.assertEqual(result["processed_examples"], 10)
        self.assertEqual(result["processed_tokens"], 28)
        self.assertEqual([row[0] for row in result["curve_val"]], [0, 2, 4, 6])
        self.assertGreater(result["peak_memory_mb"], 0)

    def test_max_steps_counts_only_consumed_batches_and_final_eval(self):
        result = self.fit(TokenLossModel(), self.cfg, self.examples, 1)
        self.assertEqual(result["processed_examples"], 2)
        self.assertEqual([row[0] for row in result["curve_val"]], [0, 1])
        ordered = list(batches(self.examples, 1, 0, True, 42))[:2]
        self.assertEqual(result["processed_tokens"], sum(int(b["attention_mask"].sum()) for b in ordered))

    def test_no_supervision_or_nan_validation_fails(self):
        model = TokenLossModel()
        records = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, -100]}]
        with self.assertRaises(ValueError):
            evaluate(model, records, 0, self.device, 1)
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        with self.assertRaises(RuntimeError):
            evaluate(model, self.examples, 0, self.device, 2)

    def test_freeze_layers_actual_trainable_parameters(self):
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4)
        params = {"lora": {"r": 2, "alpha": 4, "dropout": 0.05,
                           "target_modules": ["q_proj", "v_proj"]}}
        set_seed(42)
        full = get_peft_model(Qwen3ForCausalLM(cfg), lora_config(params, 4, 0))
        set_seed(42)
        frozen = get_peft_model(Qwen3ForCausalLM(cfg), lora_config(params, 4, 2))
        counts = [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in (full, frozen)]
        self.assertEqual(counts[0], 2 * counts[1])
        names = [name for name, p in frozen.named_parameters() if p.requires_grad]
        self.assertTrue(all("layers.2." in name or "layers.3." in name for name in names))
        self.assertEqual(frozen.peft_config["default"].layers_to_transform, [2, 3])
        with self.assertRaises(ValueError):
            lora_config(params, 4, 4)

    def test_seed_precedes_adapter_initialization(self):
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4)
        params = {"lora": {"r": 2, "alpha": 4, "dropout": 0.1, "target_modules": ["q_proj"]}}
        states = []
        for _ in range(2):
            set_seed(123)
            model = get_peft_model(Qwen3ForCausalLM(cfg), lora_config(params, 2, 0))
            states.append({name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad})
        self.assertTrue(all(torch.equal(states[0][name], states[1][name]) for name in states[0]))

    def test_split_validation_rejects_bad_labels_and_empty_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.pt"
            blob = {"examples": copy.deepcopy(self.examples), "pad_token_id": 0}
            torch.save(blob, path)
            self.assertEqual(load_split(str(path))["examples"], self.examples)
            blob["examples"][0]["labels"][1] = 999
            torch.save(blob, path)
            with self.assertRaises(ValueError):
                load_split(str(path))
            blob["examples"][0]["labels"][1] = -100
            torch.save(blob, path)
            with self.assertRaises(ValueError):
                load_split(str(path))

    def test_fingerprint_includes_tensor_content_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in TRAIN_CODE:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("code\n")
            for name in ("train.pt", "val.pt"):
                (root / name).write_bytes(b"data")
            params = {"data": {"train": "train.pt", "val": "val.pt"}, "train": {"lr": 0.1}}
            with patch("src.train.PROJECT_ROOT", root):
                before = inputs_fingerprint(params)
                (root / "train.pt").write_bytes(b"other data")
                self.assertNotEqual(before, inputs_fingerprint(params))
                after = inputs_fingerprint(params)
                params["train"]["lr"] = 0.2
                self.assertNotEqual(after, inputs_fingerprint(params))

    def test_invalid_max_steps_and_training_configuration(self):
        validate_training_config(self.cfg, None)
        for steps in (0, -1):
            with self.assertRaises(ValueError):
                validate_training_config(self.cfg, steps)
        with self.assertRaises(ValueError):
            validate_training_config({**self.cfg, "grad_accum": 0}, None)

    def test_cli_smoke_cannot_replace_production_artifacts(self):
        for option in ("--max-steps", "--val-limit"):
            with patch("sys.argv", ["train", option, "1"]), patch("src.train.load_params") as loader:
                with self.assertRaisesRegex(ValueError, "require --out"):
                    main()
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
