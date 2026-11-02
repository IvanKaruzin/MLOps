"""JSON training-view contracts and the unchanged suffix-loss objective."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from src import experiment_train
from src.data import pad_batch
from src.prompt import build_chat_text
from src.promote import validate_training_view
from src.tokenize_data import encode_example
from src.train import supervised_loss
from src.training_view import build_training_view, fingerprint

ROOT = Path(__file__).resolve().parents[1]


class TrainingViewContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT / "conf/hw67.yaml").read_text())
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.config["model"]["name"],
                revision=cls.config["model"]["revision"], local_files_only=True)
        except OSError as exc:
            raise unittest.SkipTest(f"Pinned tokenizer is not cached: {exc}")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.params = copy.deepcopy(self.config)
        self.params["model"].update(device="cpu", dtype="float32")
        source_config = yaml.safe_load((ROOT / "params.yaml").read_text())
        source_config["data"] = {}
        self.records, self.original = {}, {}
        for split in ("train", "val"):
            rows = [{"id": f"{split}-{index}", "topic": "release_year:2000", "messages": [
                {"role": "system", "content": "List film genres only."},
                {"role": "user", "content": f"A family reunites after {index + 2} years."},
                {"role": "assistant", "content": "комедия, драма"}]} for index in range(2)]
            self.records[split] = rows
            source = self.directory / f"{split}.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.params["data"][f"source_{split}"] = str(source)
            encoded = [encode_example(self.tokenizer, row, source_config, 320) for row in rows]
            for row in encoded:
                row.pop("_meta")
            self.original[split] = list(reversed(encoded))
        path = self.directory / "tokenization.yaml"
        path.write_text(yaml.safe_dump(source_config, allow_unicode=True))
        self.params["data"]["tokenization_params"] = str(path)
        self.snapshot = {path: path.read_bytes() for path in self.directory.iterdir()}
        self.original_snapshot = copy.deepcopy(self.original)

    def build(self, params=None):
        return build_training_view(params or self.params, self.original["train"], self.original["val"], self.tokenizer)

    def assert_originals_untouched(self):
        self.assertEqual(self.original, self.original_snapshot)
        self.assertEqual({path: path.read_bytes() for path in self.snapshot}, self.snapshot)

    def test_exact_serving_instruction_canonical_json_and_deterministic_view(self):
        train, val, proof = self.build()
        again = self.build()
        self.assertEqual((train, val, proof), again)
        self.assertEqual(fingerprint(proof), fingerprint(again[2]))
        evaluation = json.loads((ROOT / "evaluation/params.json").read_text())
        self.assertEqual(self.config["generation"]["system"], evaluation["system"])
        self.assertEqual(self.config["generation"]["max_new_tokens"], evaluation["max_new_tokens"])
        for split, rows in (("train", train), ("val", val)):
            self.assertEqual([row["id"] for row in rows], [row["id"] for row in self.original[split]])
            self.assertEqual(proof["splits"][split]["truncated"], 0)
            for sample in proof["splits"][split]["samples"]:
                self.assertEqual(sample["messages"][0]["content"], evaluation["system"])
                original = next(row for row in self.records[split] if row["id"] == sample["id"])
                self.assertEqual(sample["messages"][1], original["messages"][1])
                self.assertEqual(json.loads(sample["messages"][-1]["content"]), {"answer": "драма, комедия"})
                self.assertLessEqual(sample["meta"]["full_len"], 768)
        self.assert_originals_untouched()

    def test_only_complete_json_answer_and_eos_are_supervised_with_same_prefix(self):
        _, _, proof = self.build()
        template_params = {"generate": {"enable_thinking": False}}
        for split in ("train", "val"):
            for sample in proof["splits"][split]["samples"]:
                messages, encoded = sample["messages"], sample["encoded"]
                full = build_chat_text(self.tokenizer, messages, template_params, False)
                prefix = build_chat_text(self.tokenizer, messages[:-1], template_params, True)
                self.assertTrue(full.startswith(prefix))
                supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
                self.assertEqual(self.tokenizer.decode(supervised, skip_special_tokens=False),
                                 messages[-1]["content"] + self.tokenizer.eos_token)
                first = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
                self.assertTrue(all(label == -100 for label in encoded["labels"][:first]))
                self.assertEqual(self.tokenizer.decode(encoded["input_ids"][:first], skip_special_tokens=False), prefix)
        self.assert_originals_untouched()

    def test_source_mode_preserves_original_arrays(self):
        params = copy.deepcopy(self.params)
        params["train"]["response_format"] = "source"
        train, val, proof = self.build(params)
        self.assertEqual(train, self.original["train"])
        self.assertEqual(val, self.original["val"])
        self.assertEqual(proof["response_format"], "source")
        self.assertIsNone(proof["system"])
        self.assert_originals_untouched()

    def test_truncation_missing_ids_and_invalid_finite_labels_are_rejected(self):
        params = copy.deepcopy(self.params)
        params["train"]["view_max_seq_len"] = 16
        with self.assertRaises(ValueError):
            self.build(params)
        original = copy.deepcopy(self.original["train"])
        self.original["train"][0]["id"] = "missing"
        with self.assertRaises(ValueError):
            self.build()
        self.original["train"] = original
        params["train"]["response_format"] = "source"
        self.original["train"][0]["labels"][-1] = float("nan")
        with self.assertRaises(ValueError):
            self.build(params)

    def test_view_mismatch_refuses_before_loading_model(self):
        train, val, proof = self.build()
        with (patch.object(experiment_train, "prepare_training_data", return_value=(train, val, 0, {}, proof)),
              patch.object(experiment_train.AutoModelForCausalLM, "from_pretrained") as load):
            with self.assertRaises(ValueError):
                experiment_train.train(self.params, self.params["sweep"][0], self.directory / "adapter", Mock(),
                                       expected_view_sha256="changed")
            load.assert_not_called()
        self.assert_originals_untouched()

    def test_select_recomputes_view_proof_arrays_and_actual_token_counts(self):
        train, val, proof = self.build()
        sampling = {"processed_train": {"ids": [row["id"] for row in train]}}
        run = SimpleNamespace(info=SimpleNamespace(run_id="view-run"), data=SimpleNamespace(
            tags={"training_view_sha256": fingerprint(proof)}, metrics={
                "processed_tokens": sum(sum(row["attention_mask"]) for row in train),
                "processed_supervised_tokens": sum(sum(label != -100 for label in row["labels"][1:]) for row in train)}))
        mutation = None
        def download(run_id, artifact, destination):
            path = Path(destination) / artifact
            if artifact.endswith('.gz'):
                arrays = copy.deepcopy({"train": train, "val": val})
                if mutation == "arrays": arrays['train'][0]['labels'][-1] = -100
                path.write_bytes(gzip.compress(json.dumps(arrays).encode(), mtime=0))
            else:
                persisted = copy.deepcopy(proof)
                if mutation == "proof": persisted['system'] = "changed system"
                path.write_text(json.dumps(persisted))
            return str(path)
        client = Mock(download_artifacts=Mock(side_effect=download))
        with patch.object(experiment_train, "prepare_training_data", return_value=(train, val, 0, {}, proof)):
            validate_training_view(client, self.params, [run], sampling)
            for mutation in ("proof", "arrays", "hash", "tokens"):
                changed = copy.deepcopy(run)
                if mutation == "hash": changed.data.tags['training_view_sha256'] = "changed"
                if mutation == "tokens": changed.data.metrics['processed_tokens'] += 1
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_training_view(client, self.params, [changed], sampling)

    def test_suffix_loss_and_gradients_equal_full_forward_for_json_batch(self):
        train, _, _ = self.build()
        identities = sorted({token for row in train for token in row['input_ids']})
        compact = {token: index + 1 for index, token in enumerate(identities)}
        features = [{"input_ids": [compact[token] for token in row['input_ids']],
                     "attention_mask": row['attention_mask'],
                     "labels": [-100 if label == -100 else compact[label] for label in row['labels']]} for row in train]
        batch = pad_batch(features, 0)
        torch.manual_seed(7)
        cfg = Qwen3Config(vocab_size=len(compact) + 1, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                          attention_dropout=0.0)
        full = Qwen3ForCausalLM(cfg).eval()
        suffix = copy.deepcopy(full)
        context_lengths, projected_lengths = [], []
        pre = suffix.register_forward_pre_hook(lambda model, args, kwargs: context_lengths.append(kwargs['input_ids'].shape[1]), with_kwargs=True)
        post = suffix.register_forward_hook(lambda model, args, output: projected_lengths.append(output.logits.shape[1]))
        try:
            full_loss = full(**batch).loss
            suffix_loss = supervised_loss(suffix, batch)
        finally:
            pre.remove(); post.remove()
        self.assertTrue(torch.allclose(full_loss, suffix_loss, atol=1e-6, rtol=1e-6))
        full_loss.backward(); suffix_loss.backward()
        for (name, expected), (other, actual) in zip(full.named_parameters(), suffix.named_parameters()):
            self.assertEqual(name, other)
            if expected.grad is not None:
                self.assertTrue(torch.allclose(expected.grad, actual.grad, atol=1e-6, rtol=1e-5), name)
        self.assertEqual(context_lengths, [batch['input_ids'].shape[1]])
        self.assertTrue(0 < projected_lengths[0] < context_lengths[0])
        self.assert_originals_untouched()


if __name__ == '__main__':
    unittest.main()
