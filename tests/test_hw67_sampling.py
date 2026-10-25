"""Stratified sampling checks without model or GPU dependencies."""

import copy
import hashlib
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
from pathlib import Path

from src.sampling import STRATA, sample_split, source_metadata, verify_tensor_sources


class SamplingContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "source.jsonl")
        self.records = []
        for era, year in (("old", 1999), ("new", 2000)):
            for cardinality in (1, 2, 3):
                for index in range(50):
                    self.records.append({"id": f"{era}-{cardinality}-{index:02}",
                                         "topic": f"release_year:{year}",
                                         "messages": [{"role": "assistant", "content": ", ".join(
                                             f"genre{item}" for item in range(cardinality))}]})
        self.blob = {"examples": [{"id": row["id"], "input_ids": [index]}
                                  for index, row in enumerate(self.records)]}
        self.write_source()

    def write_source(self):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.records), encoding="utf-8")

    def sample(self, limit=32, seed=20261025, split="train", **kwargs):
        return sample_split(self.blob, self.path, limit, seed, split, **kwargs)

    def test_balanced_quotas_and_complete_ordered_manifest(self):
        for limit, split in ((240, "train"), (60, "val"), (32, "train"), (8, "val")):
            with self.subTest(limit=limit, split=split):
                selected, manifest = self.sample(limit, split=split)
                self.assertEqual(len(selected), limit)
                self.assertEqual(len(set(manifest["ids"])), limit)
                self.assertEqual([item["id"] for item in selected], manifest["ids"])
                self.assertEqual([item["id"] for item in manifest["selected_records"]], manifest["ids"])
                counts = Counter(item["stratum"] for item in manifest["selected_records"])
                self.assertEqual(dict(counts), manifest["strata_counts"])
                self.assertEqual(set(counts), set(STRATA))
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                self.assertEqual(manifest["source_rows"], 300)
                self.assertEqual(manifest["source_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_reproducible_nonprefix_selection_and_seed_variation(self):
        before = copy.deepcopy(self.blob)
        first = self.sample()
        self.assertEqual(first, self.sample())
        self.assertNotEqual(first[1]["ids"], [row["id"] for row in self.records[:32]])
        self.assertNotEqual(first[1]["ids"], self.sample(seed=20261026)[1]["ids"])
        self.assertEqual(self.blob, before)

    def test_source_and_tensor_permutations_preserve_identity_alignment(self):
        selected, manifest = self.sample()
        random.Random(3).shuffle(self.blob["examples"])
        random.Random(7).shuffle(self.records)
        self.write_source()
        actual, reordered_manifest = self.sample()
        self.assertEqual(actual, selected)
        self.assertEqual(reordered_manifest["ids"], manifest["ids"])
        self.assertNotEqual(reordered_manifest["source_sha256"], manifest["source_sha256"])

    def test_small_control_is_subset_of_large_pool(self):
        for split, small, large in (("train", 32, 240), ("val", 8, 60)):
            self.assertLess(set(self.sample(small, split=split)[1]["ids"]),
                            set(self.sample(large, split=split)[1]["ids"]))

    def test_boundary_year_and_cardinality(self):
        old = source_metadata(self.records[0])
        new = source_metadata(self.records[150])
        self.assertEqual(old["stratum"], "before2000__single_genre")
        self.assertEqual(new["stratum"], "since2000__single_genre")
        self.assertEqual(source_metadata(self.records[100])["stratum"], "before2000__three_plus_genres")

    def test_missing_duplicate_or_mismatched_ids_are_rejected(self):
        for change in ("source_duplicate", "source_missing", "tensor_duplicate", "tensor_missing", "tensor_unknown"):
            with self.subTest(change=change):
                records, blob = copy.deepcopy(self.records), copy.deepcopy(self.blob)
                if change == "source_duplicate": records.append(copy.deepcopy(records[0]))
                elif change == "source_missing": records.pop()
                elif change == "tensor_duplicate": blob["examples"].append(copy.deepcopy(blob["examples"][0]))
                elif change == "tensor_missing": blob["examples"][0].pop("id")
                else: blob["examples"][0]["id"] = "unknown"
                self.path.write_text("".join(json.dumps(row) + "\n" for row in records))
                with self.assertRaises(ValueError): sample_split(blob, self.path, 32, 42, "train")

    def test_invalid_metadata_is_rejected(self):
        for change in ("year", "year_type", "messages", "empty_label", "duplicate_label", "missing_id"):
            with self.subTest(change=change):
                row = copy.deepcopy(self.records[0])
                if change == "year": row["topic"] = "release_year:0000"
                elif change == "year_type": row["topic"] = 2000
                elif change == "messages": row["messages"] = []
                elif change == "empty_label": row["messages"][-1]["content"] = "drama, "
                elif change == "duplicate_label": row["messages"][-1]["content"] = "drama, DRAMA"
                else: row.pop("id")
                with self.assertRaises(ValueError): source_metadata(row)

    def test_undersized_or_absent_stratum_is_rejected_without_redistribution(self):
        self.records = self.records[1:]
        self.blob["examples"] = self.blob["examples"][1:]
        replacement = copy.deepcopy(self.records[-1])
        replacement["id"] = "new-3-extra"
        self.records.append(replacement)
        self.blob["examples"].append({"id": replacement["id"], "input_ids": [999]})
        self.write_source()
        with self.assertRaises(ValueError): self.sample(300)
        self.records = [row for row in self.records if not row["id"].startswith("old-1-")]
        self.blob["examples"] = [row for row in self.blob["examples"] if not row["id"].startswith("old-1-")]
        self.write_source()
        with self.assertRaises(ValueError): self.sample(32)

    def test_invalid_budgets_seed_method_source_or_split_are_rejected(self):
        for limit in (True, 0, 5, 301, 32.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError): self.sample(limit)
        for seed in (True, -1, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError): self.sample(seed=seed)
        with self.assertRaises(ValueError): self.sample(method="prefix")
        with self.assertRaises(ValueError): self.sample(split="test")
        with self.assertRaises(ValueError): sample_split(self.blob, self.path.parent, 32, 42, "train")
        self.path.write_text("not json\n")
        with self.assertRaises(ValueError): self.sample()

    def test_full_reencoding_rejects_changed_plots_or_labels_with_identical_ids(self):
        import torch
        import yaml
        from transformers import AutoTokenizer
        from src.tokenize_data import encode_example

        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "conf/hw67.yaml").read_text())
        try:
            tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"],
                                                      revision=config["model"]["revision"], local_files_only=True)
        except OSError as exc:
            self.skipTest(f"Pinned cached tokenizer unavailable: {exc}")
        tokenization = {"model": config["model"], "generate": {"enable_thinking": False},
                        "tokenize": {"max_seq_len": 320, "padding_side": "left"}, "data": {}}
        records = {}
        for split in ("train", "val"):
            source = Path(self.directory.name, f"{split}.jsonl")
            tensor = Path(self.directory.name, f"{split}.pt")
            row = {"id": split + "-example", "topic": "release_year:2000", "messages": [
                {"role": "system", "content": "Name the film genres."},
                {"role": "user", "content": "A family reunites after many years."},
                {"role": "assistant", "content": "драма"}]}
            records[split] = row
            source.write_text(json.dumps(row) + "\n")
            encoded = encode_example(tokenizer, row, tokenization, 320)
            encoded.pop("_meta")
            torch.save({"examples": [encoded], "model": config["model"]["name"], "max_seq_len": 320,
                        "padding_side": "left", "pad_token_id": tokenizer.pad_token_id}, tensor)
            config["data"].update({split: str(tensor), f"source_{split}": str(source)})
            tokenization["data"].update({split: str(tensor), f"{split}_jsonl": str(source)})
        original_params = Path(self.directory.name, "tokenization.yaml")
        original_params.write_text(yaml.safe_dump(tokenization))
        config["data"]["tokenization_params"] = str(original_params)
        self.assertIs(verify_tensor_sources(config)["passed"], True)
        for role, replacement in (("user", "An astronaut discovers a hostile planet."), ("assistant", "комедия")):
            changed = copy.deepcopy(records["train"])
            next(message for message in changed["messages"] if message["role"] == role)["content"] = replacement
            Path(config["data"]["source_train"]).write_text(json.dumps(changed) + "\n")
            with self.subTest(role=role), self.assertRaises(ValueError):
                verify_tensor_sources(config)
        Path(config["data"]["source_train"]).write_text(json.dumps(records["train"]) + "\n")
        original_encode = encode_example
        changed_during_check = False
        def race(tokenizer, row, params, max_seq_len):
            nonlocal changed_during_check
            result = original_encode(tokenizer, row, params, max_seq_len)
            if not changed_during_check:
                path = Path(config["data"]["source_train"])
                path.write_bytes(path.read_bytes() + b"\n")
                changed_during_check = True
            return result
        with patch("src.tokenize_data.encode_example", side_effect=race), self.assertRaises(ValueError):
            verify_tensor_sources(config)


if __name__ == "__main__":
    unittest.main()
