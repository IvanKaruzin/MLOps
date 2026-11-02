"""CPU-only safeguards for independent HW6/7 improvement data and its seal."""

import copy
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.prepare_hw67_auxiliary import (
    FROZEN_DEV_SHA256, FROZEN_HOLDOUT_SHA256, FROZEN_LEGACY_TRAIN_SHA256,
    FROZEN_TRAIN_SOURCE_SHA256, GENRE_SPECS, HOLDOUT_PATH, REFUSAL_TASKS, ROOT, SPLITS,
    filter_source_extras, legacy_synthetic_rows, train_v2_additions,
    filter_title_extras, fingerprint, normalize_title, prepare, record, source_titles,
    synthetic_rows, validate_isolation, validate_rows,
)
from src.textnorm import normalize_text


class AuxiliaryGuards(unittest.TestCase):
    def row(self, identifier, text, target="драма", **kwargs):
        return record(identifier, text, target, "group-" + identifier, "test", **kwargs)

    def test_every_extra_in_exact_cluster_is_removed_and_originals_are_retained(self):
        originals = [{"id": "original", "input": "An old immutable description.", "expected": "драма"}]
        extras = [self.row("extra-copy", "  AN OLD immutable description. "),
                  self.row("conflict-a", "A completely separate duplicated plot.", "драма"),
                  self.row("conflict-b", "A completely separate duplicated plot.", "комедия"),
                  self.row("unique", "A newly written and genuinely distinct film description.")]
        before = copy.deepcopy((originals, extras))
        kept, audit = filter_source_extras(originals, extras, [])
        self.assertEqual([row["id"] for row in kept], ["unique"])
        self.assertEqual(audit["removed_ids"], ["conflict-a", "conflict-b", "extra-copy"])
        self.assertEqual((originals, extras), before)
        conflict = next(cluster for cluster in audit["clusters"] if len(cluster["records"]) == 2
                        and {row["id"] for row in cluster["records"]} == {"conflict-a", "conflict-b"})
        self.assertEqual({row["expected"] for row in conflict["records"]}, {"драма", "комедия"})

    def test_near_cluster_excludes_both_extra_members_regardless_of_target(self):
        first = " ".join(f"word{index}" for index in range(100))
        second = first.replace("word50", "differentword")
        extras = [self.row("near-a", first), self.row("near-b", second, "комедия")]
        kept, audit = filter_source_extras([], extras, [])
        self.assertEqual(kept, [])
        self.assertEqual(audit["removed_ids"], ["near-a", "near-b"])
        self.assertEqual(audit["near_pairs_with_extra"], 1)
        self.assertGreaterEqual(audit["clusters"][0]["jaccard"], 0.85)

    def test_title_filter_removes_all_raw_and_cross_role_namesakes_but_preserves_same_role(self):
        def extra(identifier, role):
            return {**self.row(identifier, "A unique plot for " + identifier), "source_id": identifier, "role": role}
        titles = {"raw": "Ｆｏｏ： Bar!", "raw-copy": "foo bar", "cross-a": "A—Shared Title",
                  "cross-b": "a shared title", "same-a": "Other Name", "same-b": "OTHER NAME"}
        extras = [extra("raw-copy", "train"), extra("cross-a", "train"), extra("cross-b", "dev"),
                  extra("same-a", "train"), extra("same-b", "train")]
        before = copy.deepcopy(extras)
        kept, audit = filter_title_extras({"raw"}, extras, titles)
        self.assertEqual({row["id"] for row in kept}, {"same-a", "same-b"})
        self.assertEqual(audit["removed_ids"], ["cross-a", "cross-b", "raw-copy"])
        self.assertEqual(normalize_title(titles["raw"]), "foo bar")
        self.assertEqual(extras, before)

    def test_schema_rejects_lowercase_refusal_unknown_target_missing_identity_and_choices(self):
        good = self.row("good", "Explain the chemistry of soap.", "REFUSE")
        validate_rows([good], {"драма", "REFUSE"})
        mutations = [dict(expected="refuse"), dict(expected="unknown"), dict(kind="genre"),
                     dict(source="unreviewed"), dict(id=""), dict(input=None), dict(choices=["REFUSE"])]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rows([{**good, **mutation}], {"драма", "REFUSE"})
        source = {**self.row("real", "An independent source film."), "source": "pinned_source"}
        with self.assertRaises(ValueError):
            validate_rows([source], {"драма", "REFUSE"})
        with self.assertRaises(ValueError):
            validate_rows([good, good], {"драма", "REFUSE"})

    def test_exact_group_and_near_overlap_fail_closed(self):
        left = self.row("train", "An independent original plot description.")
        right = self.row("dev", "A separate development description.")
        for field, value in (("id", left["id"]), ("group_id", left["group_id"]), ("input", left["input"])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_isolation({"train": [left], "dev": [{**right, field: value}]}, [])
        with self.assertRaises(ValueError):
            validate_isolation({"train": [left]}, [left])
        long_plot = " ".join(f"protected{index}" for index in range(100))
        with self.assertRaises(ValueError):
            validate_isolation({"train": [self.row("extra", long_plot.replace("protected50", "newtoken"))]},
                               [self.row("protected", long_plot)])

    def test_synthetic_counts_all_genres_and_distinct_rendering_families(self):
        expected = {"train": {"refusal": 720, "genre": 1728},
                    "dev": {"refusal": 96, "genre": 48},
                    "sealed_final": {"refusal": 144, "genre": 72}}
        families = {}
        for split in SPLITS:
            rows = synthetic_rows(split)
            self.assertEqual(Counter(row["kind"] for row in rows), expected[split])
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            self.assertEqual(len({normalize_text(row["input"]) for row in rows}), len(rows))
            self.assertTrue(set(GENRE_SPECS) <= {row["expected"] for row in rows})
            for kind, count in expected[split].items():
                self.assertEqual(Counter(row["language"] for row in rows if row["kind"] == kind),
                                 {"en": count // 2, "ru": count // 2})
            for row in rows:
                cyrillic = sum("а" <= char.casefold() <= "я" or char.casefold() == "ё" for char in row["input"])
                if row["language"] == "ru":
                    self.assertGreater(cyrillic, 20)
                else:
                    self.assertEqual(cyrillic, 0)
            families[split] = {row["template_family"] for row in rows}
            # None of the release examples' named entities or constructions is
            # used to create training/development/final synthetic examples.
            corpus = "\n".join(row["input"].casefold() for row in rows)
            for phrase in ("paris", "france", "jed cooper", "lynch", "abstract shapes", "geometric shapes",
                           "17 and 23", "17 и 23", "столицу", "девяти мужчин"):
                self.assertNotIn(phrase, corpus)
        for left, right in (("train", "dev"), ("train", "sealed_final"), ("dev", "sealed_final")):
            self.assertFalse(families[left] & families[right])
        for spec in GENRE_SPECS.values():
            self.assertEqual(len(set(spec[:3])), 3)
        known_topics = {topic for split in ("train", "dev") for topic, _, _ in REFUSAL_TASKS[split]}
        unseen = {topic for topic, _, _ in REFUSAL_TASKS["sealed_final"]} - known_topics
        self.assertEqual(len(unseen), 4)
        final_refusal = [row for row in synthetic_rows("sealed_final") if row["kind"] == "refusal"]
        self.assertEqual(Counter(row["novel_topic"] for row in final_refusal), {True: 48, False: 96})
        self.assertIn("film_facts", known_topics)
        self.assertIn("recommendation", known_topics)
        self.assertIn("genre_definitions", known_topics)

    def test_train_revision_counts_semantic_groups_translation_views_and_task_contrasts(self):
        additions = train_v2_additions()
        genre = [row for row in additions if row["kind"] == "genre"]
        refusal = [row for row in additions if row["kind"] == "refusal"]
        self.assertEqual((len(genre), len(refusal)), (1488, 240))
        semantic = {}
        for row in genre:
            semantic.setdefault(row["premise_id"], []).append(row)
            self.assertEqual(row["premise_id"], row["parent_premise"])
            self.assertEqual(row["group_id"], row["premise_id"])
            self.assertNotIn('{"answer"', row["input"])
            if ", " in row["expected"]:
                self.assertNotIn(row["expected"], row["input"])
        self.assertEqual(len(semantic), 248)
        self.assertEqual(len({row["language_premise_id"] for row in genre}), 496)
        for rows in semantic.values():
            self.assertEqual(len(rows), 6)
            self.assertEqual(Counter(row["language"] for row in rows), {"en": 3, "ru": 3})
            self.assertEqual(len({row["expected"] for row in rows}), 1)
        all_genre = [row for row in synthetic_rows("train") if row["kind"] == "genre"]
        singletons = Counter(row["expected"] for row in all_genre if ", " not in row["expected"])
        self.assertEqual(singletons, {label: 48 for label in GENRE_SPECS})
        self.assertEqual(sum(", " in row["expected"] for row in all_genre), 864)
        self.assertEqual(len({row["expected"] for row in genre if ", " in row["expected"]}), 35)
        allowed_topics = {topic for topic, _, _ in REFUSAL_TASKS["train"]}
        for row in refusal:
            self.assertIn(row["topic"], allowed_topics)
            self.assertIn(row["contrast_premise_id"], semantic)
            self.assertEqual(row["group_id"], row["contrast_premise_id"])
            self.assertFalse(row["novel_topic"])
            self.assertEqual(row["expected"], "REFUSE")
        # Only TRAIN changes; the legacy dev/final generator remains exact.
        for split in ("dev", "sealed_final"):
            self.assertEqual(synthetic_rows(split), legacy_synthetic_rows(split))


class FrozenAuxiliaryData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = [ROOT / path for path in (
            "data/raw.jsonl", "data/train.jsonl", "data/val.jsonl", "data/test.jsonl",
            "data/tokenized/train.pt", "data/tokenized/val.pt", "data/eval/golden.jsonl",
            "data/eval/behavioral.jsonl", HOLDOUT_PATH)]
        cls.before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.inputs}
        cls.bundle, cls.holdout = prepare()

    def test_published_bundle_and_seal_match_reproduction_without_mutating_originals(self):
        published_path = ROOT / "data/eval/auxiliary.json"
        self.assertEqual(hashlib.sha256(published_path.read_bytes()).hexdigest(),
                         "0af9efde4dfb2665459f4f0161c5542d72872e71ac8a673195fd903e014c8785")
        self.assertEqual(hashlib.sha256((ROOT / HOLDOUT_PATH).read_bytes()).hexdigest(),
                         "4d9f9231d7b46927d60df8b8a2f55f89b27ad7223bc54442f4028a33fbc477fc")
        published = json.loads(published_path.read_text())
        # Creation provenance remains tied to the exact builder archived in the
        # parent MLflow run under data-preparation/. Later CLI cleanup changes
        # that file's SHA, but must not rewrite any frozen release metadata.
        self.assertEqual(published["manifest"]["builder_sha256"],
                         "514e4b01ce2680824cfbcdf9648568a0cbdc00ceee4c576ad9acb99e8d3ee6e7")
        self.assertEqual(self.bundle["manifest"]["builder_sha256"], hashlib.sha256(
            (ROOT / "scripts/prepare_hw67_auxiliary.py").read_bytes()).hexdigest())
        prepared = copy.deepcopy(self.bundle)
        published["manifest"].pop("builder_sha256")
        prepared["manifest"].pop("builder_sha256")
        self.assertEqual(published, prepared)
        self.assertEqual((ROOT / HOLDOUT_PATH).read_bytes(), self.holdout)
        self.assertEqual(self.before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.inputs})
        self.assertEqual(self.bundle["schema_version"], 1)
        self.assertNotIn("sealed_final", self.bundle)
        seal = self.bundle["manifest"]["holdout"]
        self.assertEqual(seal["sha256"], hashlib.sha256(self.holdout).hexdigest())
        self.assertEqual(seal["n"], len(self.holdout.splitlines()))
        self.assertTrue(seal["sealed_before_training"])

    def test_actual_complete_source_counts_catalogue_and_exclusion_evidence(self):
        manifest = self.bundle["manifest"]
        actual = {split: (counts["source_extra"], counts["synthetic"], counts["with_original_total"])
                  for split, counts in manifest["counts"].items()}
        self.assertEqual(actual, {"train": (2758, 2448, 10006), "dev": (344, 144, 1087),
                                  "sealed_final": (325, 216, 541)})
        self.assertEqual(len(manifest["source_audit"]["removed_ids"]), 5)
        self.assertEqual(len(manifest["source_audit"]["clusters"]), 4)
        self.assertEqual(manifest["source_audit"]["near_pairs_with_extra"], 0)
        revision = manifest["revision"]
        self.assertFalse(revision["source_selection_uses_model_outputs"])
        self.assertFalse(revision["builder_reads_model_outputs"])
        self.assertTrue(revision["train_augmentation_design_is_dev_informed"])
        self.assertNotIn("selection_uses_model_outputs", manifest)
        choices = json.loads((ROOT / "data/eval/golden.jsonl").read_text().splitlines()[0])["choices"]
        self.assertEqual(manifest["choice_catalog_sha256"], fingerprint(choices))
        raw_ids = {json.loads(line)["id"] for line in (ROOT / "data/raw.jsonl").read_text().splitlines()}
        for split in ("train", "dev"):
            rows = self.bundle[split]
            validate_rows(rows, choices)
            self.assertFalse(raw_ids & {row.get("source_id") for row in rows})
            original = "train" if split == "train" else "val"
            years = {int(json.loads(line)["topic"].split(":")[1])
                     for line in (ROOT / f"data/{original}.jsonl").read_text().splitlines()}
            self.assertTrue({row["release_year"] for row in rows if row["source"] == "pinned_source"} <= years)
        self.assertEqual(manifest["release_items_untouched"], {"golden": 90, "behavioral": 5})

    def test_dev_source_legacy_rows_and_seal_match_independent_v1_witness(self):
        self.assertEqual(fingerprint(self.bundle["dev"]), FROZEN_DEV_SHA256)
        self.assertEqual(fingerprint(self.bundle["train"][:3478]), FROZEN_LEGACY_TRAIN_SHA256)
        self.assertEqual(fingerprint([row for row in self.bundle["train"] if row["source"] == "pinned_source"]),
                         FROZEN_TRAIN_SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(self.holdout).hexdigest(), FROZEN_HOLDOUT_SHA256)
        witness_path = Path("/tmp/mlops-improvement-v1-witness.json")
        if witness_path.exists():
            witness = json.loads(witness_path.read_text())
            self.assertEqual(fingerprint(self.bundle["dev"]), witness["dev_sha256"])
            self.assertEqual(fingerprint([row for row in self.bundle["train"] if row["source"] == "pinned_source"]),
                             witness["train_source_sha256"])
            self.assertTrue(set(witness["old_train_ids"]) <= {row["id"] for row in self.bundle["train"]})
            for path, digest in witness["files"].items():
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_real_extra_titles_and_complete_source_membership_respect_frozen_roles(self):
        import yaml
        from src.collect import collect_candidates

        params = yaml.safe_load((ROOT / "params.yaml").read_text())
        source_path = ROOT / "sources/movies-dataset/9000plus.csv"
        titles = source_titles(source_path, params)
        raw_ids = {json.loads(line)["id"] for line in (ROOT / "data/raw.jsonl").read_text().splitlines()}
        raw_titles = {normalize_title(titles[identifier]) for identifier in raw_ids}
        seal = [json.loads(line) for line in self.holdout.splitlines()]
        splits = {"train": self.bundle["train"], "dev": self.bundle["dev"], "sealed_final": seal}
        owned_titles = {}
        for split, rows in splits.items():
            for row in rows:
                if row["source"] != "pinned_source":
                    continue
                title = normalize_title(titles[row["source_id"]])
                self.assertNotIn(title, raw_titles)
                self.assertIn(owned_titles.setdefault(title, split), (split,))
                self.assertEqual(row["language"], "en")
        removed = set(self.bundle["manifest"]["source_audit"]["removed_ids"])
        removed.update(self.bundle["manifest"]["title_audit"]["removed_ids"])
        candidates, _ = collect_candidates(params, [source_path])
        for split, original in (("train", "train"), ("dev", "val"), ("sealed_final", "test")):
            years = {int(json.loads(line)["topic"].split(":")[1])
                     for line in (ROOT / f"data/{original}.jsonl").read_text().splitlines()}
            eligible = {"aux-source-" + candidate.source_id for candidate in candidates
                        if candidate.source_id not in raw_ids and candidate.release_year in years}
            actual = {row["id"] for row in splits[split] if row["source"] == "pinned_source"}
            self.assertEqual(actual, eligible - removed)


if __name__ == "__main__":
    unittest.main()
