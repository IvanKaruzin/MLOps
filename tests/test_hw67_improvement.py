"""Mock-only diagnostics, sealed access and additional quality guard contracts."""

import contextlib
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import mlflow.pyfunc
import yaml

from evaluation import improvement
from evaluation.contract import write_json


CHOICES = ["драма", "комедия", "REFUSE"]


def records():
    result = []
    for index, (source, language, answer) in enumerate([
            ("pinned_source", "en", "драма"), ("pinned_source", "en", "комедия"),
            ("author_synthetic", "en", "драма"), ("author_synthetic", "ru", "комедия"),
            ("author_synthetic", "en", "REFUSE"), ("author_synthetic", "ru", "REFUSE"),
            ("author_synthetic", "en", "REFUSE"), ("author_synthetic", "ru", "REFUSE")]):
        result.append(dict(id=f"item-{index}", input=f"Independent diagnostic input {index}", expected=answer,
                           group_id=f"family-{index}", kind="refusal" if answer == "REFUSE" else "genre",
                           source=source, language=language, novel_topic=index >= 6))
    return result


def outputs(rows):
    return [dict(id=row["id"], raw=json.dumps({"answer": row["expected"]}, ensure_ascii=False),
                 error=None, latency_seconds=0.01) for row in rows]


class ImprovementGuards(unittest.TestCase):
    def setUp(self):
        self.rows = improvement.diagnostic_rows(records(), CHOICES)

    def report(self, generated=None):
        return improvement.diagnostics(self.rows, outputs(self.rows) if generated is None else generated)

    def test_exact_scoring_keeps_invalid_errors_and_false_refusals_in_denominators(self):
        generated = outputs(self.rows)
        generated[0]["raw"] = '{"answer":"REFUSE"}'
        generated[1]["raw"] = "not JSON"
        generated[2]["error"] = "RuntimeError: inference failed"
        report = self.report(generated)
        self.assertEqual(report["overall"]["n"], len(self.rows))
        self.assertEqual(report["overall"]["errors"], 1)
        self.assertEqual(report["slices"]["genre"]["valid"], 2)
        self.assertEqual(report["false_refusals"]["genre"], dict(n=4, count=1, rate=0.25))
        self.assertFalse(improvement.additional_guard({"candidate": report, "reference": self.report()}, "dev")["passed"])

    def test_development_bars_keep_source_synthetic_refusal_and_ru_separate(self):
        good = self.report()
        self.assertTrue(improvement.additional_guard({"candidate": good, "reference": good}, "dev")["passed"])
        generated = outputs(self.rows)
        generated[3]["raw"] = '{"answer":"REFUSE"}'
        result = improvement.additional_guard({"candidate": self.report(generated), "reference": good}, "dev")
        failed = {check["name"] for check in result["checks"] if not check["passed"]}
        self.assertTrue({"synthetic_genre.accuracy", "ru_positive.false_refusal_rate"} <= failed)
        self.assertNotIn("source_genre.micro_f1", failed)

    def test_revised_synthetic_bar_accepts_three_of_four_without_changing_old_policy(self):
        expanded = records()
        for index, record in enumerate(records()[2:4]):
            expanded.append({**record, "id": f"extra-{index}", "group_id": f"extra-family-{index}"})
        rows = improvement.diagnostic_rows(expanded, CHOICES)
        generated = outputs(rows)
        generated[2]["raw"] = '{"answer":"комедия"}'
        report = improvement.diagnostics(rows, generated)
        self.assertEqual(report["slices"]["synthetic_genre"]["accuracy"], 0.75)
        reports = {"candidate": report, "reference": self.report()}
        self.assertTrue(improvement.additional_guard(reports, "dev")["passed"])
        with patch.dict(improvement.GUARDS, {"synthetic_genre_accuracy": 0.80}):
            self.assertFalse(improvement.additional_guard(reports, "dev")["passed"])
        generated[3]["raw"] = '{"answer":"драма"}'
        reports["candidate"] = improvement.diagnostics(rows, generated)
        self.assertFalse(improvement.additional_guard(reports, "dev")["passed"])

    def test_final_requires_real_reference_deltas_unseen_refusals_and_zero_generation_errors(self):
        good = self.report()
        bad_reference = outputs(self.rows)
        for item in bad_reference[:2]: item["raw"] = '{"answer":"REFUSE"}'
        reference = self.report(bad_reference)
        self.assertTrue(improvement.additional_guard({"candidate": good, "reference": reference}, "holdout")["passed"])
        self.assertFalse(improvement.additional_guard({"candidate": good, "reference": good}, "holdout")["passed"])
        for role in ("candidate", "reference", "base"):
            reports = dict(candidate=good, reference=reference, base=good)
            reports[role] = copy.deepcopy(reports[role])
            reports[role]["overall"]["errors"] = 1
            result = improvement.additional_guard(reports, "holdout")
            failed = {check["name"] for check in result["checks"] if not check["passed"]}
            self.assertIn(f"{role}.generation_errors", failed)
        generated = outputs(self.rows)
        generated[-1]["raw"] = '{"answer":"драма"}'
        result = improvement.additional_guard({"candidate": self.report(generated), "reference": reference}, "holdout")
        self.assertFalse(next(check for check in result["checks"] if check["name"] == "unseen_refusal.accuracy")["passed"])

    def test_exact_five_point_delta_handles_binary_rounding_but_materially_lower_delta_fails(self):
        candidate, reference = self.report(), self.report()
        for metric in ("micro_f1", "macro_f1", "accuracy"):
            candidate["slices"]["source_genre"][metric] = 0.60
            reference["slices"]["source_genre"][metric] = 0.55
        result = improvement.additional_guard(dict(candidate=candidate, reference=reference), "holdout")
        self.assertTrue(result["passed"])
        candidate["slices"]["source_genre"]["micro_f1"] = 0.599999
        result = improvement.additional_guard(dict(candidate=candidate, reference=reference), "holdout")
        delta = next(check for check in result["checks"] if check["name"] == "source_genre.delta_reference_micro_f1")
        self.assertFalse(delta["passed"])
        candidate = self.report()
        candidate["false_refusals"]["genre"]["rate"] = 0.02 + 1e-15
        self.assertTrue(improvement.additional_guard(dict(candidate=candidate, reference=reference), "dev")["passed"])

    def test_missing_required_groups_and_unknown_language_or_targets_fail_closed(self):
        subset = improvement.diagnostic_rows(records()[:2] + records()[4:], CHOICES)
        report = improvement.diagnostics(subset, outputs(subset))
        self.assertFalse(improvement.additional_guard({"candidate": report}, "dev")["passed"])
        for mutation in (dict(language="unknown"), dict(expected="Драма"), dict(kind="refusal")):
            changed = records()
            changed[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): improvement.diagnostic_rows(changed, CHOICES)

    def test_pyfunc_prediction_and_load_errors_preserve_every_id_and_latency(self):
        model = Mock()
        model.predict.side_effect = [["raw first response"], RuntimeError("fixture failure"), *([["raw"]] * 6)]
        with (patch.object(mlflow.pyfunc, "load_model", return_value=model) as load,
              patch.object(improvement, "release_memory"), patch("src.runtime.set_seed") as seed):
            generated = improvement.generate_training("runs:/training/model", {"seed": 42}, self.rows)
        load.assert_called_once_with("runs:/training/model")
        seed.assert_called_once_with(42)
        self.assertEqual([item["id"] for item in generated], [row["id"] for row in self.rows])
        self.assertEqual(generated[0]["raw"], "raw first response")
        self.assertIsNotNone(generated[1]["error"])
        self.assertIn("fixture failure", generated[1]["error"])
        masked = improvement.generation_error(RuntimeError("https://example.test/path?token=secret token=private"))
        self.assertNotIn("secret", masked)
        self.assertNotIn("private", masked)
        self.assertTrue(all(item["latency_seconds"] >= 0 for item in generated))
        with patch.object(mlflow.pyfunc, "load_model", side_effect=RuntimeError("load")), patch.object(improvement, "release_memory"):
            failed = improvement.generate_training("runs:/training/model", {"seed": 42}, self.rows)
        self.assertEqual(len(failed), len(self.rows))
        self.assertTrue(all(item["error"] for item in failed))
        with patch.object(improvement.cli, "generate", side_effect=RuntimeError("base load")), patch.object(improvement, "release_memory"):
            failed = improvement.generate_reference({}, {}, {}, self.rows)
        self.assertEqual(len(failed), len(self.rows))
        self.assertTrue(all(item["error"] for item in failed))


class ImprovementWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        previous = Path.cwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, previous)
        self.params = dict(data=dict(auxiliary="data/eval/auxiliary.json"), model=dict(dtype="bfloat16"))
        self.config = dict(golden="data/eval/golden.jsonl", system="The fixed serving instruction.",
                           seed=42, max_new_tokens=96, enable_thinking=False)
        self.baseline = dict(kind="base", name="Qwen/fixture", revision="a" * 40)
        self.manifest = dict(model={**self.baseline, "dtype": "bfloat16", "enable_thinking": False},
                             generation=dict(system=self.config["system"], max_new_tokens=96))
        write_json("evaluation/params.json", self.config)
        write_json("evaluation/baseline.json", self.baseline)
        for name in ("contract.py", "solution.py", "cli.py"):
            Path("evaluation", name).write_text("immutable fixture source\n")
        path = Path(self.config["golden"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(task="multilabel", expected="драма", choices=CHOICES,
                                        input="Official golden must never be generated")) + "\nINVALID SECOND ROW\n")
        self.seal = Path("data/eval/improvement-holdout.jsonl")
        self.seal.write_text("\n".join(json.dumps(row) for row in records()))
        self.bundle = dict(version=1, train=[dict(id="training-only")], dev=records(), manifest=dict(holdout=dict(
            path=str(self.seal), sha256=improvement.file_sha(self.seal), n=len(records()), sealed_before_training=True)))
        write_json(self.params["data"]["auxiliary"], self.bundle)

    def test_dev_does_not_read_holdout_and_sealed_access_checks_flag_path_hash_and_count(self):
        self.seal.unlink()
        rows, proof = improvement.load_split(self.params, self.config, "dev")
        self.assertEqual(len(rows), len(records()))
        self.assertNotIn("holdout_sha256", proof)
        with patch.object(improvement, "file_sha", side_effect=AssertionError("no reads permitted")):
            with self.assertRaises(ValueError): improvement.load_split(self.params, self.config, "holdout")
        self.seal.write_text("\n".join(json.dumps(row) for row in records()))
        rows, proof = improvement.load_split(self.params, self.config, "holdout", True)
        self.assertEqual(proof["holdout_sha256"], self.bundle["manifest"]["holdout"]["sha256"])
        self.seal.write_text(self.seal.read_text() + "\n{}")
        with self.assertRaises(ValueError): improvement.load_split(self.params, self.config, "holdout", True)

    def test_serving_contract_rejects_base_revision_dtype_thinking_system_and_truncation_mismatch(self):
        improvement.validate_manifest(self.manifest, self.baseline, self.params, self.config)
        for section, key, value in (("model", "revision", "b" * 40), ("model", "dtype", "float32"),
                                    ("model", "enable_thinking", True), ("generation", "system", "changed"),
                                    ("generation", "max_new_tokens", 16)):
            changed = copy.deepcopy(self.manifest)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                improvement.validate_manifest(changed, self.baseline, self.params, self.config)

    def test_training_package_requires_finished_child_parent_and_matching_actual_weights(self):
        root = Path("downloaded/model")
        adapter = root / "adapter"
        adapter.mkdir(parents=True)
        weights = b"fixture adapter bytes"
        (adapter / "adapter_model.safetensors").write_bytes(weights)
        digest = hashlib.sha256(weights).hexdigest()
        manifest = {**self.manifest, "checkpoint": dict(adapter_sha256=digest),
                    "sampling_sha256": "sampling", "training_view_sha256": "view"}
        write_json(adapter / "manifest.json", manifest)
        (root / "MLmodel").write_text(yaml.safe_dump(dict(flavors=dict(python_function=dict(artifacts=dict(adapter=dict(path="adapter")))))))
        child = SimpleNamespace(info=SimpleNamespace(status="FINISHED", run_id="child"),
                                data=SimpleNamespace(tags={"kind": "development", "mlflow.parentRunId": "parent",
                                                           "selected_adapter_sha256": digest,
                                                           "sampling_sha256": "sampling", "training_view_sha256": "view"}))
        parent = SimpleNamespace(info=SimpleNamespace(status="FINISHED", run_id="parent"))
        client = Mock(get_run=Mock(side_effect=lambda run_id: child if run_id == "child" else parent),
                      download_artifacts=Mock(return_value=str(root)))
        actual, spec = improvement.training_package(client, "child", "downloaded")
        self.assertEqual((actual, spec["weights_sha256"]), (manifest, digest))
        child.info.status = "RUNNING"
        with self.assertRaises(ValueError): improvement.training_package(client, "child", "downloaded")
        child.info.status = "FINISHED"
        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaises(ValueError): improvement.training_package(client, "child", "downloaded")

    def test_reference_uri_must_pin_version_and_registry_binding(self):
        client = Mock()
        client.get_model_version.return_value.run_id = improvement.PINNED_REFERENCE["run_id"]
        with patch.object(improvement.cli, "registry_version") as binding:
            spec = improvement.reference_spec(client, "models:/my-qwen-adapter/5")
            self.assertEqual(spec, improvement.PINNED_REFERENCE)
            binding.assert_called_once_with(client, spec)
        for uri in ("models:/my-qwen-adapter@candidate", "models:/my-qwen-adapter/latest", "runs:/fixture/model",
                    "models:/my-qwen-adapter/6", "models:/different-model/5"):
            with self.subTest(uri=uri), self.assertRaises(ValueError): improvement.reference_spec(client, uri)
        client.get_model_version.return_value.run_id = "another-training-run"
        with patch.object(improvement.cli, "registry_version") as binding:
            with self.assertRaises(ValueError): improvement.reference_spec(client, improvement.PINNED_REFERENCE["uri"])
            binding.assert_not_called()

    def test_mock_workflow_logs_raw_and_reports_but_dev_writes_no_repository_metrics_or_aliases(self):
        adapter = Path("reference-adapter")
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"reference weights")
        client, logged = Mock(), {}
        reference = dict(kind="adapter", name="fixture", version="5", run_id="reference", uri="models:/fixture/5")
        training = dict(kind="training_run", run_id="training", uri="runs:/training/model", weights_sha256="fixture",
                        training_provenance={"auxiliary_sha256": improvement.file_sha(self.params["data"]["auxiliary"])})
        run = SimpleNamespace(info=SimpleNamespace(run_id="diagnostic-run"))
        context = Mock()
        context.__enter__ = Mock(return_value=run)
        context.__exit__ = Mock(return_value=False)
        def generate(spec, manifest, config, rows):
            self.assertEqual({row["id"] for row in rows}, {row["id"] for row in records()})
            return outputs(rows)
        with (patch.object(improvement.cli, "tracking", return_value=(self.params, client)),
              patch.object(improvement, "reference_spec", return_value=reference) as reference_loader,
              patch.object(improvement.cli, "manifest_for", return_value=(self.manifest, adapter)),
              patch.object(improvement, "training_package", return_value=(self.manifest, training)),
              patch.object(improvement, "generate_training", side_effect=lambda uri, cfg, rows: generate(None, None, cfg, rows)),
              patch.object(improvement, "generate_reference", side_effect=generate),
              patch.object(mlflow, "set_experiment"), patch.object(mlflow, "start_run", return_value=context),
              patch.object(mlflow, "set_tags"), patch.object(mlflow, "register_model") as register,
              patch.object(mlflow, "log_dict", side_effect=lambda value, path: logged.update({path: copy.deepcopy(value)})),
              contextlib.redirect_stdout(io.StringIO()) as stdout):
            self.assertEqual(improvement.main(["--split", "dev", "--training-run-id", "training", "--include-base"]), 0)
            self.assertFalse(Path("metrics").exists())
            self.assertEqual(improvement.main(["--split", "holdout", "--allow-final", "--training-run-id", "training"]), 2)
            training["training_provenance"]["auxiliary_sha256"] = "stale"
            with self.assertRaises(ValueError): improvement.main(["--split", "dev", "--training-run-id", "training"])
        register.assert_not_called()
        self.assertTrue(all(call.args[1] == improvement.PINNED_REFERENCE["uri"] for call in reference_loader.call_args_list))
        client.set_registered_model_alias.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue().splitlines()[0])["evaluation_run_id"], "diagnostic-run")
        final = json.loads(Path("metrics/generalization-eval.json").read_text())
        self.assertFalse(final["passed"])
        self.assertEqual(set(final["reports"]), {"reference", "candidate"})
        self.assertEqual({path.name for path in Path("metrics").iterdir()}, {"generalization-eval.json"})
        for role in ("candidate", "reference", "base"):
            self.assertEqual(len(logged[f"diagnostics/{role}-outputs.json"]["outputs"]), len(records()))
            self.assertIn(f"diagnostics/{role}-report.json", logged)
        self.assertEqual(logged["diagnostics/rows.json"]["choices"], CHOICES)
        self.assertTrue(all("choices" not in row for row in logged["diagnostics/rows.json"]["rows"]))


if __name__ == "__main__":
    unittest.main()
