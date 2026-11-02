"""Real-evidence binding tests with tiny files and mocked model/network loaders."""

import copy
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hw8910 import runtime
from release import validate_release


class RuntimeEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runtime, "REPOSITORY", self.root))
        # Keep tests on the in-process pinned-runtime branch despite their
        # temporary repository. Other modules' actual sys.prefix is untouched.
        self.stack.enter_context(patch.object(runtime, "sys", SimpleNamespace(prefix=str(self.root / ".venv"))))
        self.baseline_spec = {"kind": "base", "name": "Qwen/test-fixture", "revision": "a" * 40}
        self.candidate_spec = {"kind": "adapter", "name": "test-adapter", "version": "5",
                               "run_id": "training-fixture", "uri": "models:/test-adapter/5"}
        self.config = dict(system="The fixed evaluated instruction.", max_new_tokens=96,
                           seed=20261025, enable_thinking=False, critical_slices=["fixture-slice"])
        self.reports = {
            role: dict(model=copy.deepcopy(spec), evaluation_run_id="evaluation-fixture",
                       protocol_sha256="d" * 64, golden_sha256="e" * 64,
                       overall=dict(accuracy=accuracy, format_rate=1.0))
            for role, spec, accuracy in (("baseline", self.baseline_spec, 0.1),
                                         ("candidate", self.candidate_spec, 0.2))
        }
        self.gate = dict(passed=False, reason="The required behavioral cases failed")
        for name in runtime.EVIDENCE_FILES:
            value = {"fixture_path": name}
            if name == "evaluation/params.json":
                value = self.config
            elif name == "evaluation/baseline.json":
                value = self.baseline_spec
            elif name == "metrics/baseline-eval.json":
                value = self.reports["baseline"]
            elif name == "metrics/candidate-eval.json":
                value = self.reports["candidate"]
            runtime.save(self.root / name, value)
        self.repository = self.root / "hf-repository"
        self.snapshot = self.repository / "snapshots" / self.baseline_spec["revision"]
        self.snapshot.mkdir(parents=True)
        for name in ("config.json", "generation_config.json", "model.safetensors", *runtime.TOKENIZER_FILES):
            (self.snapshot / name).write_bytes(f"tiny fixture for {name}\n".encode())
        runtime.save(self.snapshot / "tokenizer_config.json", {"chat_template": "frozen template"})
        self.base_hashes = {
            str((self.snapshot / name).relative_to(self.repository)): runtime.file_sha(self.snapshot / name)
            for name in ("config.json", "generation_config.json", "model.safetensors", *runtime.TOKENIZER_FILES)
        }
        self.stack.enter_context(patch.object(runtime, "hf_assets", return_value=(
            self.repository, self.snapshot, self.base_hashes)))
        self.actual_adapter = self.root / "mlflow-original" / "adapter"
        self.actual_adapter.mkdir(parents=True)
        (self.actual_adapter / "adapter_model.safetensors").write_bytes(b"not neural weights; immutable fixture")
        runtime.save(self.actual_adapter / "adapter_config.json", {"r": 8, "target_modules": ["q_proj"]})
        self.adapter = self.workspace / "adapter"
        shutil.copytree(self.actual_adapter, self.adapter)
        self.training_manifest = dict(model={**self.baseline_spec, "dtype": "bfloat16", "device": "cpu",
                                             "enable_thinking": False},
                                      generation={"system": self.config["system"], "max_new_tokens": 96})
        self.manifest_loader = self.stack.enter_context(patch("evaluation.cli.manifest_for", return_value=(
            self.training_manifest, self.actual_adapter)))
        self.frozen_evaluation = runtime.frozen_evaluation
        self.frozen = self.stack.enter_context(patch.object(runtime, "frozen_evaluation", side_effect=lambda: (
            copy.deepcopy(self.reports), copy.deepcopy(self.gate), self.current_evidence())))
        evaluated_runtime, _, _ = runtime.evaluation_runtime(self.baseline_spec, self.training_manifest)
        adapter_hashes = {name: runtime.file_sha(self.adapter / name)
                          for name in ("adapter_model.safetensors", "adapter_config.json")}
        self.candidate_manifest = runtime.manifest(
            self.candidate_spec["run_id"], self.adapter, adapter_hashes, evaluated_runtime, self.reports["candidate"],
            model_spec=self.candidate_spec, backend_kind="adapter", synthetic=False,
            training_run_id=self.candidate_spec["run_id"], base_artifact_uri=str(self.repository),
            base_file_hashes=self.base_hashes)
        self.release_report = {key: self.candidate_manifest[key] for key in
                               ("model_id", "model_sha256", "protocol_sha256", "golden_sha256", "runtime_sha256")}
        self.release_report.update(status="complete", passed=False, reason=self.gate["reason"],
            evaluation_run_id="evaluation-fixture", synthetic=False, quality_gate=copy.deepcopy(self.gate),
            evidence_files=self.current_evidence(), training_run_id=self.candidate_spec["run_id"],
            metrics={"accuracy": 0.2, "format_rate": 1.0})
        self.baseline_manifest = runtime.manifest(
            f"{self.baseline_spec['name']}@{self.baseline_spec['revision']}", self.repository,
            self.base_hashes, evaluated_runtime, self.reports["baseline"],
            model_spec=self.baseline_spec, backend_kind="base", synthetic=False,
            acceptance_basis="initial_pinned_baseline_when_candidate_rejected",
            source_evaluation_run_id="evaluation-fixture", evidence_files=self.current_evidence())

    def current_evidence(self):
        return {name: runtime.file_sha(self.root / name) for name in runtime.EVIDENCE_FILES}

    def model_loaders(self):
        stack = ExitStack()
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        self.auto_model = stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model))
        self.tokenizer = stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(
            chat_template="frozen template")))
        self.device = stack.enter_context(patch("src.runtime.resolve_device", return_value=SimpleNamespace(type="cpu")))
        stack.enter_context(patch("src.runtime.resolve_dtype", return_value="fake dtype"))
        stack.enter_context(patch("src.runtime.set_seed"))
        return stack

    def assert_no_model_loading(self):
        self.auto_model.assert_not_called()
        self.tokenizer.assert_not_called()
        self.device.assert_not_called()

    def test_frozen_decision_calls_full_saved_report_validation_and_keeps_rejection(self):
        with (patch("evaluation.cli.validated_reports", return_value=self.reports) as validate,
              patch("evaluation.solution.quality_gate", side_effect=ValueError("behavior failed")) as gate,
              patch("evaluation.cli.generate", side_effect=AssertionError("No inference allowed"))):
            reports, result, evidence = self.frozen_evaluation()
        validate.assert_called_once_with()
        gate.assert_called_once_with(self.reports["baseline"], self.reports["candidate"], ["fixture-slice"])
        self.assertEqual(reports, self.reports)
        self.assertEqual(result, {"passed": False, "reason": "behavior failed"})
        self.assertEqual(evidence, self.current_evidence())

    def test_lightweight_environment_delegates_descriptors_to_pinned_python_and_keeps_failures(self):
        with (patch.object(runtime, "sys", SimpleNamespace(prefix=str(self.root / "prefect-env"))),
              patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run):
            runtime.verify_release_evidence(self.candidate_manifest, self.release_report, self.workspace)
            call = run.call_args
            self.assertEqual(call.args[0][:4], [str(self.root / ".venv/bin/python"), "-B", "-m", "hw8910.runtime"])
            self.assertEqual(call.kwargs["cwd"], self.root)
            self.assertEqual(call.kwargs["timeout"], 180)
            self.assertEqual(runtime.canonical(dict(manifest=self.candidate_manifest, report=self.release_report)),
                             call.kwargs["input"])
            run.return_value = SimpleNamespace(returncode=1)
            with self.assertRaises(ValueError):
                runtime.verify_release_evidence(self.candidate_manifest, self.release_report, self.workspace)

    def test_valid_rejected_evidence_is_verified_but_cannot_be_published(self):
        self.assertIsNone(runtime.verify_release_evidence(self.candidate_manifest, self.release_report, self.workspace))
        with self.assertRaises(ValueError):
            validate_release(self.candidate_manifest, self.release_report)

    def test_fake_pass_val_loss_or_synthetic_report_cannot_override_saved_rejection(self):
        mutations = [dict(passed=True), dict(passed="true"), dict(status="running"),
                     dict(passed=True, val_loss=0.001, quality_gate={"passed": True}),
                     dict(synthetic=True), dict(evaluation_run_id="another-evaluation"),
                     dict(metrics={"accuracy": 1.0, "format_rate": 1.0}),
                     dict(evidence_files={**self.current_evidence(), "metrics/candidate-outputs.json": "f" * 64})]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = copy.deepcopy(self.release_report)
                report.update(mutation)
                self.manifest_loader.reset_mock()
                with self.assertRaises(ValueError):
                    runtime.verify_release_evidence(self.candidate_manifest, report, self.workspace)
                self.manifest_loader.assert_not_called()

    def test_changes_to_saved_raw_are_detected_without_trusting_declared_hashes(self):
        runtime.save(self.root / "metrics/candidate-outputs.json", {"tampered": True})
        with self.assertRaises(ValueError):
            runtime.verify_release_evidence(self.candidate_manifest, self.release_report, self.workspace)

    def test_wrong_model_spec_run_or_runtime_is_rejected_even_with_rehashed_descriptor(self):
        variants = []
        for key, value in (("model_id", "another-training"), ("training_run_id", "another-training"),
                           ("model_spec", {**self.candidate_spec, "version": "6"})):
            manifest = copy.deepcopy(self.candidate_manifest)
            manifest[key] = value
            variants.append(manifest)
        manifest = copy.deepcopy(self.candidate_manifest)
        manifest["runtime"]["dtype"] = "float32"
        manifest["runtime_sha256"] = runtime.sha(runtime.canonical(manifest["runtime"]))
        variants.append(manifest)
        for manifest in variants:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                runtime.verify_release_evidence(manifest, self.release_report, self.workspace)

    def test_report_model_and_all_hash_bindings_are_checked_by_evidence_verifier(self):
        for key in ("model_id", "model_sha256", "protocol_sha256", "golden_sha256", "runtime_sha256"):
            report = copy.deepcopy(self.release_report)
            report[key] = "another-model" if key == "model_id" else "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.verify_release_evidence(self.candidate_manifest, report, self.workspace)

    def test_adapter_changes_fail_even_after_updating_every_declared_asset_hash(self):
        for name, payload in (("adapter_model.safetensors", b"different model bytes"),
                              ("adapter_config.json", b'{"r": 99}')):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.candidate_manifest)
                report = copy.deepcopy(self.release_report)
                original = (self.adapter / name).read_bytes()
                (self.adapter / name).write_bytes(payload)
                changed["file_hashes"][name] = runtime.file_sha(self.adapter / name)
                changed["model_sha256"] = runtime.sha(runtime.canonical(changed["file_hashes"]))
                report["model_sha256"] = changed["model_sha256"]
                try:
                    with self.assertRaises(ValueError):
                        runtime.verify_release_evidence(changed, report, self.workspace)
                finally:
                    (self.adapter / name).write_bytes(original)

    def test_baseline_bootstrap_uses_pinned_assets_with_fake_loaders_only(self):
        with self.model_loaders():
            backend = runtime.ModelBackend(self.baseline_manifest, self.workspace)
            self.assertEqual(backend.loaded_id, self.baseline_manifest["model_id"])
            self.assertEqual(backend.runtime_sha256, self.baseline_manifest["runtime_sha256"])
            self.assertEqual(backend.max_new_tokens, 96)
            self.assertEqual(backend.calls, 0)
            self.auto_model.assert_called_once()
            self.tokenizer.assert_called_once_with(str(self.snapshot), local_files_only=True)
            backend.close()

    def test_rehashed_baseline_runtime_changes_are_rejected_before_model_loading(self):
        variants = []
        for key, value in (("dtype", "float32"), ("base_revision", "b" * 40),
                           ("base_model", "another-base"), ("tokenizer_sha256", "f" * 64),
                           ("template_sha256", "f" * 64)):
            manifest = copy.deepcopy(self.baseline_manifest)
            manifest["runtime"][key] = value
            variants.append(manifest)
        for key, value in (("system", "another instruction"), ("seed", 3), ("max_new_tokens", 32),
                           ("do_sample", True), ("enable_thinking", True)):
            manifest = copy.deepcopy(self.baseline_manifest)
            manifest["runtime"]["generation"][key] = value
            variants.append(manifest)
        manifest = copy.deepcopy(self.baseline_manifest)
        manifest["model_id"] = "fake accepted champion"
        variants.append(manifest)
        for manifest in variants:
            manifest["runtime_sha256"] = runtime.sha(runtime.canonical(manifest["runtime"]))
            with self.subTest(manifest=manifest), self.model_loaders():
                with self.assertRaises(ValueError):
                    runtime.ModelBackend(manifest, self.workspace)
                self.assert_no_model_loading()

    def test_baseline_protocol_golden_schema_and_origin_bindings_precede_model_load(self):
        mutations = (("protocol_sha256", "f" * 64), ("golden_sha256", "f" * 64),
                     ("schema_version", True), ("schema_version", 2),
                     ("acceptance_basis", "candidate quality PASS"),
                     ("source_evaluation_run_id", "another-evaluation"),
                     ("model_spec", {**self.baseline_spec, "revision": "b" * 40}))
        for key, value in mutations:
            manifest = copy.deepcopy(self.baseline_manifest)
            manifest[key] = value
            with self.subTest(key=key), self.model_loaders():
                with self.assertRaises(ValueError):
                    runtime.ModelBackend(manifest, self.workspace)
                self.assert_no_model_loading()

    def test_baseline_bootstrap_cannot_be_claimed_after_a_candidate_quality_pass(self):
        self.gate = {"passed": True}
        with self.model_loaders():
            with self.assertRaises(ValueError):
                runtime.ModelBackend(self.baseline_manifest, self.workspace)
            self.assert_no_model_loading()


if __name__ == "__main__":
    unittest.main()
