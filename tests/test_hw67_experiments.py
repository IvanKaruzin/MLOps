"""Part A regressions without model downloads, GPUs or a tracking server."""

import contextlib
import copy
import io
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import torch
import yaml
from mlflow import MlflowClient
from transformers import Qwen3Config, Qwen3ForCausalLM

from src import experiment_report, experiment_train, sweep
from src.common import provenance, validate_params, verify_provenance
from src.experiment_train import evaluate, expected_sampling_manifest, prepare_data, processed_summary
from src.promote import gate, promote, select, validate_sampling


PARAMS = yaml.safe_load((Path(__file__).resolve().parents[1] / "conf/hw67.yaml").read_text())
VERSIONS = {key: key + "-value" for key in
            ("code_sha256", "config_sha256", "data_train_sha256", "data_val_sha256")}


def run_for(candidate, index, parent_id="parent"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=f"run-{index}", status="FINISHED", experiment_id="experiment"),
        data=SimpleNamespace(
            tags={**VERSIONS, "candidate": candidate["name"], "base_revision": "a" * 40,
                  "mlflow.parentRunId": parent_id},
            params={"r": str(candidate["r"]), "lr": str(candidate["lr"]),
                    "target_modules": ",".join(candidate["target_modules"])},
            metrics={"base_val_loss": 1.0, "final_val_loss": 0.95 - 0.01 * index,
                     "steps": PARAMS["train"]["max_steps"]}))


class TokenLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask, labels):
        values = labels[:, 1:][labels[:, 1:] != -100].float()
        return SimpleNamespace(loss=((self.weight - values) ** 2).mean())


class ExperimentContract(unittest.TestCase):
    def setUp(self):
        self.runs = [run_for(candidate, i) for i, candidate in enumerate(PARAMS["sweep"])]
        self.parent = SimpleNamespace(
            info=SimpleNamespace(run_id="parent", status="FINISHED", experiment_id="experiment"),
            data=SimpleNamespace(tags=VERSIONS))
        self.client = Mock()
        self.client.get_run.return_value = self.parent
        self.client.search_runs.return_value = self.runs
        self.summary = {"parent_id": "parent", "provenance": dict(VERSIONS),
                        "rows": [{"run_id": run.info.run_id} for run in self.runs]}

    def test_loss_gate_refuses_before_registry_mutation(self):
        run = self.runs[0]
        with patch("src.promote.mlflow.register_model") as register:
            for threshold in (0.0, float("nan"), -1.0):
                with self.assertRaises(ValueError):
                    promote(self.client, run, "model", "candidate", threshold)
            register.assert_not_called()
            self.client.set_registered_model_alias.assert_not_called()
        run.data.metrics["final_val_loss"] = -0.1
        with self.assertRaises(ValueError):
            gate(run, 1.0)

    def test_champion_requires_answer_quality_gate(self):
        with patch("src.promote.mlflow.register_model") as register:
            with self.assertRaises(ValueError):
                promote(self.client, self.runs[0], "model", "champion", 1.0)
            register.assert_not_called()
            self.client.set_registered_model_alias.assert_not_called()

    def test_registration_after_loss_gate_sets_only_requested_alias(self):
        version = SimpleNamespace(version="2")
        with patch("src.promote.mlflow.register_model", return_value=version) as register:
            self.assertIs(promote(self.client, self.runs[0], "model", "candidate", 1.0), version)
            register.assert_called_once_with("runs:/run-0/model", "model")
            self.client.set_registered_model_alias.assert_called_once_with("model", "candidate", "2")

    def test_selection_rejects_stale_summary_config_budget_or_base(self):
        with (patch("src.promote.provenance", return_value=VERSIONS), patch("src.promote.validate_sampling"),
              patch("src.promote.validate_training_view")):
            self.assertEqual(select(self.client, PARAMS, self.summary).info.run_id, "run-5")
            for mutation in ("summary", "provenance", "budget", "parameters", "revision", "base_loss"):
                with self.subTest(mutation=mutation):
                    runs = copy.deepcopy(self.runs)
                    self.client.search_runs.return_value = runs
                    summary = copy.deepcopy(self.summary)
                    if mutation == "summary":
                        summary["rows"][0]["run_id"] = "unknown-run"
                    elif mutation == "provenance":
                        runs[0].data.tags.pop("data_train_sha256")
                    elif mutation == "budget":
                        runs[0].data.metrics["steps"] = 3
                    elif mutation == "parameters":
                        runs[0].data.params["r"] = "999"
                    elif mutation == "revision":
                        for run in runs:
                            run.data.tags["base_revision"] = "main"
                    else:
                        runs[0].data.metrics["base_val_loss"] = 2.0
                    with self.assertRaises(ValueError):
                        select(self.client, PARAMS, summary)

    def test_validation_rejects_duplicate_configs_and_invalid_budget(self):
        for mutation in ("duplicate", "budget", "revision", "sampling", "source"):
            params = copy.deepcopy(PARAMS)
            if mutation == "duplicate":
                params["sweep"][1] = {**params["sweep"][0], "name": "different-name"}
            elif mutation == "budget":
                params["train"]["max_steps"] = True
            elif mutation == "revision":
                params["model"]["revision"] = "main"
            elif mutation == "sampling":
                params["sampling"]["method"] = "prefix"
            else:
                params["data"]["source_val"] = params["data"]["source_train"]
            with self.assertRaises(ValueError):
                validate_params(params)

    def test_validation_weights_shifted_tokens_and_restores_mode(self):
        def item(tokens):
            return {"input_ids": [0] + tokens, "attention_mask": [1] * (len(tokens) + 1),
                    "labels": [-100] + tokens}
        model = TokenLossModel().eval()
        self.assertAlmostEqual(evaluate(model, [item([1]), item([3, 3, 3])], 0, torch.device("cpu")), 4.75)
        self.assertFalse(model.training)
        model.weight.data.fill_(float("nan"))
        with self.assertRaises(ValueError):
            evaluate(model, [item([1])], 0, torch.device("cpu"))
        self.assertFalse(model.training)

    def test_sweep_creates_six_direct_distinct_child_runs(self):
        previous_uri = mlflow.get_tracking_uri()
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                mlflow.set_tracking_uri(Path(directory, "tracking").as_uri())
                mlflow.set_experiment("isolated-hw67-regression")
                worker_code = '''
import os
import sys
from src import sweep
def fixture(params, config, versions, proof, view_proof):
    sweep.mlflow.set_tags({**versions, "kind": "candidate", "candidate": config["name"]})
    return {"run_id": sweep.mlflow.active_run().info.run_id, "name": config["name"],
            "sampling_sha256": "fixture-sampling", "training_view_sha256": sweep.fingerprint(view_proof),
            "worker_pid": os.getpid()}
sweep.run_candidate = fixture
sweep.main(sys.argv[1:])
'''
                actual_run = subprocess.run
                repository = Path(__file__).resolve().parents[1]
                def launch(command, **kwargs):
                    worker_command = [command[0], "-c", worker_code, *command[3:]]
                    return actual_run(worker_command, cwd=repository, **kwargs)
                with (patch.object(sweep, "load_params", return_value=PARAMS),
                      patch.object(sweep, "connect"), patch.object(sweep, "provenance", return_value=VERSIONS),
                      patch.object(sweep, "verify_provenance"),
                      patch.object(sweep, "verify_tensor_sources", return_value={"passed": True}),
                      patch.object(experiment_train, "prepare_training_data", return_value=([], [], 0, {}, {"fixture": True})),
                      patch.object(sweep.subprocess, "run", side_effect=launch),
                      contextlib.redirect_stdout(io.StringIO())):
                    sweep.main([])
                result = json.loads(Path("metrics/sweep.json").read_text())
                parent = MlflowClient().get_run(result["parent_id"])
                children = MlflowClient().search_runs(
                    [parent.info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
                self.assertEqual(len(children), 6)
                self.assertEqual({run.data.tags["candidate"] for run in children},
                                 {config["name"] for config in PARAMS["sweep"]})
                self.assertTrue(all(run.info.status == "FINISHED" for run in children))
                self.assertEqual(len({row["run_id"] for row in result["rows"]}), 6)
                self.assertEqual(len({row["worker_pid"] for row in result["rows"]}), 6)
                self.assertNotIn(os.getpid(), {row["worker_pid"] for row in result["rows"]})
            finally:
                os.chdir(cwd)
                mlflow.set_tracking_uri(previous_uri)

    def test_model_package_bundles_exact_shared_prompt_module(self):
        client = Mock()
        client.get_run.return_value.data.params = {}
        client.active_run.return_value.info.run_id = "unit-package"

        def log_model(**kwargs):
            source = Path(kwargs["code_paths"][0])
            self.assertEqual(source.name, "src")
            self.assertEqual({path.name for path in source.iterdir()}, {"__init__.py", "prompt.py"})
            for filename in ("__init__.py", "prompt.py"):
                self.assertEqual((source / filename).read_bytes(), Path("src", filename).read_bytes())
            self.assertEqual(kwargs["python_model"], "src/adapter_model.py")

        client.pyfunc.log_model.side_effect = log_model
        with (patch.object(sweep, "mlflow", client), patch.object(sweep, "verify_provenance"),
              patch.object(sweep, "verify_source_proof"),
              patch.object(sweep, "train", return_value=({}, {"sampling_sha256": "fixture", "training_view_sha256": "view"},
                                                           ["answer", "answer"]))):
            sweep.run_candidate(PARAMS, PARAMS["sweep"][0], VERSIONS, {}, {})
        client.pyfunc.log_model.assert_called_once()

    def test_sampling_uses_complete_splits_and_main_processes_every_selected_id(self):
        params = copy.deepcopy(PARAMS)
        examples, val, _, proof = prepare_data(params)
        self.assertEqual((len(examples), len(val)), (240, 60))
        self.assertEqual(set(proof["train"]["strata_counts"].values()), {40})
        self.assertEqual(set(proof["val"]["strata_counts"].values()), {10})
        identities = [example["id"] for example in examples]
        consumed = params["train"]["max_steps"] * params["train"]["batch_size"] * params["train"]["grad_accum"]
        report = processed_summary(identities[:consumed], proof["train"]["selected_records"])
        self.assertEqual((report["count"], report["unique_count"]), (240, 240))
        self.assertEqual(report["ids"], proof["train"]["ids"])
        control = yaml.safe_load((Path(__file__).resolve().parents[1] / "conf/hw67-control.yaml").read_text())
        small_train, small_val, _, small_proof = prepare_data(control)
        self.assertEqual((len(small_train), len(small_val)), (32, 8))
        self.assertTrue(set(small_proof["train"]["ids"]) <= set(proof["train"]["ids"]))
        self.assertTrue(set(small_proof["val"]["ids"]) <= set(proof["val"]["ids"]))
        control_count = control["train"]["max_steps"] * control["train"]["batch_size"] * control["train"]["grad_accum"]
        self.assertEqual(control_count, 8)

    def test_provenance_detects_metadata_change_with_identical_tensor_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "metadata.jsonl")
            path.write_text("first metadata\n")
            params = copy.deepcopy(PARAMS)
            params["data"]["source_train"] = str(path)
            before = provenance(params)
            path.write_text("changed metadata\n")
            after = provenance(params)
            self.assertEqual(before["data_train_sha256"], after["data_train_sha256"])
            self.assertNotEqual(before["config_sha256"], after["config_sha256"])
            with self.assertRaises(ValueError):
                verify_provenance(params, before)

    def test_real_tiny_training_records_exact_main_and_control_consumption(self):
        control = yaml.safe_load((Path(__file__).resolve().parents[1] / "conf/hw67-control.yaml").read_text())
        for config, expected_count in ((PARAMS, 240), (control, 8)):
            params = copy.deepcopy(config)
            params["model"].update(device="cpu", dtype="float32")
            params["train"]["response_format"] = "source"
            selected, val, _, sampling = prepare_data(params)
            def tiny_examples(items):
                return [{"id": item["id"], "input_ids": [0, 1, 2],
                         "attention_mask": [1, 1, 1], "labels": [-100, -100, 2]} for item in items]
            tokenizer = Mock(pad_token_id=0)
            tokenizer.save_pretrained.side_effect = lambda path: Path(path, "tokenizer_config.json").write_text("{}\n")
            def tiny_model(*args, **kwargs):
                cfg = Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
                cfg._commit_hash = params["model"]["revision"]
                return Qwen3ForCausalLM(cfg)
            observed = []
            with tempfile.TemporaryDirectory() as directory:
                with (patch.object(experiment_train, "prepare_data", return_value=(
                        tiny_examples(selected), tiny_examples(val), 0, copy.deepcopy(sampling))),
                      patch.object(experiment_train.AutoModelForCausalLM, "from_pretrained", side_effect=tiny_model),
                      patch.object(experiment_train.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                      patch("src.adapter_model.AdapterModel.predict", return_value=["fixture", "fixture"]),
                      contextlib.redirect_stdout(io.StringIO())):
                    metrics, _, _ = experiment_train.train(params, params["sweep"][0], directory,
                                                           lambda values, step: observed.append((step, dict(values))))
                proof = json.loads(Path(directory, "sampling.json").read_text())
                self.assertEqual(metrics["steps"], params["train"]["max_steps"])
                self.assertEqual(metrics["processed_examples"], expected_count)
                self.assertEqual(metrics["processed_unique_examples"], expected_count)
                self.assertEqual(metrics["processed_tokens"], 3 * expected_count)
                self.assertEqual(metrics["processed_supervised_tokens"], expected_count)
                self.assertEqual(proof["processed_train"]["ids"], sampling["train"]["ids"][:expected_count])
                self.assertEqual(proof["epochs_started"], 1)
                self.assertEqual([step for step, values in observed if "train_loss" in values],
                                 list(range(1, params["train"]["max_steps"] + 1)))
                self.assertEqual([step for step, values in observed if "val_loss" in values],
                                 list(range(0, params["train"]["max_steps"] + 1, params["train"]["eval_every"])))

    def test_comparison_csv_uses_lf_and_refuses_unverified_sweep(self):
        client = Mock()
        run = copy.deepcopy(self.runs[0])
        run.data.metrics.update(seconds=1, peak_memory_mb=2, adapter_mb=3, processed_examples=240,
                                processed_unique_examples=240, train_pool_rows=240, val_rows=60)
        run.data.tags["sampling_sha256"] = "fixture-sampling"
        run.data.tags["training_view_sha256"] = "fixture-view"
        client.get_run.return_value = run
        with tempfile.TemporaryDirectory() as directory:
            summary, output = Path(directory, "sweep.json"), Path(directory, "comparison.csv")
            summary.write_text(json.dumps({"rows": [{"run_id": run.info.run_id, "name": "fixture"}]}))
            with (patch.object(experiment_report, "load_params", return_value=PARAMS),
                  patch.object(experiment_report, "connect"), patch.object(experiment_report, "MlflowClient", return_value=client),
                  patch.object(experiment_report, "select") as verified, contextlib.redirect_stdout(io.StringIO())):
                experiment_report.main(["--summary", str(summary), "--out", str(output)])
                verified.assert_called_once()
                self.assertNotIn(b"\r", output.read_bytes())
                verified.side_effect = ValueError("stale sweep")
                with self.assertRaises(ValueError):
                    experiment_report.main(["--summary", str(summary), "--out", str(output)])

    def test_select_sampling_proof_rejects_changed_pools_counts_sources_and_workers(self):
        expected = expected_sampling_manifest(PARAMS)
        sha = hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest()
        original_runs = copy.deepcopy(self.runs)
        for index, run in enumerate(original_runs):
            run.data.tags.update(sampling_sha256=sha, worker_pid=str(index + 100), source_link_sha256="a" * 64,
                                 source_train_sha256=expected["train"]["source_sha256"],
                                 source_val_sha256=expected["val"]["source_sha256"])
            run.data.metrics.update(processed_examples=240, processed_unique_examples=240,
                                    train_pool_rows=240, val_rows=60)
        bad_manifest = False
        def download(run_id, artifact, directory):
            proof = copy.deepcopy(expected)
            if bad_manifest:
                proof["train"]["ids"].reverse()
            path = Path(directory, "sampling.json")
            path.write_text(json.dumps(proof))
            return str(path)
        client = Mock(download_artifacts=Mock(side_effect=download))
        with patch("src.experiment_train.expected_sampling_manifest", return_value=expected):
            validate_sampling(client, PARAMS, original_runs)
            for mutation in ("hash", "count", "unique", "pool", "source", "worker", "proof", "source_link"):
                runs = copy.deepcopy(original_runs)
                if mutation == "hash": runs[0].data.tags.pop("sampling_sha256")
                elif mutation == "count": runs[0].data.metrics["processed_examples"] = 239
                elif mutation == "unique": runs[0].data.metrics["processed_unique_examples"] = 239
                elif mutation == "pool": runs[0].data.metrics["val_rows"] = 8
                elif mutation == "source": runs[0].data.tags["source_train_sha256"] = "different"
                elif mutation == "worker": runs[0].data.tags["worker_pid"] = runs[1].data.tags["worker_pid"]
                elif mutation == "source_link": runs[0].data.tags.pop("source_link_sha256")
                else: bad_manifest = True
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_sampling(client, PARAMS, runs)
                bad_manifest = False


if __name__ == "__main__":
    unittest.main()
