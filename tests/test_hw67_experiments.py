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
from src.promote import gate, promote, select, validate_sampling, validate_checkpoints


PARAMS = yaml.safe_load((Path(__file__).resolve().parents[1] / "conf/hw67.yaml").read_text())
# Keep the original small/control regressions independent of production budgets.
PARAMS["data"].pop("auxiliary", None)
PARAMS["sampling"]["method"] = "stratified_shuffle"
PARAMS["train"].update(max_steps=120, train_limit=240, val_limit=60, batch_size=1,
                       grad_accum=2, eval_every=30, eval_batch_size=1)
PARAMS["train"].pop("epochs", None)
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

    def test_development_cannot_register_or_masquerade_as_fair_sweep(self):
        run = copy.deepcopy(self.runs[0])
        run.data.tags["kind"] = "development"
        with patch("src.promote.mlflow.register_model") as register:
            with self.assertRaises(ValueError): promote(self.client, run, "model", "candidate", 1.0)
            register.assert_not_called()
        summary = {**self.summary, "kind": "development"}
        with patch("src.promote.provenance", return_value=VERSIONS):
            with self.assertRaises(ValueError): select(self.client, PARAMS, summary)

    def test_checkpoint_proof_binds_best_loss_budget_and_real_weights(self):
        params = copy.deepcopy(PARAMS)
        params["train"].update(max_steps=2, eval_every=1)
        weights = b"unit adapter bytes"
        digest = hashlib.sha256(weights).hexdigest()
        sampling = expected_sampling_manifest(params)
        proof = {"policy": "best_val", "metric": "original_val_token_weighted_nll", "primary_val_rows": 60,
                 "primary_val_ids_sha256": sweep.fingerprint(sampling["val"]["ids"]), "auxiliary_val_rows": 0,
                 "training_steps": 2, "selected_step": 1, "selected_val_loss": 0.2, "end_val_loss": 0.4,
                 "checkpoints": [{"step": 0, "val_loss": 1.0}, {"step": 1, "val_loss": 0.2},
                                 {"step": 2, "val_loss": 0.4}], "adapter_sha256": digest}
        run = copy.deepcopy(self.runs[0])
        run.data.metrics.update(final_val_loss=0.2, end_val_loss=0.4, selected_step=1)
        run.data.tags["selected_adapter_sha256"] = digest
        mutation = None
        def download(run_id, artifact, destination):
            path = Path(destination, artifact)
            path.parent.mkdir(exist_ok=True, parents=True)
            actual = copy.deepcopy(proof)
            if mutation == "step": actual["selected_step"] = 2
            elif mutation == "budget": actual["training_steps"] = 1
            elif mutation == "metric": actual["selected_val_loss"] = 0.1
            elif mutation == "trajectory": actual["checkpoints"].pop()
            if artifact == "checkpoint.json": path.write_text(json.dumps(actual))
            elif artifact.endswith("MLmodel"):
                path.write_text(yaml.safe_dump({"flavors": {"python_function": {"artifacts": {"adapter": {"path": "adapter"}}}}}))
            elif artifact.endswith("manifest.json"): path.write_text(json.dumps({"checkpoint": actual}))
            else: path.write_bytes(weights + (b"altered" if mutation == "weights" else b""))
            return str(path)
        client = Mock(download_artifacts=Mock(side_effect=download))
        validate_checkpoints(client, params, [run], sampling)
        for mutation in ("step", "budget", "metric", "trajectory", "weights"):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_checkpoints(client, params, [run], sampling)

    def test_registration_after_loss_gate_sets_only_requested_alias(self):
        version = SimpleNamespace(version="2")
        with patch("src.promote.mlflow.register_model", return_value=version) as register:
            self.assertIs(promote(self.client, self.runs[0], "model", "candidate", 1.0), version)
            register.assert_called_once_with("runs:/run-0/model", "model")
            self.client.set_registered_model_alias.assert_called_once_with("model", "candidate", "2")

    def test_selection_rejects_stale_summary_config_budget_or_base(self):
        with (patch("src.promote.provenance", return_value=VERSIONS), patch("src.promote.validate_sampling"),
              patch("src.promote.validate_training_view"), patch("src.promote.validate_checkpoints")):
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

    def test_validation_batching_keeps_token_weighted_objective(self):
        examples = [{"input_ids": [0] + values, "attention_mask": [1] * (len(values) + 1),
                     "labels": [-100] + values} for values in ([1], [3, 3, 3], [2, 2])]
        model = TokenLossModel()
        expected = evaluate(model, examples, 0, torch.device("cpu"), batch_size=1)
        self.assertAlmostEqual(evaluate(model, examples, 0, torch.device("cpu"), batch_size=4), expected, places=6)
        self.assertTrue(model.training)

    def test_scheduler_warmup_cosine_and_epoch_budget_are_explicit(self):
        config = {"max_steps": 10, "warmup_ratio": 0.2, "lr_schedule": "cosine"}
        curve = [experiment_train.learning_rate_multiplier(step, config) for step in range(11)]
        self.assertEqual(curve[:3], [0.5, 1.0, 1.0])
        self.assertEqual(curve[-1], 0.0)
        self.assertTrue(all(0 <= value <= 1 for value in curve))
        self.assertEqual(curve[2:], sorted(curve[2:], reverse=True))
        params = copy.deepcopy(PARAMS)
        params["train"].update(epochs=2, batch_size=4, grad_accum=1, max_steps=6)
        self.assertEqual(experiment_train.check_epoch_budget(params, 9), 3)
        params["train"]["max_steps"] = 5
        with self.assertRaises(ValueError): experiment_train.check_epoch_budget(params, 9)

    def test_checkpoint_restores_the_selected_adapter_weights(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        cfg = Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        model = get_peft_model(Qwen3ForCausalLM(cfg), LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        checkpoint = experiment_train.AdapterCheckpoint()
        checkpoint.consider(model, 1, 0.25)
        selected = {key: value.clone() for key, value in get_peft_model_state_dict(model).items()}
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad: parameter.add_(2)
        checkpoint.consider(model, 2, 0.75)
        checkpoint.restore(model)
        self.assertEqual(checkpoint.step, 1)
        for key, value in get_peft_model_state_dict(model).items():
            self.assertTrue(torch.equal(value, selected[key]), key)
        with self.assertRaises(ValueError): checkpoint.consider(model, 3, float("nan"))

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
def fixture(params, config, versions, proof, view_proof, kind="candidate"):
    sweep.mlflow.set_tags({**versions, "kind": kind, "candidate": config["name"]})
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
                before = Path("metrics/sweep.json").read_bytes()
                with (patch.object(sweep, "load_params", return_value=PARAMS),
                      patch.object(sweep, "connect"), patch.object(sweep, "provenance", return_value=VERSIONS),
                      patch.object(sweep, "verify_provenance"),
                      patch.object(sweep, "verify_tensor_sources", return_value={"passed": True}),
                      patch.object(experiment_train, "prepare_training_data", return_value=([], [], 0, {}, {"fixture": True})),
                      patch.object(sweep.subprocess, "run", side_effect=launch),
                      patch.object(sweep.mlflow, "register_model") as register,
                      contextlib.redirect_stdout(io.StringIO())):
                    sweep.main(["--development"])
                    register.assert_not_called()
                self.assertEqual(Path("metrics/sweep.json").read_bytes(), before)
                development = MlflowClient().search_runs([parent.info.experiment_id], filter_string="tags.kind = 'development'")
                self.assertEqual(len(development), 2)
                development_parent = next(run for run in development if "mlflow.parentRunId" not in run.data.tags)
                child = next(run for run in development if "mlflow.parentRunId" in run.data.tags)
                self.assertEqual(child.data.tags["mlflow.parentRunId"], development_parent.info.run_id)
                self.assertEqual(child.data.tags["candidate"], "all-r8-low")
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
        control = copy.deepcopy(PARAMS)
        control["train"].update(max_steps=4, train_limit=32, val_limit=8, eval_every=2)
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

    def test_auxiliary_ids_are_consumed_each_epoch_and_aux_changes_invalidate_provenance(self):
        params = copy.deepcopy(PARAMS)
        params["sampling"]["method"] = "full_shuffle"
        params["train"].update(train_limit=4, val_limit=2, epochs=2, max_steps=4, batch_size=4, grad_accum=1)
        blobs = {}
        with tempfile.TemporaryDirectory() as directory:
            for split, count, year in (("train", 4, 1999), ("val", 2, 2001)):
                rows = [{"id": f"{split}-{index}", "topic": f"release_year:{year}", "messages": [
                    {"role": "user", "content": f"Original {split} story number {index}."},
                    {"role": "assistant", "content": "драма, комедия"}]} for index in range(count)]
                path = Path(directory, split + ".jsonl")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                params["data"][f"source_{split}"] = str(path)
                blobs[params["data"][split]] = {"model": params["model"]["name"], "pad_token_id": 0,
                                                "examples": [{"id": row["id"]} for row in rows]}
            aux = {"version": 1, "train": [
                {"id": "aux-refuse", "input": "Explain triangle angles.", "expected": "REFUSE",
                 "kind": "refusal", "source": "author_synthetic", "group_id": "family:math"},
                {"id": "aux-genre", "input": "A solemn reconciliation between relatives.", "expected": "драма",
                 "kind": "genre", "source": "author_synthetic", "group_id": "family:relatives"}],
                "dev": [{"id": "aux-dev", "input": "An absurd workplace accident with hilarious consequences.",
                         "expected": "комедия", "kind": "genre", "source": "author_synthetic", "group_id": "family:office"}]}
            path = Path(directory, "auxiliary.json")
            path.write_text(json.dumps(aux))
            params["data"]["auxiliary"] = str(path)
            with patch.object(experiment_train, "load_split", side_effect=lambda path: copy.deepcopy(blobs[path])):
                train, val, _, sampling = prepare_data(params)
                manifest = expected_sampling_manifest(params)
            self.assertEqual((len(train), len(val)), (6, 3))
            self.assertEqual(sampling["val"]["original_selected_rows"], 2)
            self.assertEqual(manifest["processed_train"]["count"], 12)
            self.assertEqual(manifest["processed_train"]["unique_count"], 6)
            from collections import Counter
            self.assertEqual(set(Counter(manifest["processed_train"]["ids"]).values()), {2})
            self.assertEqual(manifest["processed_train"]["kind_counts"]["refusal"], 2)
            before = provenance(params)
            aux["train"][1]["expected"] = "комедия"
            path.write_text(json.dumps(aux))
            after = provenance(params)
            self.assertNotEqual(before["auxiliary_sha256"], after["auxiliary_sha256"])
            self.assertNotEqual(before["config_sha256"], after["config_sha256"])
            self.assertEqual(before["data_train_sha256"], after["data_train_sha256"])
            with self.assertRaises(ValueError): verify_provenance(params, before)

    def test_real_tiny_training_records_exact_main_and_control_consumption(self):
        control = copy.deepcopy(PARAMS)
        control["train"].update(max_steps=4, train_limit=32, val_limit=8, eval_every=2)
        full = copy.deepcopy(PARAMS)
        full["sampling"]["method"] = "full_shuffle"
        full["train"].update(max_steps=2, train_limit=6, val_limit=6, eval_every=1,
                             batch_size=4, grad_accum=2, eval_batch_size=4, epochs=2)
        for config, expected_count in ((PARAMS, 240), (control, 8), (full, 12)):
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
                self.assertEqual(metrics["processed_unique_examples"], min(expected_count, len(selected)))
                self.assertEqual(metrics["processed_tokens"], 3 * expected_count)
                self.assertEqual(metrics["processed_supervised_tokens"], expected_count)
                if config is full:
                    expected_order = [row["id"] for epoch in range(2)
                                      for row in experiment_train.ordered_epoch(selected, params, epoch)]
                    self.assertEqual(proof["processed_train"]["ids"], expected_order)
                    self.assertNotEqual(expected_order[:6], expected_order[6:])
                else:
                    self.assertEqual(proof["processed_train"]["ids"], sampling["train"]["ids"][:expected_count])
                self.assertEqual(proof["epochs_started"], params["train"].get("epochs", 1))
                checkpoint = json.loads(Path(directory, "checkpoint.json").read_text())
                self.assertEqual(checkpoint["training_steps"], metrics["steps"])
                self.assertEqual(checkpoint["selected_val_loss"], metrics["final_val_loss"])
                self.assertEqual(checkpoint["adapter_sha256"], hashlib.sha256(Path(directory, "adapter_model.safetensors").read_bytes()).hexdigest())
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
