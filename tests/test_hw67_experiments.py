"""Part A regressions without model downloads, GPUs or a tracking server."""

import contextlib
import copy
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import torch
import yaml
from mlflow import MlflowClient

from src import sweep
from src.common import validate_params
from src.experiment_train import evaluate
from src.promote import gate, promote, select


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
        with patch("src.promote.provenance", return_value=VERSIONS):
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
        for mutation in ("duplicate", "budget", "revision"):
            params = copy.deepcopy(PARAMS)
            if mutation == "duplicate":
                params["sweep"][1] = {**params["sweep"][0], "name": "different-name"}
            elif mutation == "budget":
                params["train"]["max_steps"] = True
            else:
                params["model"]["revision"] = "main"
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
                def candidate(params, config, versions):
                    mlflow.set_tags({**versions, "kind": "candidate", "candidate": config["name"]})
                    return {"run_id": mlflow.active_run().info.run_id, "name": config["name"]}
                with (patch.object(sweep, "load_params", return_value=PARAMS),
                      patch.object(sweep, "connect"), patch.object(sweep, "provenance", return_value=VERSIONS),
                      patch.object(sweep, "run_candidate", side_effect=candidate),
                      contextlib.redirect_stdout(io.StringIO())):
                    sweep.main()
                import json
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
        with (patch.object(sweep, "mlflow", client),
              patch.object(sweep, "train", return_value=({}, {}, ["answer", "answer"]))):
            sweep.run_candidate(PARAMS, PARAMS["sweep"][0], VERSIONS)
        client.pyfunc.log_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
