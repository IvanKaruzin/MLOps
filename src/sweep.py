"""Один sweep — родительский run, каждый кандидат — отдельный дочерний run."""
import json
from importlib.metadata import version
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src.common import connect, load_params, provenance, save_json
from src.experiment_train import train


def run_candidate(params, candidate, versions):
    mlflow.set_tags({**versions, "kind": "candidate", "candidate": candidate["name"]})
    if not mlflow.get_run(mlflow.active_run().info.run_id).data.params:
        mlflow.log_params({"r": candidate["r"], "lr": candidate["lr"],
            "target_modules": ",".join(candidate["target_modules"]),
            **{f"train.{k}": v for k, v in params["train"].items()},
            "model": params["model"]["name"], "dtype": params["model"]["dtype"],
            "alpha_over_r": params["lora"]["alpha_over_r"], "dropout": params["lora"]["dropout"]})
    mlflow.log_dict(params, "config.json")
    # Достаточно sha для коммита, но dirty-код дополнительно сохраняется целиком.
    for source in sorted(Path("src").glob("*.py")):
        mlflow.log_artifact(str(source), "source")
    mlflow.log_artifact("uv.lock")
    with tempfile.TemporaryDirectory(prefix="hw6-train-") as tmp:
        out = Path(tmp) / "adapter"
        package_code = Path(tmp) / "code" / "src"
        package_code.mkdir(parents=True)
        # Bundle only the shared prompt helper, so isolated registry loading
        # uses the same template without depending on the repository checkout.
        for filename in ("__init__.py", "prompt.py"):
            shutil.copy2(Path("src") / filename, package_code / filename)
        results, details, answers = train(params, candidate, out, lambda values, step: mlflow.log_metrics(values, step=step))
        mlflow.log_metrics(results)
        mlflow.set_tags(details)
        samples = {"prompt": params["generation"]["prompts"], "answer": answers}
        mlflow.log_table(samples, "samples.json")
        example = pd.DataFrame({"prompt": samples["prompt"][:1]})
        mlflow.pyfunc.log_model(artifact_path="model", python_model="src/adapter_model.py",
            code_paths=[str(package_code)],
            artifacts={"adapter": str(out)}, signature=infer_signature(example, answers[:1]),
            input_example=example,
            pip_requirements=[f"{name}=={version(name)}" for name in
                              ("mlflow", "torch", "transformers", "peft", "pandas", "numpy", "accelerate", "safetensors")])
        return {"run_id": mlflow.active_run().info.run_id, "name": candidate["name"], **results}


def main():
    p = load_params()
    if len(p["sweep"]) < 6 or len({x["name"] for x in p["sweep"]}) != len(p["sweep"]):
        raise SystemExit("Нужно минимум шесть кандидатов с уникальными именами")
    if any(p["train"][k] <= 0 for k in ("max_steps", "train_limit", "val_limit", "batch_size", "grad_accum", "eval_every")):
        raise SystemExit("Бюджеты и размеры батча должны быть положительными")
    connect(p)
    versions = provenance(p)
    rows = []
    with mlflow.start_run(run_name="sweep", tags={"kind": "sweep", **versions}) as parent:
        for candidate in p["sweep"]:
            with mlflow.start_run(run_name=candidate["name"], nested=True):
                rows.append(run_candidate(p, candidate, versions))
        mlflow.log_dict({"candidates": rows}, "comparison.json")
        result = {"parent_id": parent.info.run_id, "rows": rows, "provenance": versions}
    save_json("metrics/sweep.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
