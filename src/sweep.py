"""Один sweep — родительский run, каждый кандидат — отдельный дочерний run."""
import argparse
import hashlib
import json
from importlib.metadata import version
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src.common import connect, load_params, provenance, save_json, verify_provenance
from src.sampling import verify_tensor_sources
from src.training_view import fingerprint


def train(*args, **kwargs):
    """Import accelerator code only inside the isolated candidate worker."""
    from src.experiment_train import train as fit
    return fit(*args, **kwargs)


def verify_source_proof(params, versions, proof):
    if (proof.get("passed") is not True
            or proof.get("tokenization_params_sha256") != versions.get("tokenization_params_sha256")
            or proof.get("tokenizer") != {"name": params["model"]["name"], "revision": params["model"]["revision"]}):
        raise ValueError("Нет актуальной полной проверки связи исходных строк с тензорами")
    for split in ("train", "val"):
        item = proof["splits"][split]
        if (item.get("source_sha256") != versions.get(f"source_{split}_sha256")
                or item.get("tensor_sha256") != versions.get(f"data_{split}_sha256")
                or item.get("all_input_ids_attention_masks_labels_equal") is not True
                or not item.get("source_records") or item["source_records"] != item.get("tensor_records")):
            raise ValueError("Preflight proof относится к другим исходным строкам или тензорам")


def run_candidate(params, candidate, versions, proof, view_proof):
    verify_source_proof(params, versions, proof)
    verify_provenance(params, versions)
    source_link_sha = hashlib.sha256(json.dumps(proof, sort_keys=True, allow_nan=False).encode()).hexdigest()
    view_sha = fingerprint(view_proof)
    mlflow.set_tags({**versions, "kind": "candidate", "candidate": candidate["name"],
                     "worker_pid": str(os.getpid()), "source_link_sha256": source_link_sha,
                     "training_view_sha256": view_sha})
    mlflow.log_dict(proof, "source-link-check.json")
    mlflow.log_dict(view_proof, "training-view-check.json")
    if not mlflow.get_run(mlflow.active_run().info.run_id).data.params:
        mlflow.log_params({"r": candidate["r"], "lr": candidate["lr"],
            "target_modules": ",".join(candidate["target_modules"]),
            **{f"train.{k}": v for k, v in params["train"].items()},
            "model": params["model"]["name"], "dtype": params["model"]["dtype"],
            "sampling.method": params["sampling"]["method"], "sampling.seed": params["sampling"]["seed"],
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
        results, details, answers = train(params, candidate, out, lambda values, step: mlflow.log_metrics(values, step=step),
                                          expected_view_sha256=view_sha)
        verify_provenance(params, versions)
        mlflow.log_metrics(results)
        mlflow.set_tags(details)
        mlflow.log_artifact(str(out / "sampling.json"))
        mlflow.log_artifact(str(out / "training-view.json"))
        mlflow.log_artifact(str(out.parent / "training-view.json.gz"))
        samples = {"prompt": params["generation"]["prompts"], "answer": answers}
        mlflow.log_table(samples, "samples.json")
        example = pd.DataFrame({"prompt": samples["prompt"][:1]})
        mlflow.pyfunc.log_model(artifact_path="model", python_model="src/adapter_model.py",
            code_paths=[str(package_code)],
            artifacts={"adapter": str(out)}, signature=infer_signature(example, answers[:1]),
            input_example=example,
            pip_requirements=[f"{name}=={version(name)}" for name in
                              ("mlflow", "torch", "transformers", "peft", "pandas", "numpy", "accelerate", "safetensors")])
        return {"run_id": mlflow.active_run().info.run_id, "name": candidate["name"],
                "sampling_sha256": details["sampling_sha256"], "training_view_sha256": details["training_view_sha256"],
                "worker_pid": os.getpid(), **results}


def run_candidate_isolated(params, candidate, versions, proof, view_proof):
    """Resume exactly one pre-created nested run in a fresh Python process."""
    run_id = mlflow.active_run().info.run_id
    with tempfile.TemporaryDirectory(prefix="hw67-candidate-process-") as directory:
        request = Path(directory) / "request.json"
        result = Path(directory) / "result.json"
        save_json(request, {"params": params, "candidate": candidate, "versions": versions,
                            "run_id": run_id, "tracking_uri": mlflow.get_tracking_uri(), "source_proof": proof,
                            "training_view_proof": view_proof})
        command = [sys.executable, "-m", "src.sweep", "--candidate-request", str(request),
                   "--candidate-result", str(result)]
        environment = dict(os.environ, PYTHONHASHSEED=str(params["train"]["seed"]),
                           MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
        subprocess.run(command, check=True, env=environment, timeout=7200)
        row = json.loads(result.read_text(encoding="utf-8"))
    if (row.get("run_id") != run_id or row.get("name") != candidate["name"] or not row.get("sampling_sha256")
            or row.get("training_view_sha256") != fingerprint(view_proof)):
        raise ValueError("Изолированный процесс вернул другой run или неполный sampling proof")
    verify_provenance(params, versions)
    return row


def candidate_worker(request_path, result_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    params = request["params"]
    mlflow.set_tracking_uri(request["tracking_uri"])
    with mlflow.start_run(run_id=request["run_id"]):
        row = run_candidate(params, request["candidate"], request["versions"], request["source_proof"],
                            request["training_view_proof"])
    save_json(result_path, row)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-request")
    parser.add_argument("--candidate-result")
    args = parser.parse_args(argv)
    if args.candidate_request or args.candidate_result:
        if not args.candidate_request or not args.candidate_result:
            parser.error("Оба пути worker request/result обязательны")
        candidate_worker(args.candidate_request, args.candidate_result)
        return
    p = load_params()
    if len(p["sweep"]) < 6 or len({x["name"] for x in p["sweep"]}) != len(p["sweep"]):
        raise SystemExit("Нужно минимум шесть кандидатов с уникальными именами")
    if any(p["train"][k] <= 0 for k in ("max_steps", "train_limit", "val_limit", "batch_size", "grad_accum", "eval_every")):
        raise SystemExit("Бюджеты и размеры батча должны быть положительными")
    versions = provenance(p)
    proof = verify_tensor_sources(p)
    from src.experiment_train import prepare_training_data
    _, _, _, _, view_proof = prepare_training_data(p)
    verify_provenance(p, versions)
    save_json("metrics/source-link-check.json", proof)
    save_json("metrics/training-view-check.json", {"passed": True, "training_view_sha256": fingerprint(view_proof),
                                                  "proof": view_proof})
    connect(p)
    rows = []
    with mlflow.start_run(run_name="sweep", tags={"kind": "sweep", **versions}) as parent:
        mlflow.log_dict(proof, "source-link-check.json")
        mlflow.log_dict(view_proof, "training-view-check.json")
        for candidate in p["sweep"]:
            with mlflow.start_run(run_name=candidate["name"], nested=True):
                rows.append(run_candidate_isolated(p, candidate, versions, proof, view_proof))
        if len({row["sampling_sha256"] for row in rows}) != 1:
            raise ValueError("Кандидаты использовали разные выборки или порядок обработки")
        if len({row["training_view_sha256"] for row in rows}) != 1:
            raise ValueError("Кандидаты использовали разные представления обучающих ответов")
        mlflow.log_dict({"candidates": rows}, "comparison.json")
        result = {"parent_id": parent.info.run_id, "rows": rows, "provenance": versions}
    save_json("metrics/sweep.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
