"""Additional development and sealed-holdout diagnostics; never changes aliases."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from evaluation import cli
from evaluation.contract import fingerprint, read_json, read_rows, strict_json_loads, target_for, write_json
from evaluation.solution import score, validate_model_spec


# Revised on 2026-11-02 at the user's request, after observing development
# results: synthetic accuracy 0.80 -> 0.75. This is a post-hoc acceptance
# policy; archived evaluations retain their original thresholds and decisions.
GUARDS = {
    "source_micro_f1": 0.50, "source_macro_f1": 0.35, "dev_source_accuracy": 0.10,
    "synthetic_genre_accuracy": 0.75, "refusal_accuracy": 0.95,
    "unseen_refusal_accuracy": 0.90, "format_rate": 0.90, "false_refusal_rate": 0.02,
    "final_source_delta_micro_f1": 0.05, "final_source_delta_macro_f1": 0.05,
    "final_source_delta_accuracy": 0.05,
}
PINNED_REFERENCE = {"kind": "adapter", "name": "my-qwen-adapter", "version": "5",
                    "run_id": "724f89d55d204ee1a24b6373fb6539dd", "uri": "models:/my-qwen-adapter/5"}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def catalogue(config):
    """Read only the shared choices, never generate on the official questions."""
    with Path(config["golden"]).open(encoding="utf-8") as stream:
        first = next((line for line in stream if line.strip()), None)
    if first is None:
        raise ValueError("The frozen choices catalogue is empty")
    row = strict_json_loads(first)
    target_for(row)
    if row.get("task") != "multilabel":
        raise ValueError("Additional diagnostics require the frozen multilabel catalogue")
    return row["choices"]


def diagnostic_rows(records, choices):
    if not isinstance(records, list) or not records:
        raise ValueError("The diagnostic split must be nonempty")
    rows, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Diagnostic records must be objects")
        for key in ("id", "input", "expected", "group_id", "source", "kind"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Diagnostic rows require {key}")
        if record["id"] in seen:
            raise ValueError("Diagnostic ids must be unique")
        seen.add(record["id"])
        if record["kind"] not in ("genre", "refusal") or record["source"] not in ("pinned_source", "author_synthetic"):
            raise ValueError("Unknown diagnostic kind or source")
        if (record["kind"] == "refusal") != (record["expected"] == "REFUSE"):
            raise ValueError("Refusal kind and uppercase REFUSE target must agree")
        language = record.get("language")
        if language not in ("en", "ru"):
            raise ValueError("Diagnostic rows require an explicit en/ru language")
        slices = ["genre" if record["kind"] == "genre" else "refusal", f"language_{language}"]
        if record["kind"] == "genre":
            slices.append("source_genre" if record["source"] == "pinned_source" else "synthetic_genre")
            if language == "ru":
                slices.append("ru_positive")
        else:
            if type(record.get("novel_topic")) is not bool:
                raise ValueError("Refusal rows require an explicit novel_topic boolean")
            slices.append("unseen_refusal" if record["novel_topic"] else "seen_refusal")
        row = {key: record[key] for key in ("id", "input", "expected", "group_id", "source")}
        row.update(task="multilabel", choices=choices, slices=slices)
        target_for(row)
        rows.append(row)
    return rows


def load_split(params, config, split, allow_final=False):
    # Check authorization before opening even the bundle's holdout metadata.
    if split == "holdout" and not allow_final:
        raise ValueError("Holdout access requires explicit --allow-final after the winner is frozen")
    bundle_path = Path(params["data"]["auxiliary"])
    before = file_sha(bundle_path)
    bundle = read_json(bundle_path)
    dataset = {"auxiliary_sha256": before, "split": split}
    if split == "dev":
        records = bundle["dev"]
    elif split == "holdout":
        manifest = bundle["manifest"]["holdout"]
        path = Path(manifest["path"])
        expected_path = Path("data/eval/improvement-holdout.jsonl").resolve()
        if (path.resolve() != expected_path or manifest.get("sealed_before_training") is not True
                or file_sha(path) != manifest.get("sha256")):
            raise ValueError("Sealed holdout path, bytes or declaration differ from the frozen manifest")
        records = read_rows(path)
        if len(records) != manifest.get("n") or file_sha(path) != manifest["sha256"]:
            raise ValueError("Sealed holdout changed while reading or has the wrong row count")
        dataset["holdout_sha256"] = manifest["sha256"]
    else:
        raise ValueError("Split must be dev or holdout")
    if file_sha(bundle_path) != before:
        raise ValueError("Auxiliary bundle changed while reading")
    rows = diagnostic_rows(records, catalogue(config))
    dataset.update(rows_sha256=fingerprint(rows), rows=len(rows), choices_sha256=fingerprint(rows[0]["choices"]))
    return rows, dataset


def validate_manifest(manifest, baseline, params, config):
    validate_model_spec(baseline)
    model, generation = manifest["model"], manifest["generation"]
    if baseline["kind"] != "base":
        raise ValueError("The immutable diagnostic baseline must identify the pinned base")
    if ((model["name"], model["revision"]) != (baseline["name"], baseline["revision"])
            or model["dtype"] != params["model"]["dtype"]
            or model["enable_thinking"] != config["enable_thinking"]
            or generation["system"] != config["system"] or generation["max_new_tokens"] != config["max_new_tokens"]
            or config["max_new_tokens"] != 96 or os.environ.get("HW6_DTYPE", model["dtype"]) != model["dtype"]):
        raise ValueError("Model package and frozen generation protocol differ")


def training_package(client, run_id, directory):
    import yaml
    run = client.get_run(run_id)
    if (run.info.status != "FINISHED" or run.data.tags.get("kind") not in ("development", "candidate")
            or not run.data.tags.get("mlflow.parentRunId")):
        raise ValueError("Expected a finished training child run")
    parent = client.get_run(run.data.tags["mlflow.parentRunId"])
    if parent.info.status != "FINISHED":
        raise ValueError("The training parent must be finished")
    root = Path(client.download_artifacts(run_id, "model", directory)).resolve()
    metadata = yaml.safe_load((root / "MLmodel").read_text())
    adapter = (root / metadata["flavors"]["python_function"]["artifacts"]["adapter"]["path"]).resolve()
    if not adapter.is_relative_to(root):
        raise ValueError("Adapter path escapes the logged model package")
    manifest = read_json(adapter / "manifest.json")
    digest = file_sha(adapter / "adapter_model.safetensors")
    if (manifest.get("checkpoint", {}).get("adapter_sha256") != digest
            or run.data.tags.get("selected_adapter_sha256") != digest
            or any(not manifest.get(key) or manifest[key] != run.data.tags.get(key)
                   for key in ("sampling_sha256", "training_view_sha256"))):
        raise ValueError("Training checkpoint and packaged adapter weights differ")
    return manifest, {"kind": "training_run", "run_id": run_id, "uri": f"runs:/{run_id}/model",
                      "weights_sha256": digest, "manifest_sha256": fingerprint(manifest),
                      "parent_run_id": parent.info.run_id,
                      "training_provenance": {key: run.data.tags.get(key) for key in
                                              ("code_sha256", "config_sha256", "auxiliary_sha256",
                                               "sampling_sha256", "training_view_sha256")}}


def reference_spec(client, uri):
    match = re.fullmatch(r"models:/([^/@]+)/([1-9][0-9]*)", uri)
    if not match or uri != PINNED_REFERENCE["uri"]:
        raise ValueError("This improvement episode requires the preregistered my-qwen-adapter/5 reference")
    version = client.get_model_version(match[1], match[2])
    spec = dict(kind="adapter", name=match[1], version=match[2], run_id=version.run_id, uri=uri)
    if spec != PINNED_REFERENCE:
        raise ValueError("Reference registry version is not bound to the preregistered v5 training run")
    cli.registry_version(client, spec)
    return spec


def release_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def generation_error(exc):
    message = re.sub(r"https?://\S+", "[URL]", str(exc))
    message = re.sub(r"(?i)\b(?:bearer\s+|(?:token|password|api[_-]?key)\s*[:=]\s*)\S+", "[REDACTED]", message)
    return f"{type(exc).__name__}: {message[:1000]}"


def generate_training(uri, config, rows):
    import mlflow
    import pandas as pd
    from src.runtime import set_seed
    started = time.monotonic()
    try:
        model = mlflow.pyfunc.load_model(uri)
    except Exception as exc:
        release_memory()
        return [dict(id=row["id"], raw="", error=generation_error(exc),
                     latency_seconds=time.monotonic() - started) for row in rows]
    set_seed(config["seed"])
    outputs = []
    try:
        for row in rows:
            started = time.monotonic()
            raw, error = "", None
            try:
                answer = model.predict(pd.DataFrame({"prompt": [row["input"]]}))
                if isinstance(answer, str) or len(answer) != 1 or not isinstance(answer[0], str):
                    raise ValueError("Model must return exactly one raw string per prompt")
                raw = answer[0]
            except Exception as exc:
                error = generation_error(exc)
            outputs.append(dict(id=row["id"], raw=raw, error=error, latency_seconds=time.monotonic() - started))
    finally:
        del model
        release_memory()
    return outputs


def generate_reference(spec, manifest, config, rows):
    started = time.monotonic()
    try:
        return cli.generate(spec, manifest, config, rows)
    except Exception as exc:
        release_memory()
        return [dict(id=row["id"], raw="", error=generation_error(exc),
                     latency_seconds=time.monotonic() - started) for row in rows]


def diagnostics(rows, outputs):
    report = score(rows, outputs)
    report["outputs_sha256"] = fingerprint(outputs)
    report["false_refusals"] = {}
    for name in ("genre", "ru_positive"):
        items = [item for item in report["per_item"] if name in item["slices"]]
        count = sum(isinstance(item["parsed"], dict) and item["parsed"].get("answer") == "REFUSE" for item in items)
        report["false_refusals"][name] = dict(n=len(items), count=count, rate=count / len(items) if items else None)
    return report


def additional_guard(reports, split):
    candidate = reports["candidate"]
    checks = []
    def check(name, actual, threshold, n, maximum=False):
        valid = type(actual) in (int, float) and math.isfinite(actual) and n > 0
        passes = valid and ((actual <= threshold if maximum else actual >= threshold)
                            or math.isclose(actual, threshold, rel_tol=0, abs_tol=1e-12))
        checks.append(dict(name=name, actual=actual, threshold=threshold, n=n,
                           passed=bool(passes)))
    def minimum(group, metric, threshold):
        values = candidate["slices"].get(group, {})
        check(f"{group}.{metric}", values.get(metric), threshold, values.get("n", 0))
    minimum("source_genre", "micro_f1", GUARDS["source_micro_f1"])
    minimum("source_genre", "macro_f1", GUARDS["source_macro_f1"])
    if split == "dev":
        minimum("source_genre", "accuracy", GUARDS["dev_source_accuracy"])
    minimum("synthetic_genre", "accuracy", GUARDS["synthetic_genre_accuracy"])
    minimum("refusal", "accuracy", GUARDS["refusal_accuracy"])
    for group in ("genre", "refusal"):
        minimum(group, "format_rate", GUARDS["format_rate"])
    for group in ("genre", "ru_positive"):
        values = candidate["false_refusals"][group]
        check(f"{group}.false_refusal_rate", values["rate"], GUARDS["false_refusal_rate"], values["n"], maximum=True)
    for role, report in reports.items():
        check(f"{role}.generation_errors", report["overall"]["errors"], 0, report["overall"]["n"], maximum=True)
    if split == "holdout":
        minimum("unseen_refusal", "accuracy", GUARDS["unseen_refusal_accuracy"])
        reference = reports.get("reference", {}).get("slices", {}).get("source_genre", {})
        actual = candidate["slices"].get("source_genre", {})
        for metric in ("micro_f1", "macro_f1", "accuracy"):
            delta = actual[metric] - reference[metric] if metric in actual and metric in reference else None
            check(f"source_genre.delta_reference_{metric}", delta, GUARDS[f"final_source_delta_{metric}"],
                  min(actual.get("n", 0), reference.get("n", 0)))
    return dict(passed=all(check["passed"] for check in checks), split=split, thresholds=dict(GUARDS), checks=checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--reference-uri", default=PINNED_REFERENCE["uri"])
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--allow-final", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "holdout" and not args.allow_final:
        parser.error("Sealed holdout requires --allow-final after the development winner is frozen")
    params, client = cli.tracking()
    config, baseline = read_json("evaluation/params.json"), read_json(cli.BASELINE)
    if (type(config.get("seed")) is not int or type(config.get("enable_thinking")) is not bool
            or not isinstance(config.get("system"), str) or not config["system"].strip()
            or type(config.get("max_new_tokens")) is not int or config["max_new_tokens"] != 96):
        raise ValueError("Diagnostics require the complete frozen generation protocol")
    rows, dataset = load_split(params, config, args.split, args.allow_final)
    reference = reference_spec(client, args.reference_uri)
    reference_manifest, reference_adapter = cli.manifest_for(reference)
    validate_manifest(reference_manifest, baseline, params, config)
    specs = {"reference": {**reference, "weights_sha256": file_sha(reference_adapter / "adapter_model.safetensors"),
                            "manifest_sha256": fingerprint(reference_manifest)}}
    reports, outputs = {}, {}
    with tempfile.TemporaryDirectory(prefix="hw67-improvement-") as directory:
        manifest, training = training_package(client, args.training_run_id, directory)
        validate_manifest(manifest, baseline, params, config)
        if training["training_provenance"].get("auxiliary_sha256") != dataset["auxiliary_sha256"]:
            raise ValueError("Training run and diagnostic split refer to different auxiliary bundles")
        specs["candidate"] = training
        if args.include_base:
            specs["base"] = baseline
        import mlflow
        mlflow.set_experiment("hw67-evaluation")
        kind = "development-evaluation" if args.split == "dev" else "generalization-evaluation"
        code_paths = {"evaluation/improvement.py": Path(__file__),
                      **{name: Path(name) for name in ("evaluation/contract.py", "evaluation/solution.py", "evaluation/cli.py")}}
        code = {name: file_sha(path) for name, path in code_paths.items()}
        scorer = {name: value for name, value in code.items() if name in ("evaluation/contract.py", "evaluation/solution.py")}
        evidence = dict(dataset=dataset, config=config, config_sha256=fingerprint(config),
                        scorer_sha256=fingerprint(scorer), evaluation_code=code, models=specs,
                        baseline=baseline, baseline_sha256=file_sha(cli.BASELINE))
        protocol_sha = fingerprint(evidence)
        with mlflow.start_run(run_name=kind, tags=dict(kind=kind, training_run_id=args.training_run_id,
                                                     protocol_sha256=protocol_sha)) as run:
            for role, spec in specs.items():
                generated = (generate_training(training["uri"], config, rows) if role == "candidate"
                             else generate_reference(reference if role == "reference" else baseline, manifest, config, rows))
                outputs[role] = generated
                reports[role] = diagnostics(rows, generated)
                mlflow.log_dict({"outputs": generated}, f"diagnostics/{role}-outputs.json")
                mlflow.log_dict(reports[role], f"diagnostics/{role}-report.json")
            guard = additional_guard(reports, args.split)
            if (read_json("evaluation/params.json") != config or file_sha(cli.BASELINE) != evidence["baseline_sha256"]
                    or file_sha(params["data"]["auxiliary"]) != dataset["auxiliary_sha256"]
                    or any(file_sha(code_paths[name]) != digest for name, digest in code.items())
                    or (args.split == "holdout" and file_sha("data/eval/improvement-holdout.jsonl") != dataset["holdout_sha256"])):
                raise ValueError("Diagnostic data, baseline, configuration or scoring code changed during generation")
            summary = dict(evaluation_run_id=run.info.run_id, passed=guard["passed"], split=args.split,
                           protocol_sha256=protocol_sha, evidence=evidence, additional_guard=guard,
                           reports={role: dict(overall=report["overall"], slices=report["slices"],
                                               false_refusals=report["false_refusals"], outputs_sha256=report["outputs_sha256"])
                                    for role, report in reports.items()})
            mlflow.log_dict(summary, "diagnostics/summary.json")
            mlflow.log_dict({"choices": rows[0]["choices"],
                             "rows": [{key: value for key, value in row.items() if key != "choices"} for row in rows]},
                            "diagnostics/rows.json")
            mlflow.set_tags(dict(decision="pass" if guard["passed"] else "reject"))
        if args.split == "holdout":
            write_json("metrics/generalization-eval.json", summary)
    compact = dict(evaluation_run_id=summary["evaluation_run_id"], split=args.split, passed=summary["passed"],
                   failed_checks=[check["name"] for check in guard["checks"] if not check["passed"]])
    print(json.dumps(compact, ensure_ascii=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
