"""Compare pinned models only on the same verified free-genre protocol."""

import csv
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = ("accuracy", "format_rate", "precision", "recall", "micro_f1", "macro_f1")


def paired_changes(before, after):
    """Count changes on paired questions, without treating two scores as independent."""
    a = {item["id"]: item for item in before["per_item"]}
    b = {item["id"]: item for item in after["per_item"]}
    if (len(a) != len(before["per_item"]) or len(b) != len(after["per_item"])
            or not a or set(a) != set(b)):
        raise ValueError("Comparisons require the same nonempty unique question ids")
    for key in ("protocol_sha256", "golden_sha256", "baseline_spec_sha256"):
        if before[key] != after[key]:
            raise ValueError(f"Cannot compare different protocols: {key}")
    return {
        "n": len(a),
        "improved": [identity for identity in a if not a[identity]["correct"] and b[identity]["correct"]],
        "regressed": [identity for identity in a if a[identity]["correct"] and not b[identity]["correct"]],
        "metric_deltas": {key: after["overall"][key] - before["overall"][key] for key in METRICS},
    }


def main():
    from evaluation.cli import current_reports
    from evaluation.contract import read_json, write_json

    os.chdir(ROOT)
    current_reports()  # Recompute raw metrics and validate all model/protocol bindings first.
    roles = ("baseline", "previous", "sampling_control", "unaligned", "candidate")
    reports = {role: read_json(f"metrics/{role}-eval.json") for role in roles}
    comparisons = {
        "sampling_control_vs_previous": paired_changes(reports["previous"], reports["sampling_control"]),
        "candidate_vs_previous": paired_changes(reports["previous"], reports["candidate"]),
        "candidate_vs_sampling_control": paired_changes(reports["sampling_control"], reports["candidate"]),
        "candidate_vs_unaligned": paired_changes(reports["unaligned"], reports["candidate"]),
        "candidate_vs_baseline": paired_changes(reports["baseline"], reports["candidate"]),
    }
    table = []
    for role, report in reports.items():
        overall = report["overall"]
        model = report["model"]
        table.append({
            "role": role, "model": model.get("uri", model["name"]),
            "training_run_id": model.get("run_id", ""),
            "n": overall["n"], "correct": overall["correct"],
            "valid": overall["valid"], "errors": overall["errors"],
            **{key: overall[key] for key in METRICS},
            "behavioral_correct": report["behavioral"]["overall"]["correct"],
            "behavioral_n": report["behavioral"]["overall"]["n"],
        })
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(table[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    Path("docs/quality-comparison.csv").write_text(output.getvalue())
    class_rows = []
    for genre in reports["candidate"]["labels"]:
        class_rows.append({
            "genre": genre, "support": reports["candidate"]["overall"]["per_label"][genre]["support"],
            **{f"{role}.f1": report["overall"]["per_label"][genre]["f1"]
               for role, report in reports.items()},
        })
    per_class = io.StringIO()
    class_writer = csv.DictWriter(per_class, fieldnames=list(class_rows[0]), lineterminator="\n")
    class_writer.writeheader()
    class_writer.writerows(class_rows)
    Path("docs/per-label-comparison.csv").write_text(per_class.getvalue())
    result = {
        "protocol_sha256": reports["candidate"]["protocol_sha256"],
        "golden_sha256": reports["candidate"]["golden_sha256"],
        "evaluation_run_id": reports["candidate"]["evaluation_run_id"],
        "models": {role: report["model"] for role, report in reports.items()},
        "overall": table,
        "pairwise": comparisons,
        "slices": {role: report["slices"] for role, report in reports.items()},
        "limitations": [
            "Historical MCQ accuracy and free-genre metrics measure different tasks and are not subtracted.",
            "Training losses from prefix, stratified-small and stratified-main use different validation inputs.",
            "The common golden is deliberately balanced by era and label cardinality, not population-weighted.",
            "One training seed and one corpus do not establish a general causal effect or training variance.",
            "The main sweep changes sample size and step budget together; its delta is the combined change.",
            "Aligned versus unaligned main keeps sampled films and budget fixed, but changes both the training system instruction and target serialization.",
        ],
    }
    write_json("metrics/experiment-comparison.json", result)
    from mlflow import MlflowClient
    client = MlflowClient()
    for filename in ("docs/quality-comparison.csv", "docs/per-label-comparison.csv", "metrics/experiment-comparison.json"):
        client.log_artifact(result["evaluation_run_id"], filename, artifact_path="comparison")
    print(output.getvalue())


if __name__ == "__main__":
    main()
