"""Preserve a verified intermediate sweep before the next experiment replaces it."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    import yaml
    from mlflow import MlflowClient
    from evaluation.cli import registry_version
    from src.common import connect, save_json, validate_params
    from src.promote import select

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("control", "unaligned"), default="control")
    args = parser.parse_args(argv)
    stage = args.stage
    role = "sampling_control" if stage == "control" else "unaligned"
    os.chdir(ROOT)
    config_path = ROOT / f"conf/hw67-{stage}.yaml"
    params = yaml.safe_load(config_path.read_text())
    validate_params(params)
    connect(params)
    client = MlflowClient()
    summary = json.loads((ROOT / "metrics/sweep.json").read_text())
    best = select(client, params, summary)
    spec = json.loads((ROOT / "metrics/candidate-model.json").read_text())
    registry_version(client, spec)
    if spec["run_id"] != best.info.run_id:
        raise ValueError(f"Model does not match the best verified {stage} run")
    with (ROOT / "docs/comparison.csv").open() as stream:
        comparison = list(csv.DictReader(stream))
    if {row["run_id"] for row in comparison} != {row["run_id"] for row in summary["rows"]}:
        raise ValueError(f"Comparison CSV does not belong to the {stage} sweep")

    # Keep every control run immutable, as well as a convenient latest pointer.
    archive = ROOT / "metrics" / stage / summary["parent_id"]
    sources = [(ROOT / "metrics" / name, archive / name)
               for name in ("sweep.json", "candidate-model.json", "source-link-check.json")]
    sources.extend([(config_path, archive / "config.yaml"),
                    (ROOT / "docs/comparison.csv", archive / "comparison.csv")])
    for source, target in sources:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Refusing to overwrite a different archived {stage} file: {target}")
    archive.mkdir(parents=True, exist_ok=True)
    for source, target in sources:
        if not target.exists():
            shutil.copy2(source, target)
    shutil.copy2(ROOT / "docs/comparison.csv", ROOT / f"docs/{stage}-comparison.csv")
    save_json(ROOT / f"metrics/{stage}-sweep.json", summary)
    save_json(ROOT / f"metrics/{stage}-model.json", spec)
    reference_path = ROOT / "evaluation/comparison-models.json"
    references = json.loads(reference_path.read_text())
    references[role] = spec
    save_json(reference_path, references)
    print(json.dumps({"archived": str(archive.relative_to(ROOT)), "model": spec},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
