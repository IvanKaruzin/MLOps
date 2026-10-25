"""Таблица из сервера, а не вручную переписанные числа."""
import csv
import argparse
import io
import json
from pathlib import Path

from mlflow import MlflowClient
from src.common import connect, load_params
from src.promote import select


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="metrics/sweep.json")
    parser.add_argument("--out", default="docs/comparison.csv")
    args = parser.parse_args(argv)
    p = load_params()
    connect(p)
    summary = json.loads(Path(args.summary).read_text())
    client = MlflowClient()
    select(client, p, summary)
    rows = []
    for item in summary["rows"]:
        run = client.get_run(item["run_id"])
        rows.append({"name": item["name"], "run_id": run.info.run_id,
            **{k: run.data.params[k] for k in ("r", "lr", "target_modules")},
            **{k: run.data.metrics[k] for k in ("base_val_loss", "final_val_loss", "seconds", "peak_memory_mb", "adapter_mb",
                                               "processed_examples", "processed_unique_examples", "train_pool_rows", "val_rows")},
            "sampling_sha256": run.data.tags["sampling_sha256"],
            "training_view_sha256": run.data.tags["training_view_sha256"]})
    rows.sort(key=lambda row: row["final_val_loss"])
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = Path(args.out)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())


if __name__ == "__main__":
    main()
