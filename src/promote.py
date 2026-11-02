"""Выбор из одного законченного sweep; порог проверяется ДО записи в реестр."""
import argparse
import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path
from collections import Counter

import mlflow
from mlflow import MlflowClient

from src.common import connect, load_params, provenance, save_json


def validate_sampling(client, params, runs):
    """Reject mixed pools, incomplete consumption and altered server manifests."""
    from src.experiment_train import expected_sampling_manifest
    expected = expected_sampling_manifest(params)
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    expected_sha = canonical(expected)
    processed = expected["processed_train"]
    pids = []
    source_links = {run.data.tags.get("source_link_sha256") for run in runs}
    if len(source_links) != 1 or not all(isinstance(value, str) and len(value) == 64 for value in source_links):
        raise ValueError("Кандидаты имеют разную или отсутствующую проверку source/tensor")
    with tempfile.TemporaryDirectory(prefix="hw67-select-sampling-") as directory:
        for index, run in enumerate(runs):
            if run.data.tags.get("sampling_sha256") != expected_sha:
                raise ValueError("Кандидаты имеют другую выборку или порядок обработки")
            required = {"processed_examples": processed["count"], "processed_unique_examples": processed["unique_count"],
                        "train_pool_rows": expected["train"]["selected_rows"], "val_rows": expected["val"]["selected_rows"]}
            if any(run.data.metrics.get(key) != value for key, value in required.items()):
                raise ValueError("Фактический бюджет или размеры выборок не соответствуют конфигу")
            for name in ("train", "val"):
                if run.data.tags.get(f"source_{name}_sha256") != expected[name]["source_sha256"]:
                    raise ValueError("Исходные строки sampling не соответствуют версии run")
            destination = Path(directory) / str(index)
            destination.mkdir()
            manifest = json.loads(Path(client.download_artifacts(run.info.run_id, "sampling.json", str(destination))).read_text())
            if manifest != expected or canonical(manifest) != expected_sha:
                raise ValueError("Серверный sampling manifest не соответствует данным и бюджету")
            pid = run.data.tags.get("worker_pid", "")
            if not isinstance(pid, str) or not pid.isdigit() or int(pid) <= 0:
                raise ValueError("Нет идентификатора изолированного процесса кандидата")
            pids.append(pid)
    if len(set(pids)) != len(runs):
        raise ValueError("Кандидаты выполнялись без независимых процессов")
    return expected


def validate_training_view(client, params, runs, sampling):
    """Recompute the actual representation, including labels and target-token counts."""
    from src.experiment_train import prepare_training_data
    from src.training_view import fingerprint
    train, val, _, _, expected = prepare_training_data(params)
    expected_sha = fingerprint(expected)
    by_id = {row["id"]: row for row in train}
    consumed = Counter(sampling["processed_train"]["ids"])
    tokens = sum(sum(by_id[identity]["attention_mask"]) * count for identity, count in consumed.items())
    supervised = sum(sum(label != -100 for label in by_id[identity]["labels"][1:]) * count
                     for identity, count in consumed.items())
    with tempfile.TemporaryDirectory(prefix="hw67-select-view-") as directory:
        for index, run in enumerate(runs):
            if run.data.tags.get("training_view_sha256") != expected_sha:
                raise ValueError("Training view отсутствует или не соответствует текущему формату и данным")
            if (run.data.metrics.get("processed_tokens") != tokens
                    or run.data.metrics.get("processed_supervised_tokens") != supervised):
                raise ValueError("Фактические токены не соответствуют использованному training view")
            destination = Path(directory) / str(index)
            destination.mkdir()
            proof = json.loads(Path(client.download_artifacts(run.info.run_id, "training-view.json", str(destination))).read_text())
            if proof != expected or fingerprint(proof) != expected_sha:
                raise ValueError("Сохранённый training view proof изменён или устарел")
            compressed = Path(client.download_artifacts(run.info.run_id, "training-view.json.gz", str(destination)))
            arrays = json.loads(gzip.decompress(compressed.read_bytes()))
            if arrays != {"train": train, "val": val}:
                raise ValueError("Сохранённые input ids/labels отличаются от actual training view")


def validate_checkpoints(client, params, runs, sampling=None):
    """Bind the selected validation checkpoint to the adapter actually served."""
    import yaml
    if sampling is None:
        from src.experiment_train import expected_sampling_manifest
        sampling = expected_sampling_manifest(params)
    val_ids = sampling["val"].get("original_ids", sampling["val"]["ids"])
    from src.training_view import fingerprint
    with tempfile.TemporaryDirectory(prefix="hw67-select-checkpoint-") as directory:
        for index, run in enumerate(runs):
            destination = Path(directory) / str(index)
            destination.mkdir()
            def download(path):
                return Path(client.download_artifacts(run.info.run_id, path, str(destination)))
            proof = json.loads(download("checkpoint.json").read_text())
            steps, selected = proof.get("training_steps"), proof.get("selected_step")
            if (type(steps) is not int or steps != params["train"]["max_steps"] or type(selected) is not int
                    or not 1 <= selected <= steps or proof.get("primary_val_rows") != len(val_ids)
                    or proof.get("primary_val_ids_sha256") != fingerprint(val_ids)
                    or proof.get("auxiliary_val_rows") != sampling["val"]["selected_rows"] - len(val_ids)
                    or proof.get("policy") != params["train"].get("checkpoint_selection", "best_val")
                    or proof.get("metric") != "original_val_token_weighted_nll"):
                raise ValueError("Checkpoint selection does not match the original VAL and full training budget")
            trajectory = proof.get("checkpoints", [])
            expected_steps = [0, *range(params["train"]["eval_every"], steps + 1, params["train"]["eval_every"])]
            if expected_steps[-1] != steps:
                expected_steps.append(steps)
            if ([point.get("step") for point in trajectory] != expected_steps
                    or any(not math.isfinite(point.get("val_loss", float("nan"))) or point["val_loss"] < 0
                           for point in trajectory)):
                raise ValueError("Checkpoint trajectory is incomplete or nonfinite")
            chosen = next((point for point in trajectory if point["step"] == selected), None)
            expected = (min(trajectory[1:], key=lambda point: (point["val_loss"], point["step"]))
                        if proof["policy"] == "best_val" else trajectory[-1])
            final_loss = run.data.metrics.get("final_val_loss", float("nan"))
            if (chosen != expected or run.data.metrics.get("selected_step") != selected
                    or not math.isclose(proof.get("selected_val_loss", float("nan")), final_loss, rel_tol=1e-6, abs_tol=1e-6)
                    or not math.isclose(chosen["val_loss"], final_loss, rel_tol=1e-5, abs_tol=1e-6)
                    or proof.get("end_val_loss") != trajectory[-1]["val_loss"]
                    or run.data.metrics.get("end_val_loss") != proof["end_val_loss"]):
                raise ValueError("Selected checkpoint metrics differ from the exported model selection")
            metadata = yaml.safe_load(download("model/MLmodel").read_text())
            location = metadata["flavors"]["python_function"]["artifacts"]["adapter"]["path"]
            manifest = json.loads(download(f"model/{location}/manifest.json").read_text())
            weights = download(f"model/{location}/adapter_model.safetensors")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if (manifest.get("checkpoint") != proof or proof.get("adapter_sha256") != digest
                    or manifest.get("sampling_sha256") != run.data.tags.get("sampling_sha256")
                    or manifest.get("training_view_sha256") != run.data.tags.get("training_view_sha256")
                    or run.data.tags.get("selected_adapter_sha256") != digest):
                raise ValueError("Selected checkpoint does not match the packaged adapter weights")


def gate(run, max_ratio):
    m = run.data.metrics
    values = [m.get(k, float("nan")) for k in ("final_val_loss", "base_val_loss")]
    if not math.isfinite(max_ratio) or max_ratio < 0 or not all(math.isfinite(v) for v in values) or values[0] < 0 or values[1] <= 0:
        raise ValueError("Некорректная метрика или порог")
    if run.info.status != "FINISHED" or values[0] / values[1] > max_ratio:
        raise ValueError(f"Порог не пройден: val/base={values[0] / values[1]:.4f}, максимум {max_ratio}")
    return values[0] / values[1]


def select(client, params, summary):
    current = provenance(params)
    parent = client.get_run(summary["parent_id"])
    if summary.get("kind", "sweep") != "sweep" or parent.data.tags.get("kind", "sweep") != "sweep":
        raise ValueError("Development runs cannot be selected as a final fair sweep")
    if parent.info.status != "FINISHED":
        raise ValueError("Sweep не завершён")
    version_keys = ("code_sha256", "config_sha256", "data_train_sha256", "data_val_sha256")
    if any(parent.data.tags.get(key) != current[key] or summary.get("provenance", {}).get(key) != current[key]
           for key in version_keys):
        raise ValueError("Версии sweep не соответствуют текущим коду, конфигу или данным")
    runs = client.search_runs([parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}' and tags.kind = 'candidate'",
        max_results=1000)
    names = {c["name"] for c in params["sweep"]}
    if len(runs) != len(names) or {r.data.tags.get("candidate") for r in runs} != names:
        raise ValueError("Число отдельных runs не совпадает с конфигурациями")
    if ({row.get("run_id") for row in summary.get("rows", [])} != {run.info.run_id for run in runs}
            or len(summary.get("rows", [])) != len(runs)):
        raise ValueError("Локальная сводка не соответствует серверным runs")
    candidates = {candidate["name"]: candidate for candidate in params["sweep"]}
    for run in runs:
        for key in version_keys:
            if run.data.tags.get(key) != current[key]:
                raise ValueError(f"Run {run.info.run_id}: отсутствует или устарел {key}; повторите sweep")
        if run.info.status != "FINISHED" or not math.isfinite(run.data.metrics.get("final_val_loss", float("nan"))):
            raise ValueError("Есть незавершённый кандидат или отсутствует val loss")
        candidate = candidates[run.data.tags["candidate"]]
        if (float(run.data.params.get("r", "nan")) != candidate["r"]
                or float(run.data.params.get("lr", "nan")) != candidate["lr"]
                or run.data.params.get("target_modules") != ",".join(candidate["target_modules"])):
            raise ValueError("Сохранённые параметры не соответствуют конфигурации кандидата")
        if run.data.metrics.get("steps") != params["train"]["max_steps"]:
            raise ValueError("Кандидаты имеют разный бюджет обучения")
        base = run.data.metrics.get("base_val_loss", float("nan"))
        if not math.isfinite(base) or base <= 0 or run.data.metrics["final_val_loss"] < 0:
            raise ValueError("Некорректный loss кандидата")
    revisions = {r.data.tags.get("base_revision") for r in runs}
    if len(revisions) != 1 or not all(isinstance(revision, str) and len(revision) == 40
                                    and all(char in "0123456789abcdef" for char in revision) for revision in revisions):
        raise ValueError("Кандидаты обучены на разных ревизиях базы")
    losses = [run.data.metrics["base_val_loss"] for run in runs]
    if any(not math.isclose(loss, losses[0], rel_tol=1e-6, abs_tol=1e-6) for loss in losses):
        raise ValueError("Исходный val loss кандидатов различается")
    sampling = validate_sampling(client, params, runs)
    validate_training_view(client, params, runs, sampling)
    validate_checkpoints(client, params, runs, sampling)
    return min(runs, key=lambda r: (r.data.metrics["final_val_loss"], r.info.run_id))


def promote(client, run, name, alias, threshold):
    ratio = gate(run, threshold)
    if run.data.tags.get("kind", "candidate") != "candidate":
        raise ValueError("Development models cannot be registered through the final candidate path")
    if alias == "champion":
        raise ValueError("Champion назначается только через make promote после проверки качества ответов")
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
    client.set_model_version_tag(name, version.version, "val_ratio", str(ratio))
    client.set_model_version_tag(name, version.version, "gate.max_val_ratio", str(threshold))
    client.set_model_version_tag(name, version.version, "validation", "passed")
    client.set_registered_model_alias(name, alias, version.version)
    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-val-ratio", type=float)
    args = ap.parse_args()
    p = load_params()
    connect(p)
    client = MlflowClient()
    summary = json.loads(Path("metrics/sweep.json").read_text())
    best = select(client, p, summary)
    threshold = p["registry"]["max_val_ratio"] if args.max_val_ratio is None else args.max_val_ratio
    version = promote(client, best, p["registry"]["name"], "candidate", threshold)
    result = {"name": version.name, "version": version.version, "run_id": best.info.run_id,
        "uri": f"models:/{version.name}@candidate", "val_loss": best.data.metrics["final_val_loss"]}
    save_json("metrics/promoted.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
