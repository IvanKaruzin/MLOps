"""Measure the same eight real HTTP requests at concurrency one and two."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def request(url, method="GET", payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read()), time.perf_counter() - started, None
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), time.perf_counter() - started, f"HTTP {exc.code}"
    except Exception as exc:
        return None, None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def benchmark(base_url, out):
    out.mkdir(parents=True, exist_ok=True)
    status, ready, _, error = request(base_url + "/ready")
    if status != 200 or error or not ready.get("ready"):
        raise RuntimeError(f"Service is not ready: {status}")
    model_id, runtime_sha = ready["model_id"], ready["runtime_sha256"]
    if model_id.startswith("fixture-"):
        raise ValueError("This benchmark must use the real baseline, not STUB")
    config = json.loads((ROOT / "evaluation/params.json").read_text())
    records = [json.loads(line) for line in (ROOT / config["golden"]).read_text().splitlines() if line.strip()][:8]
    limit = config["max_new_tokens"]
    def attempt(mode, record):
        status, response, latency, error = request(base_url + "/generate", "POST",
            dict(prompt=record["input"], max_new_tokens=limit))
        response = response or {}
        if status == 200 and (response.get("model_id") != model_id or response.get("runtime_sha256") != runtime_sha):
            error = "Loaded model/runtime changed during the benchmark"
        return dict(id=f"c{mode}-{record['id']}", input_id=record["id"], concurrency=mode,
            prompt=record["input"], max_new_tokens=limit, status=status, error=error,
            client_latency_seconds=latency, request_id=response.get("request_id"),
            model_id=response.get("model_id"), runtime_sha256=response.get("runtime_sha256"),
            output_tokens=response.get("output_tokens"), answer=response.get("answer"),
            queue_seconds=response.get("queue_seconds"), generation_seconds=response.get("generation_seconds"),
            server_total_seconds=response.get("total_seconds"))
    rows, batches = [], []
    raw = out / "requests.jsonl"
    with raw.open("w", encoding="utf-8") as stream:
        for mode in (1, 2):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=mode) as pool:
                results = list(pool.map(lambda record: attempt(mode, record), records))
            wall = time.perf_counter() - started
            for row in results:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            rows.extend(results)
            successes = [row for row in results if row["status"] == 200 and row["error"] is None]
            failures = [row for row in results if row not in successes]
            def mean(items, field):
                values = [row[field] for row in items if row[field] is not None]
                return statistics.mean(values) if values else None
            batches.append(dict(concurrency=mode, n=len(results), successes=len(successes), errors=len(failures),
                batch_wall_seconds=wall, successful_requests_per_second=len(successes) / wall,
                successful_mean_latency_seconds=mean(successes, "client_latency_seconds"),
                successful_max_latency_seconds=max((r["client_latency_seconds"] for r in successes), default=None),
                error_mean_latency_seconds=mean(failures, "client_latency_seconds"),
                error_max_latency_seconds=max((r["client_latency_seconds"] for r in failures), default=None),
                mean_queue_seconds=mean(successes, "queue_seconds"),
                mean_generation_seconds=mean(successes, "generation_seconds"),
                output_tokens=sum(r["output_tokens"] for r in successes)))
    bad_status, bad_response, bad_latency, _ = request(base_url + "/generate", "POST",
                                                     dict(prompt="A movie plot", max_new_tokens=0))
    old = {r["id"]: r for r in json.loads((ROOT / "metrics/baseline-outputs.json").read_text())}
    parity = [{"id": row["input_id"], "matches_saved_baseline_raw": row["answer"] == old[row["input_id"]]["raw"]}
              for row in rows if row["concurrency"] == 1 and row["status"] == 200]
    summary = dict(backend="transformers", real_model_loaded=True, model_id=model_id, runtime_sha256=runtime_sha,
        max_new_tokens=limit, warmup_completed_before_benchmark=True, batches=batches,
        same_input_ids=[r["id"] for r in records], raw_requests=str(raw.relative_to(ROOT)),
        invalid_request=dict(status=bad_status, body=bad_response, latency_seconds=bad_latency),
        saved_baseline_raw_parity=parity,
        limitations="Eight requests per mode; one generation lock; full-response latency, no p95/TTFT/ITL claim.")
    (out / "benchmark.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    if any(batch["errors"] for batch in batches) or not 400 <= (bad_status or 0) < 500:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/hw8910")
    args = parser.parse_args()
    benchmark(args.base_url.rstrip("/"), args.out.resolve())
