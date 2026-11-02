"""Load the pinned real baseline twice and retain real HTTP restart evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from benchmark_hw8910 import request


ROOT = Path(__file__).resolve().parents[1]


def smoke(workspace, out):
    accepted = workspace / "accepted.json"
    before = hashlib.sha256(accepted.read_bytes()).hexdigest()
    manifest = json.loads(accepted.read_text())
    if manifest.get("synthetic") is not False:
        raise ValueError("Real smoke requires the explicitly pinned baseline")
    params = json.loads((ROOT / "evaluation/params.json").read_text())
    golden = json.loads((ROOT / params["golden"]).read_text().splitlines()[0])
    saved = json.loads((ROOT / "metrics/baseline-outputs.json").read_text())
    expected = next(row["raw"] for row in saved if row["id"] == golden["id"])
    runs = []
    with tempfile.TemporaryDirectory(prefix="hw8910-real-smoke-") as directory:
        for iteration in (1, 2):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            command = [sys.executable, "-B", str(ROOT / "service.py"), "--workspace", str(workspace),
                       "--port", str(port), "--warmup-seconds", "1", "--backend", "transformers"]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                               TRANSFORMERS_OFFLINE="1", MLFLOW_HTTP_REQUEST_MAX_RETRIES="1",
                               MLFLOW_HTTP_REQUEST_TIMEOUT="10")
            log_path = Path(directory) / f"server-{iteration}.log"
            with log_path.open("w") as log:
                server = subprocess.Popen(command, cwd=ROOT, env=environment,
                                          stdout=log, stderr=subprocess.STDOUT)
                started = time.perf_counter()
                observations = []
                try:
                    while True:
                        status, ready, _, _ = request(url + "/ready", timeout=1)
                        if status is not None and (not observations or observations[-1]["status"] != status):
                            observations.append(dict(status=status, body=ready))
                        if status == 200:
                            break
                        if ready and ready.get("error"):
                            raise RuntimeError("Real backend rejected the manifest or its dependencies")
                        if server.poll() is not None or time.perf_counter() - started > 180:
                            raise RuntimeError("Real model failed loading/warmup; inspect the service locally")
                        time.sleep(.25)
                    health_status, health, _, _ = request(url + "/health")
                    elapsed = time.perf_counter() - started
                    status, answer, latency, error = request(url + "/generate", "POST",
                        dict(prompt=golden["input"], max_new_tokens=params["max_new_tokens"]))
                    bad_status, bad_body, _, _ = request(url + "/generate", "POST",
                        dict(prompt=golden["input"], max_new_tokens=0))
                    debug_status, _, _, _ = request(url + "/debug")
                    assert health_status == 200 and health["backend"] == "transformers"
                    assert status == 200 and error is None and answer["answer"] == expected
                    assert answer["model_id"] == manifest["model_id"]
                    assert answer["runtime_sha256"] == manifest["runtime_sha256"]
                    assert 400 <= bad_status < 500 and debug_status == 404
                    runs.append(dict(iteration=iteration, command=command, process_id=server.pid,
                        ready_observations=observations, startup_seconds=elapsed,
                        health=dict(status=health_status, body=health), input_id=golden["id"],
                        generation=dict(status=status, client_latency_seconds=latency, body=answer),
                        invalid_request=dict(status=bad_status, body=bad_body), debug_status=debug_status,
                        matches_saved_baseline_raw=True))
                finally:
                    server.send_signal(signal.SIGINT)
                    try:
                        exit_code = server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                        raise RuntimeError("Owned real smoke service required forced shutdown")
                    if runs and runs[-1]["iteration"] == iteration:
                        runs[-1]["server_exit_code"] = exit_code
                        assert exit_code in (0, -signal.SIGINT)
    after = hashlib.sha256(accepted.read_bytes()).hexdigest()
    assert before == after and len(runs) == 2
    summary = dict(passed=True, backend="transformers", real_model_loaded=True,
        acceptance_basis=manifest["acceptance_basis"], accepted_before_sha256=before,
        accepted_after_sha256=after, runtime_sha256=manifest["runtime_sha256"],
        model_id=manifest["model_id"], source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("service.py", "hw8910/runtime.py")},
        restart_verified=True, both_owned_services_stopped=True, runs=runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, restarts=len(runs), model_id=manifest["model_id"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "data/hw8910")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/hw8910/smoke.json")
    args = parser.parse_args()
    smoke(args.workspace.resolve(), args.out.resolve())
