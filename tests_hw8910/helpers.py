"""Real subprocess and HTTP helpers shared by the seed and repaired contracts."""

import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("HW8910_CODE_ROOT", PROJECT)).resolve()


def environment():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(CODE_ROOT)}


def run_python(arguments, cwd, *, timeout=30):
    return subprocess.run(
        [sys.executable, "-B", *map(str, arguments)], cwd=cwd, env=environment(),
        text=True, capture_output=True, timeout=timeout,
    )


def lab(workspace, cwd, *arguments):
    return run_python([CODE_ROOT / "lab.py", "--workspace", workspace, *arguments], cwd)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def request(base_url, path, payload=None, *, timeout=5):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path, data=body,
        headers={} if body is None else {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as response:
        return response.code, json.loads(response.read())


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextlib.contextmanager
def server(workspace, cwd, *, warmup_seconds=0):
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-B", str(CODE_ROOT / "service.py"), "--workspace", str(workspace),
             "--port", str(port), "--warmup-seconds", str(warmup_seconds)],
            cwd=cwd, env=environment(), stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    output.seek(0)
                    raise AssertionError(f"HTTP server exited before /health: {output.read()}")
                try:
                    if request(base_url, "/health")[0] == 200:
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() >= deadline:
                    output.seek(0)
                    raise AssertionError(f"HTTP server never became live: {output.read()}")
                time.sleep(0.025)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def wait_ready(base_url):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status, ready = request(base_url, "/ready")
        debug_status, debug = request(base_url, "/debug")
        if status == 200 and ready.get("ready") is True and debug_status == 200 and debug.get("warmup_finished") is True:
            return debug
        time.sleep(0.025)
    raise AssertionError("HTTP service did not finish loading and warmup")
