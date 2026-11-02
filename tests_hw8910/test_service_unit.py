"""ASGI state/input/queue tests; disk integrity is covered by release tests."""

import concurrent.futures
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import release
from service import MAX_BODY_BYTES, StubBackend, create_app


def write_manifest(root, version="A", limit=32):
    runtime = {"base_revision": "fixture-only", "tokenizer_sha256": "a" * 64,
               "template_sha256": "b" * 64,
               "generation": {"max_new_tokens": limit, "do_sample": False}, "dtype": "float32"}
    digest = hashlib.sha256(json.dumps(runtime, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    manifest = {"schema_version": 1, "model_id": f"fixture-{version}",
                "artifact_uri": f"model-{version}.txt", "model_sha256": "c" * 64,
                "protocol_sha256": "d" * 64, "golden_sha256": "e" * 64,
                "runtime": runtime, "runtime_sha256": digest, "synthetic": True}
    (root / "accepted.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def wait_until_ready(client):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get("/ready")
        if response.status_code == 200:
            return response.json()
        if response.json().get("error"):
            raise AssertionError(response.json())
        time.sleep(0.005)
    raise AssertionError("service did not become ready")


def wait_until_failed(client):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get("/ready")
        if response.json().get("error"):
            return response
        time.sleep(0.005)
    raise AssertionError("expected loader failure was not reported")


class ServiceContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = write_manifest(self.root)
        # These unit tests isolate the server from the separately tested disk
        # verifier. Production and network integration use the actual verifier.
        self.verifier = Mock(return_value={"fixture": "c" * 64})
        self.patch = patch.object(release, "verify_artifact", self.verifier, create=True)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_load_occurs_only_in_lifespan_and_readiness_waits_for_warmup(self):
        entered, finish = threading.Event(), threading.Event()
        loaded, closed = [], []

        class BlockingBackend(StubBackend):
            def warmup(self):
                entered.set()
                if not finish.wait(3):
                    raise RuntimeError("test warmup was not released")
                self.warmup_calls += 1

            def close(self):
                closed.append(self.loaded_id)

        def factory(manifest, workspace):
            loaded.append(manifest["model_id"])
            return BlockingBackend(manifest, workspace)

        app = create_app(self.root, 0, factory)
        self.assertEqual(loaded, [])
        self.verifier.assert_not_called()
        with TestClient(app) as client:
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 503)
                self.assertEqual(client.post("/generate", json={"prompt": "hello"}).status_code, 503)
                self.assertEqual(client.get("/debug").json()["calls"], 0)
            finally:
                finish.set()
            ready = wait_until_ready(client)
            self.assertEqual(ready["model_id"], "fixture-A")
            self.assertEqual(client.get("/debug").json()["warmup_calls"], 1)
            self.assertEqual(loaded, ["fixture-A"])
            self.verifier.assert_called_once()
        self.assertEqual(closed, ["fixture-A"])

    def test_invalid_requests_are_rejected_before_any_generation_call(self):
        with TestClient(create_app(self.root, 0)) as client:
            wait_until_ready(client)
            before = client.get("/debug").json()["calls"]
            requests = [{}, {"prompt": ""}, {"prompt": " \n\t"}, {"prompt": 12},
                        {"prompt": True}, {"prompt": "x" * 4097}, ["not an object"],
                        {"prompt": "valid", "temperature": 0.2}]
            requests.extend({"prompt": "valid", "max_new_tokens": limit}
                            for limit in (0, -1, 100000, "32", True, 32.0, None, 1, 31, 33))
            for request in requests:
                with self.subTest(request=str(request)[:100]):
                    response = client.post("/generate", json=request)
                    self.assertGreaterEqual(response.status_code, 400)
                    self.assertLess(response.status_code, 500)
            self.assertEqual(client.get("/debug").json()["calls"], before)
            self.assertEqual(client.post("/generate", json={"prompt": "valid"}).status_code, 200)
            self.assertEqual(client.post("/generate", json={"prompt": "valid", "max_new_tokens": 32}).status_code, 200)
            self.assertEqual(client.get("/debug").json()["calls"], before + 2)

    def test_loaded_identity_is_stable_after_pointer_change_and_removal(self):
        with TestClient(create_app(self.root, 0)) as client:
            wait_until_ready(client)
            initial = client.post("/generate", json={"prompt": "first"}).json()
            changed = write_manifest(self.root, "B", 96)
            for remove in (False, True):
                if remove:
                    (self.root / "accepted.json").unlink()
                response = client.post("/generate", json={"prompt": "after publication"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["model_id"], initial["model_id"])
                self.assertEqual(response.json()["runtime_sha256"], initial["runtime_sha256"])
                self.assertIn("fixture-A", response.json()["answer"])
                self.assertNotEqual(response.json()["runtime_sha256"], changed["runtime_sha256"])
        write_manifest(self.root, "B", 96)
        with TestClient(create_app(self.root, 0)) as client:
            self.assertEqual(wait_until_ready(client)["model_id"], "fixture-B")
            response = client.post("/generate", json={"prompt": "after restart"})
            self.assertIn("fixture-B", response.json()["answer"])

    def test_integrity_failure_prevents_backend_load_and_keeps_not_ready(self):
        self.verifier.side_effect = ValueError("secret filesystem location")
        factory = Mock(side_effect=AssertionError("backend must not load"))
        with TestClient(create_app(self.root, 0, factory)) as client:
            response = wait_until_failed(client)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"], "ValueError")
            self.assertNotIn("secret", response.text)
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.post("/generate", json={"prompt": "valid"}).status_code, 503)
        factory.assert_not_called()

    def test_warmup_failure_never_allows_generation(self):
        class BrokenWarmup(StubBackend):
            def warmup(self):
                raise RuntimeError("sensitive backend detail")

        with TestClient(create_app(self.root, 0, BrokenWarmup)) as client:
            self.assertEqual(wait_until_failed(client).json()["error"], "RuntimeError")
            self.assertEqual(client.post("/generate", json={"prompt": "valid"}).status_code, 503)
            debug = client.get("/debug").json()
            self.assertFalse(debug["warmup_finished"])
            self.assertEqual(debug["calls"], 0)

    def test_concurrent_requests_use_one_compute_slot_and_measure_queue(self):
        counts = {"active": 0, "maximum": 0}
        guard = threading.Lock()

        class SlowBackend(StubBackend):
            def generate(self, prompt, limit):
                with guard:
                    counts["active"] += 1
                    counts["maximum"] = max(counts["maximum"], counts["active"])
                try:
                    time.sleep(0.06)
                    return super().generate(prompt, limit)
                finally:
                    with guard:
                        counts["active"] -= 1

        barrier = threading.Barrier(2)
        with TestClient(create_app(self.root, 0, SlowBackend)) as client:
            wait_until_ready(client)

            def request(index):
                barrier.wait(timeout=3)
                response = client.post("/generate", json={"prompt": f"request {index}"})
                self.assertEqual(response.status_code, 200)
                return response.json()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(request, range(2)))
            self.assertEqual(counts["maximum"], 1)
            self.assertEqual(client.get("/debug").json()["calls"], 2)
            self.assertEqual(len({r["request_id"] for r in responses}), 2)
            self.assertGreater(max(r["queue_seconds"] for r in responses), 0.03)
            for response in responses:
                self.assertGreater(response["generation_seconds"], 0.05)
                self.assertAlmostEqual(response["total_seconds"],
                                       response["generation_seconds"] + response["queue_seconds"], places=6)

    def test_fixed_runtime_limit_is_96_for_real_mode_and_debug_is_absent(self):
        write_manifest(self.root, limit=96)
        # Explicit backend injection avoids importing Transformers in this test.
        with TestClient(create_app(self.root, 0, StubBackend, backend_kind="transformers")) as client:
            wait_until_ready(client)
            self.assertEqual(client.get("/debug").status_code, 404)
            self.assertEqual(client.post("/generate", json={"prompt": "valid"}).status_code, 200)
            self.assertEqual(client.post("/generate", json={"prompt": "valid", "max_new_tokens": 32}).status_code, 422)
            self.assertEqual(client.post("/generate", json={"prompt": "valid", "max_new_tokens": 96}).status_code, 200)

    def test_streamed_body_limit_is_checked_without_a_content_length(self):
        with TestClient(create_app(self.root, 0)) as client:
            wait_until_ready(client)
            before = client.get("/debug").json()["calls"]
            chunks = (b"x" * 8192 for _ in range(MAX_BODY_BYTES // 8192 + 1))
            response = client.post("/generate", content=chunks, headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(client.get("/debug").json()["calls"], before)


if __name__ == "__main__":
    unittest.main()
