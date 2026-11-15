"""HTTP behavior of the opt-in simulator, independent of Kubernetes evidence."""

from dataclasses import replace
import json
import time
import unittest

from fastapi.testclient import TestClient

from hw11.app import create_app
from hw11.backend import Settings


def application(**overrides):
    values = dict(warmup_seconds=0, hit_iterations=10, miss_iterations=20, chunk_iterations=5)
    values.update(overrides)
    return create_app(replace(Settings(), **values))


def wait_ready(client):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if client.get("/ready").status_code == 200:
            return
        time.sleep(.005)
    raise AssertionError("Simulator did not become ready")


def events(response):
    return [json.loads(line) for line in response.text.splitlines()]


class ServiceTests(unittest.TestCase):
    def test_probes_have_separate_meanings_and_warmup_gates_traffic(self):
        with TestClient(application(warmup_seconds=.15)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/startup").status_code, 503)
            self.assertEqual(client.get("/ready").status_code, 503)
            response = client.post("/stream", json={"prompt": "early"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"], "backend_not_ready")
            self.assertEqual(response.headers["x-backend-mode"], "simulated")
            wait_ready(client)
            self.assertEqual(client.get("/startup").status_code, 200)

    def test_faulty_probe_mode_reproduces_early_request_failure(self):
        with TestClient(application(warmup_seconds=.15, broken_probes=True)) as client:
            self.assertEqual(client.get("/startup").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)
            self.assertEqual(client.post("/stream", json={"prompt": "cold"}).status_code, 503)
            self.assertFalse(client.get("/debug").json()["warmup_finished"])

    def test_stream_identity_final_marker_and_success_histograms(self):
        with TestClient(application()) as client:
            wait_ready(client)
            response = client.post("/stream", json=dict(prompt="hello", request_id="req-1", session_id="s-1",
                                                       max_chunks=2, chunk_delay_seconds=0))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-request-id"], "req-1")
            output = events(response)
            self.assertEqual([event["event"] for event in output], ["chunk", "chunk", "done"])
            for event in output:
                self.assertEqual(event["type"], event["event"])
                self.assertEqual(event["model_version"], "A")
                self.assertEqual(event["backend_mode"], "simulated")
                self.assertEqual(event["session_id"], "s-1")
                self.assertEqual(event["artifact_sha256"], response.headers["x-artifact-sha256"])
            self.assertEqual(output[-1]["chunks"], 2)
            self.assertGreaterEqual(output[-1]["ttfc_seconds"], output[-1]["queue_seconds"])
            metrics = client.get("/metrics").text
            self.assertIn('hw11_requests_total{outcome="completed"} 1.0', metrics)
            self.assertIn('hw11_ttfc_seconds_count 1.0', metrics)
            self.assertNotIn('request_id=', metrics)
            self.assertNotIn('session_id=', metrics)

    def test_generate_preserves_previous_response_fields(self):
        with TestClient(application()) as client:
            wait_ready(client)
            response = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 32})
            self.assertEqual(response.status_code, 200)
            output = response.json()
            for name in ("answer", "output_tokens", "request_id", "model_id", "runtime_sha256",
                         "queue_seconds", "generation_seconds", "total_seconds"):
                self.assertIn(name, output)
            self.assertAlmostEqual(output["total_seconds"], output["generation_seconds"] + output["queue_seconds"])
            self.assertEqual(client.post("/generate", json={"prompt": "hello", "max_new_tokens": 31}).status_code, 422)

    def test_drain_removes_readiness_but_serves_propagation_window(self):
        with TestClient(application()) as client:
            wait_ready(client)
            self.assertEqual(client.post("/drain").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.post("/stream", json={"prompt": "still routed", "max_chunks": 1, "chunk_delay_seconds": 0})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(events(response)[-1]["event"], "done")

    def test_two_dialogues_are_isolated_and_history_can_move(self):
        with TestClient(application()) as client:
            wait_ready(client)
            for session, own, other in (("one", "ALPHA-CONTROL", "BETA-CONTROL"),
                                        ("two", "BETA-CONTROL", "ALPHA-CONTROL")):
                payload = dict(session_id=session, prompt="continue", max_chunks=1, chunk_delay_seconds=0,
                               messages=[dict(role="user", content=own)])
                text = client.post("/stream", json=payload).text
                self.assertIn(own, text)
                self.assertNotIn(other, text)
            self.assertEqual(client.post("/reset-cache").json()["cache_size"], 0)

    def test_invalid_input_and_large_body_have_identity_and_no_compute(self):
        with TestClient(application()) as client:
            wait_ready(client)
            for payload in ({"prompt": ""}, {"prompt": "x", "max_chunks": True},
                            {"prompt": "x", "chunk_delay_seconds": float("inf")},
                            {"prompt": "x", "max_chunks": 32, "chunk_delay_seconds": 2},
                            {"prompt": "x", "request_id": "bad\r\nheader"}):
                if payload.get("chunk_delay_seconds") == float("inf"):
                    # JSON itself has no Infinity literal; test the encoded parser boundary.
                    response = client.post("/stream", content='{"prompt":"x","chunk_delay_seconds":1e999}',
                                           headers={"content-type": "application/json"})
                else:
                    response = client.post("/stream", json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertIn("x-model-version", response.headers)
            response = client.post("/stream", content=b"x" * 65537)
            self.assertEqual(response.status_code, 413)
            self.assertIn("x-request-id", response.headers)
            self.assertEqual(client.get("/debug").json()["running"], 0)

    def test_idle_cache_reset_and_verified_cache_hit(self):
        with TestClient(application()) as client:
            wait_ready(client)
            payload = dict(prompt="same prefix", max_chunks=1, chunk_delay_seconds=0)
            self.assertFalse(events(client.post("/stream", json=payload))[-1]["cache_hit"])
            self.assertTrue(events(client.post("/stream", json=payload))[-1]["cache_hit"])
            self.assertEqual(client.post("/reset-cache").json()["cache_size"], 0)
            self.assertFalse(events(client.post("/stream", json=payload))[-1]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
