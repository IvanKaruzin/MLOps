"""Observable simulator admission, integrity and physical worker lifetime."""

import asyncio
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest

from hw11.backend import AdmissionError, Backend, Settings


def small_settings(**overrides):
    return replace(Settings(), warmup_seconds=0, hit_iterations=10, miss_iterations=20,
                   chunk_iterations=5, **overrides)


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def make_backend(self, **settings):
        backend = Backend(small_settings(**settings))
        self.addAsyncCleanup(backend.close)
        await backend.load()
        return backend

    async def test_artifact_identity_is_verified_and_loaded(self):
        backend = await self.make_backend()
        self.assertTrue(backend.ready)
        self.assertEqual(backend.loaded_version, "A")
        self.assertEqual(backend.loaded_sha256, backend.settings.artifact_sha256)

    async def test_corrupt_artifact_never_warms_or_admits(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.json"
            artifact.write_text('{"model_version":"B"}')
            backend = await self.make_backend(artifact_path=str(artifact))
            self.assertFalse(backend.ready)
            self.assertEqual(backend.error, "ValueError")
            with self.assertRaises(AdmissionError) as failure:
                await backend.acquire()
            self.assertEqual(failure.exception.status, 503)

    async def test_waiting_queue_is_bounded_and_measured(self):
        backend = await self.make_backend(max_waiting=1)
        first = await backend.acquire()
        waiter = asyncio.create_task(backend.acquire())
        await asyncio.sleep(.01)
        self.assertEqual(backend.waiting, 1)
        with self.assertRaises(AdmissionError) as failure:
            await backend.acquire()
        self.assertEqual(failure.exception.reason, "queue_full")
        first.close()
        second = await waiter
        self.assertGreater(second.queue_seconds, .005)
        self.assertEqual(backend.running, 1)
        self.assertEqual(backend.waiting, 0)
        second.close()
        self.assertEqual(backend.running, 0)

    async def test_timeout_removes_waiter_without_releasing_active_slot(self):
        backend = await self.make_backend(queue_timeout_seconds=.01)
        first = await backend.acquire()
        with self.assertRaises(AdmissionError) as failure:
            await backend.acquire()
        self.assertEqual(failure.exception.reason, "queue_timeout")
        self.assertEqual((backend.running, backend.waiting), (1, 0))
        first.close()

    async def test_cancelled_waiter_does_not_consume_capacity(self):
        backend = await self.make_backend()
        first = await backend.acquire()
        waiter = asyncio.create_task(backend.acquire())
        await asyncio.sleep(.005)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(backend.waiting, 0)
        self.assertEqual(backend.running, 1)
        first.close()
        following = await backend.acquire()
        following.close()

    async def test_cancelled_compute_holds_slot_until_thread_really_finishes(self):
        backend = await self.make_backend(max_waiting=0)
        entered, release = threading.Event(), threading.Event()

        def cpu_work(iterations):
            entered.set()
            release.wait(2)

        backend.cpu_work = cpu_work
        lease = await backend.acquire()
        computing = asyncio.create_task(lease.compute(1))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            computing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await computing
            lease.close()
            self.assertEqual(backend.running, 1)
            with self.assertRaises(AdmissionError):
                await backend.acquire()
        finally:
            release.set()
        for _ in range(100):
            if backend.running == 0:
                break
            await asyncio.sleep(.005)
        self.assertEqual(backend.running, 0)
        lease.close()
        self.assertEqual(backend.running, 0)

    async def test_cache_has_hit_miss_and_lru_capacity(self):
        backend = await self.make_backend(cache_capacity=1)
        lease = await backend.acquire()
        try:
            self.assertFalse(await backend.prepare(lease, "alpha", []))
            self.assertTrue(await backend.prepare(lease, "alpha", []))
            self.assertFalse(await backend.prepare(lease, "beta", []))
            self.assertEqual(len(backend.cache), 1)
            self.assertFalse(await backend.prepare(lease, "alpha", []))
        finally:
            lease.close()

    async def test_never_ready_is_live_but_cannot_admit(self):
        backend = await self.make_backend(never_ready=True)
        self.assertEqual(backend.loaded_version, "A")
        self.assertFalse(backend.ready)
        with self.assertRaises(AdmissionError):
            await backend.acquire()

    async def test_settings_reject_unbounded_or_nonfinite_work(self):
        for settings in ({"warmup_seconds": float("inf")}, {"max_running": 0},
                         {"max_waiting": -1}, {"miss_iterations": 0}, {"cache_capacity": 100000}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(Settings(), **settings)


class StreamingDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_failure_accounts_interruption_and_releases_compute_slot(self):
        import json
        from hw11.app import create_app
        from starlette.requests import ClientDisconnect

        app = create_app(small_settings())
        async with app.router.lifespan_context(app):
            while not app.state.backend.warmup_finished:
                await asyncio.sleep(.001)
            body = json.dumps(dict(prompt="disconnect", max_chunks=3, chunk_delay_seconds=0)).encode()
            delivered = False

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    raise OSError("Client connection was lost")

            scope = dict(type="http", asgi={"version": "3.0", "spec_version": "2.4"},
                         http_version="1.1", method="POST", scheme="http", path="/stream",
                         raw_path=b"/stream", query_string=b"", root_path="",
                         headers=[(b"content-type", b"application/json")],
                         client=("127.0.0.1", 1234), server=("localhost", 8000))
            with self.assertRaises(ClientDisconnect):
                await app(scope, receive, send)
            backend = app.state.backend
            self.assertEqual(backend.running, 0)
            text = backend.metrics.render().decode()
            self.assertIn('hw11_requests_total{outcome="interrupted"} 1.0', text)
            self.assertIn('hw11_requests_total{outcome="completed"} 0.0', text)
            self.assertIn('hw11_ttfc_seconds_count 0.0', text)


    async def test_header_send_failure_before_generator_start_releases_lease(self):
        import json
        from hw11.app import create_app
        from starlette.requests import ClientDisconnect

        app = create_app(small_settings())
        async with app.router.lifespan_context(app):
            while not app.state.backend.warmup_finished:
                await asyncio.sleep(.001)
            body = json.dumps(dict(prompt="disconnect before first chunk")).encode()
            delivered = False

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                if message["type"] == "http.response.start":
                    raise OSError("Client disconnected while waiting for admission")

            scope = dict(type="http", asgi={"version": "3.0", "spec_version": "2.4"},
                         http_version="1.1", method="POST", scheme="http", path="/stream",
                         raw_path=b"/stream", query_string=b"", root_path="",
                         headers=[(b"content-type", b"application/json")],
                         client=("127.0.0.1", 1234), server=("localhost", 8000))
            with self.assertRaises(ClientDisconnect):
                await app(scope, receive, send)
            backend = app.state.backend
            self.assertEqual((backend.running, backend.waiting), (0, 0))
            text = backend.metrics.render().decode()
            self.assertIn('hw11_requests_total{outcome="interrupted"} 1.0', text)
            self.assertIn('hw11_requests_total{outcome="completed"} 0.0', text)
            self.assertIn('hw11_ttfc_seconds_count 0.0', text)
            # The leaked slot used to reject all subsequent admissions permanently.
            lease = await asyncio.wait_for(backend.acquire(), timeout=.1)
            lease.close()


if __name__ == "__main__":
    unittest.main()
