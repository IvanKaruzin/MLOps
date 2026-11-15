"""Opt-in streaming simulator; existing homework service modes remain untouched."""

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

from hw11.backend import AdmissionError, Backend, Settings
from service import BodySizeLimit, GenerateRequest


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: StrictStr = Field(pattern="^(system|user|assistant)$")
    content: StrictStr = Field(min_length=1, max_length=4096)


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: StrictStr | None = Field(default=None, pattern="^[A-Za-z0-9._:-]{1,128}$")
    session_id: StrictStr = Field(default="anonymous", pattern="^[A-Za-z0-9._:-]{1,128}$")
    prompt: StrictStr = Field(min_length=1, max_length=4096)
    messages: list[Message] = Field(default_factory=list, max_length=32)
    max_chunks: StrictInt = Field(default=4, ge=1, le=32)
    chunk_delay_seconds: float = Field(default=.05, ge=0, le=2, allow_inf_nan=False)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Blank prompt")
        return value

    @model_validator(mode="after")
    def bounded_work(self):
        if self.max_chunks * self.chunk_delay_seconds > 30:
            raise ValueError("Maximum configured chunk delay is 30 seconds")
        if sum(len(message.content) for message in self.messages) > 16384:
            raise ValueError("Conversation is too long")
        return self


def identity(backend, request_id, session_id="anonymous"):
    return dict(request_id=request_id, session_id=session_id, pod=backend.settings.pod_name,
                model_version=backend.loaded_version, artifact_sha256=backend.loaded_sha256,
                backend_mode="simulated")


def record(backend, request, outcome, **details):
    if request.scope["state"].get("accounted"):
        return
    request.scope["state"]["accounted"] = True
    backend.metrics.requests.labels(outcome).inc()
    print(json.dumps(dict(event="request_finished", **identity(backend, request.state.request_id,
                      getattr(request.state, "session_id", "anonymous")), outcome=outcome, source_ip=request.client.host if request.client else None,
                      backend_ready=backend.warmup_finished, utc=time.time(), monotonic=time.monotonic(),
                      running=backend.running, waiting=backend.waiting, **details), ensure_ascii=False), flush=True)


class ManagedStreamingResponse(StreamingResponse):
    """Close the iterator on send failure as well as on normal exhaustion."""

    def __init__(self, content, cleanup, **kwargs):
        super().__init__(content, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                # Closing an unstarted async generator does not execute its finally.
                self.cleanup()


class IdentityHeaders:
    """Pure ASGI middleware keeps streaming/cancellation semantics intact."""

    def __init__(self, app, backend):
        self.app = app
        self.backend = backend

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["request_id"] = str(uuid.uuid4())

        async def identified(message):
            if message["type"] == "http.response.start":
                values = identity(self.backend, state["request_id"])
                headers = {"x-request-id": values["request_id"], "x-pod-name": values["pod"],
                           "x-model-version": values["model_version"] or "unloaded",
                           "x-artifact-sha256": values["artifact_sha256"] or "unloaded",
                           "x-backend-mode": "simulated"}
                message["headers"] = list(message.get("headers", [])) + [
                    (key.encode(), str(value).encode()) for key, value in headers.items()]
                if scope["path"] in ("/stream", "/generate") and message["status"] >= 400 and not state.get("accounted"):
                    record(self.backend, Request(scope), "rejected", status=message["status"], reason="invalid_request")
            await send(message)

        await self.app(scope, receive, identified)


def create_app(settings=None):
    settings = settings or Settings.from_env()
    backend = Backend(settings)

    @asynccontextmanager
    async def lifespan(app):
        loading = asyncio.create_task(backend.load())
        try:
            yield
        finally:
            loading.cancel()
            with suppress(asyncio.CancelledError):
                await loading
            await backend.close()

    app = FastAPI(lifespan=lifespan)
    app.state.backend = backend
    app.add_middleware(BodySizeLimit)
    app.add_middleware(IdentityHeaders, backend=backend)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Invalid nonfinite JSON numbers cannot be serialized by FastAPI's default input echo.
        return JSONResponse({"detail": [{"type": item["type"], "loc": item["loc"],
                                        "msg": item["msg"]} for item in exc.errors()]}, status_code=422)

    @app.get("/health")
    async def health():
        return dict(alive=True, backend_mode="simulated")

    @app.get("/startup")
    async def startup():
        good = settings.broken_probes or backend.warmup_finished
        return JSONResponse(dict(started=good, **backend.debug()), status_code=200 if good else 503)

    @app.get("/ready")
    async def ready():
        good = not backend.draining and (settings.broken_probes or backend.warmup_finished)
        return JSONResponse({**backend.debug(), "ready": good}, status_code=200 if good else 503)

    @app.post("/drain")
    async def drain():
        # Keep accepting during EndpointSlice propagation; SIGTERM stops HTTP admission.
        backend.draining = True
        return backend.debug()

    @app.get("/debug")
    async def debug():
        return backend.debug()

    @app.post("/reset-cache")
    async def reset_cache():
        if backend.running or backend.waiting:
            return JSONResponse({"error": "cache reset requires idle backend"}, status_code=409)
        backend.cache.clear()
        return backend.debug()

    @app.get("/metrics")
    async def metrics():
        return Response(backend.metrics.render(), media_type="text/plain; version=0.0.4")

    async def admit(request):
        print(json.dumps(dict(event="request_received",
                              **identity(backend, request.state.request_id,
                                         getattr(request.state, "session_id", "anonymous")),
                              source_ip=request.client.host if request.client else None,
                              backend_ready=backend.warmup_finished, utc=time.time(),
                              monotonic=time.monotonic()), ensure_ascii=False), flush=True)
        try:
            return await backend.acquire()
        except AdmissionError as exc:
            record(backend, request, "rejected", status=exc.status, reason=exc.reason)
            return JSONResponse({"error": exc.reason, **identity(backend, request.state.request_id,
                                getattr(request.state, "session_id", "anonymous"))}, status_code=exc.status)
        except asyncio.CancelledError:
            record(backend, request, "interrupted", reason="cancelled_while_waiting")
            raise

    @app.post("/stream")
    async def stream(payload: StreamRequest, request: Request):
        request.state.request_id = payload.request_id or request.state.request_id
        request.state.session_id = payload.session_id
        started = time.monotonic()
        lease = await admit(request)
        if isinstance(lease, JSONResponse):
            return lease
        common = identity(backend, request.state.request_id, payload.session_id)

        async def events():
            first = None
            completed = False
            try:
                hit = await backend.prepare(lease, payload.prompt, payload.messages)
                context = "\n".join([message.content for message in payload.messages] + [payload.prompt])
                content = f"SIMULATED {backend.loaded_version}: {context}"
                for index in range(payload.max_chunks):
                    await lease.compute(settings.chunk_iterations)
                    await asyncio.sleep(payload.chunk_delay_seconds)
                    if first is None:
                        first = time.monotonic()
                    event = dict(event="chunk", type="chunk", **common, index=index, content=content if index == 0 else f"part-{index}")
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                finished = time.monotonic()
                done = dict(event="done", type="done", **common, chunks=payload.max_chunks,
                            queue_seconds=lease.queue_seconds, cache_hit=hit,
                            ttfc_seconds=first - started, total_seconds=finished - started)
                yield json.dumps(done) + "\n"
                completed = True
                backend.metrics.ttfc.observe(first - started)
                backend.metrics.duration.observe(finished - started)
                record(backend, request, "completed", chunks=payload.max_chunks, queue_seconds=lease.queue_seconds,
                       cache_hit=hit, ttfc_seconds=first - started, total_seconds=finished - started)
            except asyncio.CancelledError:
                raise
            except Exception:
                record(backend, request, "error", reason="generation_failed")
                raise
            finally:
                if not completed:
                    record(backend, request, "interrupted", reason="stream_not_completed")
                lease.close()

        def cleanup():
            record(backend, request, "interrupted", reason="response_not_completed")
            lease.close()

        return ManagedStreamingResponse(events(), cleanup=cleanup, media_type="application/x-ndjson")

    @app.post("/generate")
    async def generate(payload: GenerateRequest, request: Request):
        started = time.monotonic()
        limit = payload.max_new_tokens if payload.max_new_tokens is not None else 32
        if limit != 32:
            return JSONResponse({"error": "Use fixed simulator max_new_tokens=32"}, status_code=422)
        lease = await admit(request)
        if isinstance(lease, JSONResponse):
            return lease
        try:
            hit = await backend.prepare(lease, payload.prompt, [])
            finished = time.monotonic()
            backend.metrics.ttfc.observe(finished - started)
            backend.metrics.duration.observe(finished - started)
            record(backend, request, "completed", chunks=1, cache_hit=hit, queue_seconds=lease.queue_seconds,
                   ttfc_seconds=finished - started, total_seconds=finished - started)
            return dict(answer=f"SIMULATED {backend.loaded_version}: {payload.prompt}", output_tokens=1,
                        model_id=backend.loaded_version, runtime_sha256=backend.loaded_sha256,
                        **identity(backend, request.state.request_id), queue_seconds=lease.queue_seconds,
                        generation_seconds=finished - started - lease.queue_seconds, total_seconds=finished - started)
        except asyncio.CancelledError:
            record(backend, request, "interrupted", reason="generate_cancelled")
            raise
        except Exception:
            record(backend, request, "error", reason="generation_failed")
            return JSONResponse({"error": "generation_failed"}, status_code=500)
        finally:
            lease.close()

    return app
