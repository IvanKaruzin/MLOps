"""Local FastAPI model service with an explicit loaded-version boundary.

The default backend is visibly synthetic. Real weights are loaded only during
the application lifespan and only with ``--backend transformers``.
"""

import argparse
import asyncio
import copy
import json
import math
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator


MAX_PROMPT_CHARS = 4096
MAX_BODY_BYTES = 65536


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: StrictStr = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: StrictInt | None = None

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace characters")
        return value

    @model_validator(mode="before")
    @classmethod
    def explicit_limit_must_be_an_integer(cls, value):
        if isinstance(value, dict) and "max_new_tokens" in value and value["max_new_tokens"] is None:
            raise ValueError("max_new_tokens cannot be null")
        return value


class BodySizeLimit:
    """Bound the actual streamed body, including requests without Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                response = JSONResponse({"error": "body too large"}, status_code=413)
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class StubBackend:
    """A backend for control-plane tests; never presented as a model answer."""

    def __init__(self, manifest, workspace=None):
        runtime = manifest["runtime"]
        legacy_fixture = (manifest["model_id"].startswith("fixture-")
                          and runtime.get("base_revision") == "fixture-only")
        if manifest.get("synthetic") is not True and not legacy_fixture:
            raise ValueError("A real manifest requires the transformers backend")
        self.loaded_id = manifest["model_id"]
        self.runtime_sha256 = manifest["runtime_sha256"]
        self.max_new_tokens = runtime["generation"]["max_new_tokens"]
        self.calls = 0
        self.warmup_calls = 0

    def warmup(self):
        time.sleep(0.01)
        self.warmup_calls += 1

    def generate(self, prompt, limit):
        if limit != self.max_new_tokens:
            raise ValueError("Generation limit differs from the loaded runtime")
        self.calls += 1
        time.sleep(0.01)
        return {"answer": f"STUB ONLY: {self.loaded_id}", "output_tokens": 1}


def transformers_backend(manifest, workspace):
    # Importing the app, or testing with StubBackend, never imports model code.
    from hw8910.runtime import ModelBackend

    return ModelBackend(manifest, workspace)


class ServiceState:
    def __init__(self, root, warmup_seconds=5, backend_factory=StubBackend):
        if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
            raise ValueError("warmup_seconds must be finite and nonnegative")
        self.root = Path(root).resolve()
        self.warmup_seconds = warmup_seconds
        self.backend_factory = backend_factory
        self.backend = None
        self.loaded_id = None
        self.runtime_sha256 = None
        self.max_new_tokens = None
        self.load_error = None
        self.warmup_finished = threading.Event()
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._generation_lock = threading.Lock()

    @property
    def ready(self):
        return self._ready.is_set()

    def load(self):
        try:
            from release import verify_artifact

            manifest = json.loads((self.root / "accepted.json").read_text(encoding="utf-8"))
            verify_artifact(manifest, self.root)
            backend = self.backend_factory(copy.deepcopy(manifest), self.root)
            limit = manifest["runtime"]["generation"]["max_new_tokens"]
            if (type(limit) is not int or limit <= 0
                    or backend.loaded_id != manifest["model_id"]
                    or backend.runtime_sha256 != manifest["runtime_sha256"]
                    or backend.max_new_tokens != limit):
                raise ValueError("Loaded backend differs from the accepted runtime")
            self.backend = backend
            self.loaded_id = backend.loaded_id
            self.runtime_sha256 = backend.runtime_sha256
            self.max_new_tokens = limit
            if self._stopping.wait(self.warmup_seconds):
                return
            backend.warmup()
            if not self._stopping.is_set():
                self.warmup_finished.set()
                self._ready.set()
        except Exception as exc:
            # Do not expose filesystem paths or backend exception contents.
            self.load_error = type(exc).__name__
            self._ready.clear()

    def stop(self):
        self._ready.clear()
        self._stopping.set()

    def close(self):
        with self._generation_lock:
            if self.backend is not None:
                close = getattr(self.backend, "close", None)
                if callable(close):
                    close()

    def generate(self, request):
        if not isinstance(request, GenerateRequest):
            request = GenerateRequest.model_validate(request)
        if not self.ready:
            raise HTTPException(status_code=503, detail="Model is not ready")
        limit = self.max_new_tokens if request.max_new_tokens is None else request.max_new_tokens
        if limit != self.max_new_tokens:
            raise HTTPException(status_code=422, detail="Use the fixed evaluated max_new_tokens")
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        with self._generation_lock:
            acquired = time.perf_counter()
            try:
                output = self.backend.generate(request.prompt, limit)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="Input is unsupported by the loaded runtime") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Generation failed") from exc
            finished = time.perf_counter()
        if (not isinstance(output, dict) or not isinstance(output.get("answer"), str)
                or type(output.get("output_tokens")) is not int
                or not 0 <= output["output_tokens"] <= limit):
            raise HTTPException(status_code=500, detail="Backend returned an invalid response")
        return dict(answer=output["answer"], output_tokens=output["output_tokens"],
                    request_id=request_id, model_id=self.loaded_id, runtime_sha256=self.runtime_sha256,
                    queue_seconds=acquired - started, generation_seconds=finished - acquired,
                    total_seconds=finished - started)


def create_app(workspace=Path(".lab"), warmup_seconds=5, backend_factory=None, backend_kind="stub"):
    if backend_kind not in ("stub", "transformers"):
        raise ValueError("Unknown backend kind")
    factory = backend_factory or (StubBackend if backend_kind == "stub" else transformers_backend)
    state = ServiceState(workspace, warmup_seconds, factory)

    @asynccontextmanager
    async def lifespan(app):
        loading = asyncio.create_task(asyncio.to_thread(state.load))
        try:
            yield
        finally:
            state.stop()
            await loading
            state.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(BodySizeLimit)
    app.state.service = state

    @app.get("/health")
    def health():
        return {"alive": True, "backend": "STUB ONLY" if backend_kind == "stub" else "transformers"}

    @app.get("/ready")
    def readiness():
        payload = dict(ready=state.ready, model_id=state.loaded_id, runtime_sha256=state.runtime_sha256)
        if state.load_error is not None:
            payload["error"] = state.load_error
        return JSONResponse(payload, status_code=200 if state.ready else 503)

    @app.post("/generate")
    def generate(request: GenerateRequest):
        return state.generate(request)

    if backend_kind == "stub":
        @app.get("/debug")
        def debug():
            return dict(loaded_id=state.loaded_id, runtime_sha256=state.runtime_sha256,
                        calls=getattr(state.backend, "calls", 0),
                        warmup_calls=getattr(state.backend, "warmup_calls", 0),
                        warmup_finished=state.warmup_finished.is_set())

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(".lab"))
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--warmup-seconds", type=float, default=5)
    parser.add_argument("--backend", choices=("stub", "transformers"), default="stub")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("Invalid port or warmup duration")
    import uvicorn

    app = create_app(args.workspace, args.warmup_seconds, backend_kind=args.backend)
    label = "STUB ONLY" if args.backend == "stub" else "TRANSFORMERS"
    print(f"{label}: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
