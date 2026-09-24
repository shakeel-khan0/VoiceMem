"""Loopback HTTP sidecar for caller-isolated VoiceMem access."""

from __future__ import annotations

import hashlib
import contextlib
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# The verified E5 model is local. Avoid startup network probes and fail clearly
# if that local model cache is ever removed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from voicemem import VoiceMem


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MEMORY_MODEL = "openai/gpt-oss-120b"
DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent / "data"


class SearchRequest(BaseModel):
    caller_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class IngestRequest(BaseModel):
    caller_id: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=16000)


class VoiceMemManager:
    """Own long-lived VoiceMem instances and the shared warm local E5 model."""

    def __init__(self, memory_root: str | Path | None = None) -> None:
        self.memory_root = Path(
            memory_root or os.environ.get("VOICEMEM_SIDECAR_MEMORY_ROOT", DEFAULT_MEMORY_ROOT)
        ).resolve()
        self._clients: dict[str, VoiceMem] = {}
        self._ingest_lock = threading.Lock()
        self.available = False
        self.error_type: str | None = None

    def initialize(self) -> None:
        try:
            if not os.environ.get("GROQ_API_KEY", "").strip():
                raise RuntimeError("GROQ_API_KEY is not configured")
            self.memory_root.mkdir(parents=True, exist_ok=True)
            # Build the terminal caller up front and load shared E5 once.
            caller_id = os.environ.get("TEST_CALLER_ID", "terminal_test_001").strip()
            client = self.client(caller_id)
            client.warmup(audio=False, verbose=False)
            # VoiceMem initializes a few search-only components lazily, so one
            # empty-store lookup keeps that cold cost out of the first live turn.
            client.search("sidecar startup warmup", top_k=1)
            self.available = True
            self.error_type = None
        except Exception as exc:
            self.available = False
            self.error_type = type(exc).__name__

    def client(self, caller_id: str) -> VoiceMem:
        caller_id = caller_id.strip()
        if not caller_id:
            raise ValueError("caller_id is required")
        if caller_id not in self._clients:
            namespace = hashlib.sha256(caller_id.encode("utf-8")).hexdigest()[:24]
            self._clients[caller_id] = VoiceMem.from_config({
                "mode": "text",
                "memory_root": str(self.memory_root / namespace),
                "user_id": caller_id,
                "llm": {
                    "provider": "groq",
                    "config": {
                        "model": GROQ_MEMORY_MODEL,
                        "base_url": GROQ_BASE_URL,
                    },
                },
                "embedding": {"provider": "local"},
                "slots": {"provider": "local"},
            })
        return self._clients[caller_id]

    def search(self, request: SearchRequest) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("VoiceMem is unavailable")
        started = perf_counter()
        result = self.client(request.caller_id).search(request.query, top_k=request.top_k)
        memories = [
            *[str(value) for value in result.result_leftbrain],
            *[str(value) for value in result.result_rightbrain],
        ]
        return {
            "ok": True,
            "memories": memories,
            "count": len(memories),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        }

    def ingest(self, request: IngestRequest) -> dict[str, Any]:
        """Manual preparation endpoint; the live Phase 2 agent never calls it."""
        if not self.available:
            raise RuntimeError("VoiceMem is unavailable")
        # VoiceMem's standalone CLI prints truncated extracted facts. The
        # sidecar keeps private caller content out of production logs.
        captured = io.StringIO()
        with self._ingest_lock, contextlib.redirect_stdout(captured):
            result = self.client(request.caller_id).ingest(request.content)
        internal_output = captured.getvalue()
        if "抽取失败" in internal_output or "没能入库" in internal_output:
            return {
                "ok": False,
                "persisted": False,
                "memory_count": 0,
                "error_type": "VoiceMemIngestError",
            }
        persisted = bool(result.get("persistent_memory_created"))
        return {
            "ok": True,
            "persisted": persisted,
            "memory_count": len(result.get("memory_ids", [])),
        }

    def close(self) -> None:
        # Ingest is synchronous and already persistent. VoiceMem.flush() runs
        # optional LLM batch work, which is unrelated to this read-only live
        # path and would make sidecar shutdown wait on an external service.
        self._clients.clear()


def create_app(manager: VoiceMemManager | None = None) -> FastAPI:
    memory = manager or VoiceMemManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await run_in_threadpool(memory.initialize)
        yield
        await run_in_threadpool(memory.close)

    app = FastAPI(title="VoiceMem Local Sidecar", lifespan=lifespan)
    app.state.memory = memory

    @app.get("/health")
    async def health():
        return {
            "alive": True,
            "voicemem_available": memory.available,
            "error_type": memory.error_type,
        }

    @app.post("/memory/search")
    async def search(request: SearchRequest):
        try:
            return await run_in_threadpool(memory.search, request)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=type(exc).__name__) from None

    @app.post("/memory/ingest")
    async def ingest(request: IngestRequest):
        try:
            return await run_in_threadpool(memory.ingest, request)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=type(exc).__name__) from None

    return app


app = create_app()
