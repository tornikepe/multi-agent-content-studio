"""FastAPI app. Each endpoint streams one pipeline phase as SSE.

Stateless by design — no job registry, no background tasks that outlive a
request — so it runs identically on an always-on host or serverless (Vercel).

    POST /api/generate  -> SSE  research -> write -> review -> awaiting_approval
    POST /api/revise    -> SSE  edit against feedback -> review -> awaiting_approval
    POST /api/publish   -> SSE  final polish -> done
    GET  /              -> the single-page UI
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from . import orchestrator
from .llm import friendly_error
from .models import STREAM_END, Emitter, GenerateRequest, PublishRequest, ReviseRequest

app = FastAPI(title="Multi-Agent Content Studio")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def sse(phase: Callable[[Emitter], Awaitable[None]]) -> StreamingResponse:
    """Run a pipeline phase in the background and stream its events as SSE.

    The producer task and the consumer generator share one in-request queue, so
    the whole exchange lives inside a single HTTP request — serverless-safe.
    """
    em = Emitter()

    async def run() -> None:
        try:
            await phase(em)
        except Exception as e:  # surface any failure in plain language
            await em.emit("error", message=friendly_error(e))
        finally:
            await em.close()

    async def gen():
        task = asyncio.create_task(run())
        try:
            while True:
                event = await em.queue.get()
                if event["type"] == STREAM_END:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            await task

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    return sse(lambda em: orchestrator.generate(em, req.topic.strip(), req.options()))


@app.post("/api/revise")
async def revise(req: ReviseRequest):
    return sse(lambda em: orchestrator.revise(em, req.topic.strip(), req.options(), req.draft, req.feedback))


@app.post("/api/publish")
async def publish(req: PublishRequest):
    return sse(lambda em: orchestrator.publish(em, req.topic.strip(), req.options(), req.draft))


@app.get("/")
async def index():
    return FileResponse(FRONTEND)


@app.get("/health")
async def health():
    return {"status": "ok"}
