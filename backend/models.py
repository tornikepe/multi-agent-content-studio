"""Request schemas and a lightweight per-request event Emitter.

The app is stateless: each phase (generate / revise / publish) is one HTTP
request that streams its own events. The client holds the draft between phases,
so there is no server-side session — which is what makes it deploy cleanly to
serverless (Vercel) as well as any always-on host.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .config import MAX_TOPIC_CHARS

# Sentinel pushed onto the queue to tell the SSE generator to close.
STREAM_END = "__end__"


class _Base(BaseModel):
    topic: str = Field(min_length=3, max_length=MAX_TOPIC_CHARS)
    tone: str = "informative"
    length: str = "medium"
    angle: str | None = Field(default=None, max_length=MAX_TOPIC_CHARS)
    language: str = "en"  # "en" | "ka" — language of the generated content

    def options(self) -> dict:
        return {"tone": self.tone, "length": self.length, "angle": self.angle, "language": self.language}


class GenerateRequest(_Base):
    """Kick off research -> write -> review -> (edit loop)."""


class ReviseRequest(_Base):
    """Human requested changes: re-edit the draft against feedback, then re-review."""
    draft: str
    feedback: str | None = None


class PublishRequest(_Base):
    """Human approved: final polish + publish."""
    draft: str


@dataclass
class Emitter:
    """Collects pipeline events for one request and hands them to the SSE stream."""

    queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)

    async def emit(self, type: str, **data) -> None:
        await self.queue.put({"type": type, **data})

    async def stage_start(self, agent: str, label: str) -> None:
        await self.emit("stage_start", agent=agent, label=label)

    async def close(self) -> None:
        await self.queue.put({"type": STREAM_END})
