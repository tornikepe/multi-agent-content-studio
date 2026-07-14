"""The five agents. Each is a coroutine that does one job and emits SSE events.

    Researcher -> Writer -> Reviewer -> (Editor loop) -> [human gate] -> Publisher

Researcher uses Claude's server-side `web_search` tool. Writer/Editor/Publisher
stream their output token-by-token. Reviewer returns a structured JSON verdict.
Each agent takes an `Emitter` and reports progress through it.
"""

from __future__ import annotations

import json

from . import config
from .llm import client, is_fatal_account_error, stream_text
from .models import Emitter


async def _token_emitter(em: Emitter, agent: str):
    async def on_delta(text: str) -> None:
        await em.emit("token", agent=agent, text=text)
    return on_delta


# --- Researcher --------------------------------------------------------------

async def run_researcher(em: Emitter, topic: str, options: dict) -> tuple[str, list[dict]]:
    await em.stage_start("researcher", "Researcher")
    tools = [{"type": "web_search_20260209", "name": "web_search"}]
    messages = [{"role": "user", "content": config.researcher_user(topic, options)}]
    sources: list[dict] = []

    try:
        # Server-side web search runs inside the create call. If it needs more
        # than the built-in iteration budget it returns `pause_turn`; we append
        # the assistant turn and resend to let it continue.
        while True:
            resp = await client.messages.create(
                model=config.MODEL,
                max_tokens=8000,
                system=config.RESEARCHER_SYSTEM,
                tools=tools,
                messages=messages,
            )
            for block in resp.content:
                btype = getattr(block, "type", None)
                if btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
                    query = (getattr(block, "input", None) or {}).get("query")
                    if query:
                        await em.emit("search", query=query)
                elif btype == "web_search_tool_result":
                    content = getattr(block, "content", None)
                    if isinstance(content, list):
                        for r in content:
                            if getattr(r, "type", None) == "web_search_result":
                                sources.append({"title": getattr(r, "title", "") or r.url, "url": r.url})
            if resp.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            break
        notes = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not notes.strip():
            raise RuntimeError("empty research output")
    except Exception as e:
        # Account-level failures (bad key / no credits) can't be rescued by a
        # fallback — let them bubble up so the user sees the real reason.
        if is_fatal_account_error(e):
            raise
        # Otherwise web search is just unavailable — degrade to model knowledge.
        await em.emit("log", level="warn",
                      message=f"Web search unavailable ({type(e).__name__}); using model knowledge instead.")
        notes = await stream_text(
            system=config.FALLBACK_RESEARCHER_SYSTEM,
            user=config.researcher_user(topic, options),
            on_delta=await _token_emitter(em, "researcher"),
            max_tokens=2000,
        )

    # De-duplicate sources by URL, keep the first 8.
    seen: set[str] = set()
    unique: list[dict] = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique.append(s)
    unique = unique[:8]
    if unique:
        await em.emit("sources", sources=unique)
    await em.emit("stage_output", agent="researcher", content=notes)
    return notes, unique


# --- Writer ------------------------------------------------------------------

async def run_writer(em: Emitter, topic: str, options: dict, notes: str) -> str:
    await em.stage_start("writer", "Writer")
    draft = await stream_text(
        system=config.WRITER_SYSTEM,
        user=config.writer_user(topic, options, notes),
        on_delta=await _token_emitter(em, "writer"),
        max_tokens=config.length_tokens(options),
    )
    await em.emit("stage_output", agent="writer", content=draft)
    return draft


# --- Reviewer ----------------------------------------------------------------

async def run_reviewer(em: Emitter, topic: str, options: dict, draft: str, cycle: int = 0) -> dict:
    await em.stage_start("reviewer", "Reviewer" if cycle == 0 else f"Reviewer · re-check {cycle}")
    resp = await client.messages.create(
        model=config.MODEL,
        max_tokens=4000,
        system=config.REVIEWER_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": config.REVIEW_SCHEMA},
                       "effort": "high"},
        messages=[{"role": "user", "content": config.reviewer_user(topic, options, draft)}],
    )
    text = next(b.text for b in resp.content if getattr(b, "type", None) == "text")
    data = json.loads(text)
    data["score"] = max(1, min(10, int(data.get("score", 5))))
    await em.emit("review", revision=cycle, **data)
    return data


# --- Editor ------------------------------------------------------------------

async def run_editor(em: Emitter, topic: str, options: dict, draft: str, review: dict, cycle: int) -> str:
    await em.emit("revision_start", cycle=cycle)
    await em.stage_start("editor", f"Editor · revision {cycle}")
    revised = await stream_text(
        system=config.EDITOR_SYSTEM,
        user=config.editor_user(topic, options, draft, review),
        on_delta=await _token_emitter(em, "editor"),
        max_tokens=config.length_tokens(options),
    )
    await em.emit("stage_output", agent="editor", content=revised)
    return revised


# --- Publisher ---------------------------------------------------------------

async def run_publisher(em: Emitter, topic: str, options: dict, draft: str) -> str:
    await em.stage_start("publisher", "Publisher")
    final = await stream_text(
        system=config.PUBLISHER_SYSTEM,
        user=config.publisher_user(topic, options, draft),
        on_delta=await _token_emitter(em, "publisher"),
        max_tokens=config.length_tokens(options) + 800,
    )
    await em.emit("stage_output", agent="publisher", content=final)
    return final
