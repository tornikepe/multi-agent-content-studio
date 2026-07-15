"""The five agents. Each is a coroutine that does one job and emits SSE events.

    Researcher -> Writer -> Reviewer -> (Editor loop) -> [human gate] -> Publisher

All model calls go through the active `provider` (Claude / Groq / Gemini / offline).
Only Claude supports live web search; other providers research from model knowledge.
Writer/Editor/Publisher stream token-by-token; Reviewer returns a structured JSON verdict.
"""

from __future__ import annotations

from . import config
from .llm import is_fatal_account_error, offline_provider, provider
from .models import Emitter


async def _token_emitter(em: Emitter, agent: str):
    async def on_delta(text: str) -> None:
        await em.emit("token", agent=agent, text=text)
    return on_delta


def _lang(options: dict) -> str:
    return options.get("language", "en")


# Resilient wrappers: on a non-fatal provider error (rate limit / transient),
# fall back to the offline provider so a run always completes instead of erroring.
async def _stream(**kw) -> str:
    try:
        return await provider.stream(**kw)
    except Exception as e:
        if is_fatal_account_error(e):
            raise
        return await offline_provider.stream(**kw)


async def _json(**kw) -> dict:
    try:
        return await provider.complete_json(**kw)
    except Exception as e:
        if is_fatal_account_error(e):
            raise
        return await offline_provider.complete_json(**kw)


# --- Researcher --------------------------------------------------------------

async def run_researcher(em: Emitter, topic: str, options: dict) -> tuple[str, list[dict]]:
    await em.stage_start("researcher", "Researcher")
    lang = _lang(options)

    if provider.supports_web_search:
        try:
            notes, sources = await _web_research(em, topic, options)
            if notes.strip():
                if sources:
                    await em.emit("sources", sources=sources)
                await em.emit("stage_output", agent="researcher", content=notes)
                return notes, sources
        except Exception as e:
            if is_fatal_account_error(e):
                raise
            await em.emit("log", level="warn",
                          message=f"Web search unavailable ({type(e).__name__}); using model knowledge instead.")

    # No web search (free/offline providers, or Claude web search unavailable):
    # produce research notes from model knowledge.
    notes = await _stream(
        system=config.sys(config.FALLBACK_RESEARCHER_SYSTEM, options),
        user=config.researcher_user(topic, options),
        on_delta=await _token_emitter(em, "researcher"),
        max_tokens=2000, kind="research", lang=lang,
    )
    await em.emit("stage_output", agent="researcher", content=notes)
    return notes, []


async def _web_research(em: Emitter, topic: str, options: dict) -> tuple[str, list[dict]]:
    """Claude-only: gather cited research via the server-side web_search tool."""
    client = provider.client  # type: ignore[attr-defined]
    tools = [{"type": "web_search_20260209", "name": "web_search"}]
    messages = [{"role": "user", "content": config.researcher_user(topic, options)}]
    sources: list[dict] = []
    while True:
        resp = await client.messages.create(
            model=provider.model, max_tokens=8000,  # type: ignore[attr-defined]
            system=config.sys(config.RESEARCHER_SYSTEM, options), tools=tools, messages=messages,
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
    seen: set[str] = set()
    unique = [s for s in sources if not (s["url"] in seen or seen.add(s["url"]))][:8]
    return notes, unique


# --- Writer ------------------------------------------------------------------

async def run_writer(em: Emitter, topic: str, options: dict, notes: str) -> str:
    await em.stage_start("writer", "Writer")
    draft = await _stream(
        system=config.sys(config.WRITER_SYSTEM, options),
        user=config.writer_user(topic, options, notes),
        on_delta=await _token_emitter(em, "writer"),
        max_tokens=config.length_tokens(options), kind="write", lang=_lang(options),
    )
    await em.emit("stage_output", agent="writer", content=draft)
    return draft


# --- Reviewer ----------------------------------------------------------------

async def run_reviewer(em: Emitter, topic: str, options: dict, draft: str, cycle: int = 0) -> dict:
    await em.stage_start("reviewer", "Reviewer" if cycle == 0 else f"Reviewer · re-check {cycle}")
    data = await _json(
        system=config.sys(config.REVIEWER_SYSTEM, options),
        user=config.reviewer_user(topic, options, draft),
        schema=config.REVIEW_SCHEMA, lang=_lang(options),
    )
    data["score"] = max(1, min(10, int(data.get("score", 5))))
    data.setdefault("verdict", "")
    data.setdefault("strengths", [])
    data.setdefault("issues", [])
    await em.emit("review", revision=cycle, **data)
    return data


# --- Editor ------------------------------------------------------------------

async def run_editor(em: Emitter, topic: str, options: dict, draft: str, review: dict, cycle: int) -> str:
    await em.emit("revision_start", cycle=cycle)
    await em.stage_start("editor", f"Editor · revision {cycle}")
    revised = await _stream(
        system=config.sys(config.EDITOR_SYSTEM, options),
        user=config.editor_user(topic, options, draft, review),
        on_delta=await _token_emitter(em, "editor"),
        max_tokens=config.length_tokens(options), kind="edit", lang=_lang(options),
    )
    await em.emit("stage_output", agent="editor", content=revised)
    return revised


# --- Publisher ---------------------------------------------------------------

async def run_publisher(em: Emitter, topic: str, options: dict, draft: str) -> str:
    await em.stage_start("publisher", "Publisher")
    final = await _stream(
        system=config.sys(config.PUBLISHER_SYSTEM, options),
        user=config.publisher_user(topic, options, draft),
        on_delta=await _token_emitter(em, "publisher"),
        max_tokens=config.length_tokens(options) + 800, kind="publish", lang=_lang(options),
    )
    await em.emit("stage_output", agent="publisher", content=final)
    return final
