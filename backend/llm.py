"""Thin async wrapper around the Anthropic SDK.

Centralizes the client and a streaming helper so the agents don't each
re-implement the stream-and-accumulate boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

import anthropic
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from .config import MODEL

# Load .env *before* constructing the client — the SDK reads ANTHROPIC_API_KEY
# from the environment at construction time, so this must happen first. We point
# at the project-root .env explicitly so it works regardless of the cwd.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
client = AsyncAnthropic()

OnDelta = Callable[[str], Awaitable[None]]


def friendly_error(e: Exception) -> str:
    """Turn a raw SDK exception into a message a human can act on."""
    msg = str(getattr(e, "message", "") or e)
    low = msg.lower()
    if isinstance(e, anthropic.AuthenticationError) or "invalid x-api-key" in low:
        return "Invalid API key. Check ANTHROPIC_API_KEY in your .env file (get one at console.anthropic.com)."
    if "credit balance is too low" in low or "plans & billing" in low:
        return ("Your Anthropic account is out of credits. Add credits at "
                "console.anthropic.com → Plans & Billing, then run again.")
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the API. Wait a few seconds and try again."
    return f"{type(e).__name__}: {msg}"


def is_fatal_account_error(e: Exception) -> bool:
    """Account-level failures (bad key, no credits, no access) that no fallback can rescue."""
    low = str(getattr(e, "message", "") or e).lower()
    return (
        isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError))
        or "credit balance is too low" in low
        or "invalid x-api-key" in low
    )


async def stream_text(
    *,
    system: str,
    user: str,
    on_delta: OnDelta,
    max_tokens: int = 2600,
    model: str = MODEL,
) -> str:
    """Stream a single-turn completion, calling `on_delta` for each text chunk.

    Returns the fully accumulated text once the stream closes. Streaming keeps
    large `max_tokens` requests under the SDK's HTTP timeout and powers the
    live "typing" effect in the UI.
    """
    parts: list[str] = []
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                chunk = event.delta.text
                parts.append(chunk)
                await on_delta(chunk)
    return "".join(parts)
