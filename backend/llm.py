"""LLM provider layer — pluggable backends so the app runs on any (or no) key.

Providers, in auto-detect order (override with the LLM_PROVIDER env var):
    groq      — free, fast (Llama 3.3 70B). Set GROQ_API_KEY.
    gemini    — free, strong multilingual (Gemini 2.0 Flash). Set GEMINI_API_KEY.
    anthropic — Claude Opus 4.8; the only provider with live web search. Set ANTHROPIC_API_KEY.
    offline   — no key needed. Synthesizes a demo run so the site always works.

Free providers win the auto-detect over Anthropic so that "add a free key → it
works" holds even if a (possibly out-of-credit) ANTHROPIC_API_KEY is also set.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Awaitable, Callable

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OnDelta = Callable[[str], Awaitable[None]]


# --------------------------------------------------------------------------- #
# Error helpers
# --------------------------------------------------------------------------- #

def friendly_error(e: Exception) -> str:
    """Turn a raw provider exception into a message a human can act on."""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code in (401, 403):
            return "The provider rejected the API key — check GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY."
        if code == 429:
            return "The provider is rate-limiting or the free quota is used up. Wait a moment and retry, or switch providers."
        detail = ""
        try:
            detail = (e.response.json().get("error", {}) or {}).get("message", "") or e.response.text[:160]
        except Exception:
            detail = e.response.text[:160]
        return f"Provider error {code}. {detail}"

    msg = str(getattr(e, "message", "") or e)
    low = msg.lower()
    try:
        import anthropic
        if isinstance(e, anthropic.AuthenticationError) or "invalid x-api-key" in low:
            return "Invalid Anthropic API key. Check ANTHROPIC_API_KEY (or switch to a free provider — see .env.example)."
        if isinstance(e, anthropic.RateLimitError):
            return "Rate limited by the API. Wait a few seconds and try again."
    except Exception:
        pass
    if "credit balance is too low" in low or "plans & billing" in low:
        return ("Your Anthropic account is out of credits. Add credits at console.anthropic.com → Plans & Billing, "
                "or use a free provider (Groq/Gemini) — see .env.example.")
    return f"{type(e).__name__}: {msg}"


def is_fatal_account_error(e: Exception) -> bool:
    """Account-level failures that no fallback can rescue (used in the web-search path)."""
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403):
        return True
    low = str(getattr(e, "message", "") or e).lower()
    try:
        import anthropic
        if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return True
    except Exception:
        pass
    return "credit balance is too low" in low or "invalid x-api-key" in low


# --------------------------------------------------------------------------- #
# HTTP retry helpers (handle free-tier rate limits gracefully)
# --------------------------------------------------------------------------- #

_RETRY_CODES = {429, 500, 502, 503}


def _retry_wait(resp, fallback: float) -> float:
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return min(float(ra), 15.0)
        except ValueError:
            pass
    return fallback


async def _post_retry(url: str, headers: dict, payload: dict):
    delay, last = 2.0, None
    async with httpx.AsyncClient(timeout=180) as c:
        for attempt in range(3):
            r = await c.post(url, headers=headers, json=payload)
            if r.status_code in _RETRY_CODES and attempt < 2:
                last = httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                await asyncio.sleep(_retry_wait(r, delay)); delay *= 2; continue
            r.raise_for_status()
            return r
    raise last


async def _stream_retry(url: str, headers: dict, payload: dict, parse, on_delta) -> str:
    """Stream an SSE response; retry the whole request on 429/5xx before any token is emitted."""
    delay, last = 2.0, None
    for attempt in range(3):
        wait, parts = None, []
        async with httpx.AsyncClient(timeout=180) as c:
            async with c.stream("POST", url, headers=headers, json=payload) as r:
                if r.status_code >= 400:
                    await r.aread()
                    err = httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    if r.status_code in _RETRY_CODES and attempt < 2:
                        last, wait = err, _retry_wait(r, delay)
                    else:
                        raise err
                else:
                    async for line in r.aiter_lines():
                        d = parse(line)
                        if d:
                            parts.append(d)
                            await on_delta(d)
                    return "".join(parts)
        await asyncio.sleep(wait); delay *= 2
    raise last


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #

class Provider:
    name = "base"
    label = "Base"
    supports_web_search = False

    async def stream(self, *, system: str, user: str, on_delta: OnDelta,
                     max_tokens: int = 2600, kind: str | None = None, lang: str = "en") -> str:
        raise NotImplementedError

    async def complete_json(self, *, system: str, user: str, schema: dict,
                            max_tokens: int = 4000, lang: str = "en") -> dict:
        raise NotImplementedError


class AnthropicProvider(Provider):
    name, label, supports_web_search = "anthropic", "Claude (Opus 4.8)", True

    def __init__(self):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    async def stream(self, *, system, user, on_delta, max_tokens=2600, kind=None, lang="en"):
        parts: list[str] = []
        async with self.client.messages.stream(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for ev in stream:
                if ev.type == "content_block_delta" and getattr(ev.delta, "type", None) == "text_delta":
                    parts.append(ev.delta.text)
                    await on_delta(ev.delta.text)
        return "".join(parts)

    async def complete_json(self, *, system, user, schema, max_tokens=4000, lang="en"):
        resp = await self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}, "effort": "high"},
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return json.loads(text)


class _OpenAICompatProvider(Provider):
    """Shared base for OpenAI-compatible chat APIs (Groq)."""
    url = ""
    api_key = ""

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _parse(line: str):
        if not line.startswith("data: "):
            return None
        data = line[6:].strip()
        if data == "[DONE]":
            return None
        try:
            return json.loads(data)["choices"][0]["delta"].get("content")
        except Exception:
            return None

    async def stream(self, *, system, user, on_delta, max_tokens=2600, kind=None, lang="en"):
        payload = {
            "model": self.model, "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        return await _stream_retry(self.url, self._headers(), payload, self._parse, on_delta)

    async def complete_json(self, *, system, user, schema, max_tokens=4000, lang="en"):
        sys2 = system + "\n\nRespond with ONLY a JSON object matching this schema:\n" + json.dumps(schema)
        payload = {
            "model": self.model, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys2}, {"role": "user", "content": user}],
        }
        r = await _post_retry(self.url, self._headers(), payload)
        return json.loads(r.json()["choices"][0]["message"]["content"])


class GroqProvider(_OpenAICompatProvider):
    name, label = "groq", "Groq · Llama 3.3 70B"

    def __init__(self):
        self.api_key = os.environ["GROQ_API_KEY"]
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.url = "https://api.groq.com/openai/v1/chat/completions"


class GeminiProvider(Provider):
    name, label = "gemini", "Gemini 2.0 Flash"

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.base = "https://generativelanguage.googleapis.com/v1beta/models"

    @staticmethod
    def _parse(line: str):
        if not line.startswith("data: "):
            return None
        try:
            obj = json.loads(line[6:])
        except Exception:
            return None
        out = "".join(p.get("text", "")
                      for cand in obj.get("candidates", [])
                      for p in cand.get("content", {}).get("parts", []))
        return out or None

    async def stream(self, *, system, user, on_delta, max_tokens=2600, kind=None, lang="en"):
        url = f"{self.base}/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        return await _stream_retry(url, {}, payload, self._parse, on_delta)

    async def complete_json(self, *, system, user, schema, max_tokens=4000, lang="en"):
        url = f"{self.base}/{self.model}:generateContent?key={self.api_key}"
        sys2 = system + "\n\nRespond with ONLY a JSON object matching this schema:\n" + json.dumps(schema)
        payload = {
            "system_instruction": {"parts": [{"text": sys2}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "response_mime_type": "application/json"},
        }
        r = await _post_retry(url, {}, payload)
        data = r.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        return json.loads(text)


class OfflineProvider(Provider):
    """No key needed — synthesizes a coherent demo run so the site always works."""
    name, label = "offline", "Offline demo (no API key)"

    async def stream(self, *, system, user, on_delta, max_tokens=2600, kind=None, lang="en"):
        text = _offline_text(kind, _topic_of(user), lang)
        for chunk in re.findall(r"\S+\s*", text):  # stream word-by-word for the typing effect
            await on_delta(chunk)
            await asyncio.sleep(0.012)
        return text

    async def complete_json(self, *, system, user, schema, max_tokens=4000, lang="en"):
        return _offline_review(lang)


# --------------------------------------------------------------------------- #
# Offline content templates (English + Georgian)
# --------------------------------------------------------------------------- #

def _topic_of(user: str) -> str:
    m = re.search(r"Topic:\s*(.+)", user)
    return (m.group(1).strip() if m else "the subject")


def _offline_text(kind: str | None, topic: str, lang: str) -> str:
    ka = lang == "ka"
    if kind == "research":
        if ka:
            return (f"## კვლევის ჩანაწერები (offline დემო)\n"
                    f"- offline რეჟიმში ვებ-ძებნა არ არის — დაამატე უფასო Groq/Gemini გასაღები რეალური კვლევისთვის.\n"
                    f"- მთავარი კუთხე: რა არის „{topic}“ და რატომ არის მნიშვნელოვანი.\n"
                    f"- Anthropic გასაღები + კრედიტი ცოცხალ, დამოწმებულ წყაროებს ჩართავს.")
        return (f"## Research notes (offline demo)\n"
                f"- No web search in offline mode — add a free Groq/Gemini key for real model research.\n"
                f"- Key angle: what \"{topic}\" is and why it matters.\n"
                f"- An Anthropic key with credits adds live, cited sources.")
    meta = ""
    if kind == "publish":
        meta = (f"\n\n## Meta\n"
                + (f"მოკლე მიმოხილვა თემაზე „{topic}“ — offline დემო.\n\n**თეგები:** დემო, მრავალაგენტიანი, ai, კონტენტი\n"
                   if ka else
                   f"A short overview of \"{topic}\" — offline demo.\n\n**Tags:** demo, multi-agent, ai, content\n"))
    if ka:
        return (f"# {topic}\n\n"
                f"ეს არის Multi-Agent Studio-ს **offline დემო**. რეალური სტატიის დასაგენერირებლად "
                f"დაამატე უფასო API გასაღები (Groq ან Gemini) — იხ. `.env.example`.\n\n"
                f"## რატომ არის მნიშვნელოვანი\n\n„{topic}“ განსაზღვრავს, როგორ იქმნება და გამოიყენება თანამედროვე სისტემები.\n\n"
                f"## როგორ მუშაობს\n\nაგენტების კონვეიერი იკვლევს, წერს, აფასებს და ასწორებს — ხოლო შენ ამტკიცებ საბოლოო დრაფტს.\n\n"
                f"## დასკვნა\n\nჩართე პროვაიდერი და აგენტები „{topic}“-ზე რეალურ, დამოწმებულ სტატიას დაწერენ.{meta}")
    return (f"# {topic}\n\n"
            f"This is an **offline demo** of the Multi-Agent Content Studio. Add a free API key "
            f"(Groq or Gemini) to generate real posts — see `.env.example`.\n\n"
            f"## Why it matters\n\n\"{topic}\" shapes how modern systems are built and used.\n\n"
            f"## How it works\n\nA pipeline of agents researches, drafts, reviews, and revises — while you approve the final draft.\n\n"
            f"## Takeaway\n\nConnect a provider to see the agents produce a real, sourced article on \"{topic}\".{meta}")


def _offline_review(lang: str) -> dict:
    if lang == "ka":
        return {"score": 8, "verdict": "Offline დემო შეფასება — ჩართე პროვაიდერი რეალური კრიტიკისთვის",
                "strengths": ["სუფთა სტრუქტურა", "თემასთან შესაბამისი"],
                "issues": [{"severity": "low", "issue": "Offline placeholder კონტენტი",
                            "fix": "დაამატე Groq/Gemini/Anthropic გასაღები რეალური გენერაციისთვის"}]}
    return {"score": 8, "verdict": "Offline demo review — connect a provider for a real critique",
            "strengths": ["Clear structure", "On-topic"],
            "issues": [{"severity": "low", "issue": "Offline placeholder content",
                        "fix": "Add a Groq/Gemini/Anthropic key for real generation"}]}


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #

def _build(name: str) -> Provider:
    return {"anthropic": AnthropicProvider, "groq": GroqProvider,
            "gemini": GeminiProvider, "offline": OfflineProvider}[name]()


def _detect() -> Provider:
    forced = os.getenv("LLM_PROVIDER", "").lower().strip()
    if forced:
        try:
            return _build(forced)
        except Exception:
            return OfflineProvider()
    # Gemini first: its free tier has far higher token limits than Groq's, so the
    # multi-call pipeline runs without rate-limiting (and within serverless timeouts).
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiProvider()
    if os.getenv("GROQ_API_KEY"):
        return GroqProvider()
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    return OfflineProvider()


provider = _detect()
# Always-available fallback so a rate-limited free provider never hard-fails a run.
offline_provider = provider if provider.name == "offline" else OfflineProvider()
