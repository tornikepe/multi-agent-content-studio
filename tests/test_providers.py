"""Provider layer: offline generation, auto-detect, and error mapping."""

import asyncio

import httpx

from backend import config, llm


def _run(coro):
    return asyncio.run(coro)


async def _collect(provider, **kw):
    chunks = []

    async def on_delta(t):
        chunks.append(t)

    out = await provider.stream(on_delta=on_delta, **kw)
    return out, chunks


def test_offline_stream_english():
    out, chunks = _run(_collect(
        llm.OfflineProvider(), system="", user="Topic: vector databases",
        max_tokens=500, kind="write", lang="en",
    ))
    assert "vector databases" in out          # topic is woven in
    assert "".join(chunks) == out             # streamed chunks reconstruct the text
    assert out.startswith("#")                # markdown post


def test_offline_stream_georgian():
    out, _ = _run(_collect(
        llm.OfflineProvider(), system="", user="Topic: ვექტორული ბაზები",
        max_tokens=500, kind="write", lang="ka",
    ))
    assert any("Ⴀ" <= c <= "ჿ" for c in out)  # contains Georgian script


def test_offline_review_is_valid():
    data = _run(llm.OfflineProvider().complete_json(
        system="", user="Topic: x", schema=config.REVIEW_SCHEMA,
    ))
    assert 1 <= data["score"] <= 10
    assert isinstance(data["strengths"], list)
    assert isinstance(data["issues"], list)


def test_active_provider_is_offline_in_tests():
    # conftest forces LLM_PROVIDER=offline
    assert llm.provider.name == "offline"
    assert llm.offline_provider.name == "offline"


def _http_error(code):
    req = httpx.Request("POST", "http://example.test")
    return httpx.HTTPStatusError(str(code), request=req, response=httpx.Response(code, request=req))


def test_friendly_error_rate_limit():
    msg = llm.friendly_error(_http_error(429)).lower()
    assert "rate" in msg or "quota" in msg


def test_friendly_error_bad_key():
    assert "key" in llm.friendly_error(_http_error(401)).lower()


def test_is_fatal_account_error():
    assert llm.is_fatal_account_error(_http_error(401)) is True
    assert llm.is_fatal_account_error(_http_error(429)) is False
