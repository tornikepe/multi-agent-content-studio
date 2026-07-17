"""End-to-end pipeline tests through the FastAPI app (offline provider)."""

import json

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _frames(text: str):
    return [json.loads(l[6:]) for l in text.split("\n") if l.startswith("data: ")]


def _types(text: str):
    return [f["type"] for f in _frames(text)]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_config_reports_offline():
    body = client.get("/api/config").json()
    assert body["provider"] == "offline"
    assert body["web_search"] is False


def test_index_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Peit" in r.text


def test_generate_runs_to_the_gate():
    r = client.post("/api/generate", json={"topic": "multi-agent AI systems", "length": "short"})
    assert r.status_code == 200
    types = _types(r.text)
    for expected in ("stage_start", "stage_output", "review", "awaiting_approval"):
        assert expected in types, f"missing {expected} in {types}"
    gate = next(f for f in _frames(r.text) if f["type"] == "awaiting_approval")
    assert gate["draft"].strip()
    assert 1 <= gate["review"]["score"] <= 10


def test_generate_rejects_short_topic():
    assert client.post("/api/generate", json={"topic": "ab"}).status_code == 422


def test_publish_produces_a_final_post():
    r = client.post("/api/publish", json={"topic": "a topic", "draft": "# Draft\n\nBody."})
    frames = _frames(r.text)
    assert "done" in [f["type"] for f in frames]
    done = next(f for f in frames if f["type"] == "done")
    assert done["final"].strip()


def test_revise_returns_to_the_gate():
    r = client.post("/api/revise", json={
        "topic": "a topic", "draft": "# Draft\n\nBody.", "feedback": "add an example",
    })
    assert "awaiting_approval" in _types(r.text)


def test_generate_in_georgian():
    r = client.post("/api/generate", json={"topic": "ხელოვნური ინტელექტი", "length": "short", "language": "ka"})
    gate = next(f for f in _frames(r.text) if f["type"] == "awaiting_approval")
    assert any("Ⴀ" <= c <= "ჿ" for c in gate["draft"])
