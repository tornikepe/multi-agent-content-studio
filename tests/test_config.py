"""Config & guardrail invariants."""

from backend import config


def test_guardrails_are_sane():
    assert 1 <= config.SCORE_THRESHOLD <= 10
    assert config.MAX_REVISIONS >= 0
    assert config.MAX_TOPIC_CHARS > 10


def test_length_tokens_ordered():
    short = config.length_tokens({"length": "short"})
    long = config.length_tokens({"length": "long"})
    assert short < long
    # unknown length falls back to a sensible default, not an error
    assert config.length_tokens({}) > 0


def test_language_directive():
    assert config.lang_directive({"language": "en"}) == ""
    ka = config.lang_directive({"language": "ka"})
    assert "Georgian" in ka
    # sys() attaches the directive to a base prompt
    assert config.sys("BASE", {"language": "ka"}).startswith("BASE")
    assert config.sys("BASE", {"language": "ka"}) != "BASE"


def test_review_schema_shape():
    props = config.REVIEW_SCHEMA["properties"]
    assert props["score"]["type"] == "integer"
    assert set(config.REVIEW_SCHEMA["required"]) == {"score", "verdict", "strengths", "issues"}
    assert config.REVIEW_SCHEMA["additionalProperties"] is False


def test_prompt_builders_include_inputs():
    assert "my topic" in config.writer_user("my topic", {"tone": "witty"}, "notes")
    assert "the draft body" in config.reviewer_user("t", {}, "the draft body")
