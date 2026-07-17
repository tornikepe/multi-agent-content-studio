"""Request-schema validation."""

import pytest
from pydantic import ValidationError

from backend.models import GenerateRequest, PublishRequest, ReviseRequest


def test_topic_too_short_is_rejected():
    with pytest.raises(ValidationError):
        GenerateRequest(topic="ab")


def test_options_carry_language():
    r = GenerateRequest(topic="a valid topic", tone="witty", length="short", language="ka")
    opts = r.options()
    assert opts == {"tone": "witty", "length": "short", "angle": None, "language": "ka"}


def test_language_defaults_to_english():
    assert GenerateRequest(topic="a valid topic").options()["language"] == "en"


def test_revise_requires_a_draft():
    with pytest.raises(ValidationError):
        ReviseRequest(topic="a valid topic")  # missing draft


def test_publish_requires_a_draft():
    ok = PublishRequest(topic="a valid topic", draft="# Draft")
    assert ok.draft == "# Draft"
    with pytest.raises(ValidationError):
        PublishRequest(topic="a valid topic")
