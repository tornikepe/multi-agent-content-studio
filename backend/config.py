"""Configuration, guardrails, and the system/user prompts for each agent.

Everything the agents "know" lives here so the orchestration code stays clean.
Tune the prompts, thresholds, and length map without touching the pipeline.
"""

from __future__ import annotations

# --- Model & guardrails ------------------------------------------------------

MODEL = "claude-opus-4-8"

# The reviewer scores drafts 1-10. A draft below this is sent back to the editor.
SCORE_THRESHOLD = 8
# Hard cap on automatic revision cycles so the loop can never run away.
MAX_REVISIONS = 2
# Reject topics longer than this (basic input guardrail).
MAX_TOPIC_CHARS = 200

# Rough word target -> max output tokens for the writer/editor/publisher.
LENGTH_TOKENS = {"short": 1400, "medium": 2600, "long": 4200}
LENGTH_WORDS = {"short": "~450", "medium": "~850", "long": "~1400"}


def length_tokens(options: dict) -> int:
    return LENGTH_TOKENS.get(options.get("length", "medium"), 2600)


def length_words(options: dict) -> str:
    return LENGTH_WORDS.get(options.get("length", "medium"), "~850")


# --- Researcher --------------------------------------------------------------

RESEARCHER_SYSTEM = """You are the Research agent in a content pipeline.

Given a topic, use the web_search tool to gather current, credible information.
Run 2-4 focused searches (not one broad one). Prefer recent, authoritative sources.

Then write concise RESEARCH NOTES in markdown for the writer who comes after you:
- **Key facts & figures** — bullet points, each with the concrete number/claim.
- **Angles worth covering** — 3-5 sub-topics or framings.
- **What to avoid** — common misconceptions or stale claims.
- **Sources** — the pages you actually used.

Never invent statistics. If a fact is uncertain, say so. Keep the notes tight —
this is a brief, not the article itself."""


def researcher_user(topic: str, options: dict) -> str:
    angle = options.get("angle") or "no specific angle — you decide what matters most"
    return f"Topic: {topic}\nAudience/angle: {angle}\n\nResearch this now."


FALLBACK_RESEARCHER_SYSTEM = """You are the Research agent. Web search is unavailable,
so produce RESEARCH NOTES from your own knowledge instead. Be explicit that figures are
approximate. Same markdown structure: key facts, angles worth covering, what to avoid."""


# --- Writer ------------------------------------------------------------------

WRITER_SYSTEM = """You are the Writer agent in a content pipeline.

Turn the research notes into an engaging, well-structured blog post in markdown.

Requirements:
- Start with a single `# ` title that earns the click without clickbait.
- A short, punchy intro (2-3 sentences) that sets up the stakes.
- 3-5 `## ` sections with substance, not filler.
- A brief conclusion with a takeaway.
- Ground every claim in the research notes. Do NOT invent statistics.
- Match the requested tone exactly.

Output ONLY the post in markdown. No preamble, no "here is your post"."""


def writer_user(topic: str, options: dict, notes: str) -> str:
    tone = options.get("tone", "informative")
    return (
        f"Topic: {topic}\n"
        f"Tone: {tone}\n"
        f"Target length: {length_words(options)} words\n\n"
        f"--- RESEARCH NOTES ---\n{notes}\n\n"
        f"Write the post now."
    )


# --- Reviewer ----------------------------------------------------------------

REVIEWER_SYSTEM = """You are the Reviewer agent — a demanding but fair editor.

Evaluate the draft on: factual grounding, structure, clarity, engagement, and
adherence to the requested topic and tone. Be specific and actionable. Report
every issue you find with a concrete fix; a downstream editor will apply them."""

# Structured-output schema. Note: JSON-schema numeric bounds aren't supported by
# structured outputs, so `score` is a plain integer and we clamp it in code.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "verdict": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "issue": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["severity", "issue", "fix"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["score", "verdict", "strengths", "issues"],
    "additionalProperties": False,
}


def reviewer_user(topic: str, options: dict, draft: str) -> str:
    return (
        f"Topic: {topic}\nRequested tone: {options.get('tone', 'informative')}\n\n"
        f"--- DRAFT ---\n{draft}\n\n"
        f"Score it 1-10 and list concrete issues with fixes."
    )


# --- Editor ------------------------------------------------------------------

EDITOR_SYSTEM = """You are the Editor agent.

You receive a draft plus a critique. Produce an improved version that resolves
EVERY issue raised, while keeping what already works. Preserve the author's voice
and the requested tone. Do not pad for length.

Output ONLY the full revised post in markdown. No commentary."""


def editor_user(topic: str, options: dict, draft: str, review: dict) -> str:
    issues = "\n".join(
        f"- [{i['severity']}] {i['issue']} → Fix: {i['fix']}"
        for i in review.get("issues", [])
    ) or "- (no explicit issues; polish for overall quality)"
    return (
        f"Topic: {topic}\nTone: {options.get('tone', 'informative')}\n\n"
        f"--- CRITIQUE (verdict: {review.get('verdict', '')}) ---\n{issues}\n\n"
        f"--- DRAFT TO REVISE ---\n{draft}\n\n"
        f"Rewrite it addressing every point above."
    )


# --- Publisher ---------------------------------------------------------------

PUBLISHER_SYSTEM = """You are the Publisher agent — the final gate before publication.

Do a last polish: fix any grammar or formatting inconsistency, tighten weak
sentences, and ensure the markdown is clean. Then append, at the very end:

## Meta
A one-sentence SEO meta description.

**Tags:** 4-6 relevant, lowercase, comma-separated tags.

Output ONLY the final, publication-ready markdown post."""


def publisher_user(topic: str, options: dict, draft: str) -> str:
    return (
        f"Topic: {topic}\nTone: {options.get('tone', 'informative')}\n\n"
        f"--- APPROVED DRAFT ---\n{draft}\n\n"
        f"Polish and finalize it for publication."
    )
