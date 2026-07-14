"""Pipeline phases — stateless, one per HTTP request.

    generate:  research -> write -> review -> (bounded edit loop) -> awaiting_approval
    revise:    edit against human feedback -> re-review -> awaiting_approval
    publish:   final polish -> done

The client holds the draft between phases (it's echoed back on `awaiting_approval`
and sent to `revise`/`publish`), so nothing is stored server-side. That's what
keeps the human-in-the-loop gate working on stateless/serverless hosts.
"""

from __future__ import annotations

from . import agents, config
from .models import Emitter


async def generate(em: Emitter, topic: str, options: dict) -> None:
    notes, sources = await agents.run_researcher(em, topic, options)
    draft = await agents.run_writer(em, topic, options, notes)
    review = await agents.run_reviewer(em, topic, options, draft, cycle=0)

    # Automatic quality loop, bounded by MAX_REVISIONS.
    cycle = 0
    while review["score"] < config.SCORE_THRESHOLD and cycle < config.MAX_REVISIONS:
        cycle += 1
        draft = await agents.run_editor(em, topic, options, draft, review, cycle)
        review = await agents.run_reviewer(em, topic, options, draft, cycle=cycle)

    await em.emit("awaiting_approval", draft=draft, review=review, sources=sources)


async def revise(em: Emitter, topic: str, options: dict, draft: str, feedback: str | None) -> None:
    fb = (feedback or "").strip() or "Improve overall quality and polish."
    pseudo_review = {
        "score": 0,
        "verdict": "Human reviewer requested changes",
        "strengths": [],
        "issues": [{"severity": "high", "issue": fb, "fix": fb}],
    }
    draft = await agents.run_editor(em, topic, options, draft, pseudo_review, cycle=1)
    review = await agents.run_reviewer(em, topic, options, draft, cycle=1)
    await em.emit("awaiting_approval", draft=draft, review=review)


async def publish(em: Emitter, topic: str, options: dict, draft: str) -> None:
    final = await agents.run_publisher(em, topic, options, draft)
    await em.emit("done", final=final)
