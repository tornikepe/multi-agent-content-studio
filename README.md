# 🏭 Multi-Agent Content Studio

An orchestrated pipeline of specialized AI agents that turns a topic into a
publication-ready blog post — with a **human-in-the-loop** approval gate and a
live, animated view of every agent working in real time.

> **Researcher → Writer → Reviewer → Editor → Publisher**, coordinated with
> Claude (Opus 4.8) tool-use and streamed to the browser over Server-Sent Events.

---

## Why this project

Single-prompt LLM apps are last year's problem. The in-demand skill now is
**agent orchestration**: multiple specialized agents, guardrails, quality loops,
and a human checkpoint. This project demonstrates all of it end-to-end:

| Capability | Where it shows up |
|---|---|
| **Multi-agent orchestration** | `backend/orchestrator.py` chains five agents with a bounded revision loop |
| **Pluggable LLM providers** | `backend/llm.py` — Claude, or free **Groq / Gemini**, or a zero-key **offline** mode; auto-detected |
| **Server-side tool use** | On Claude, the Researcher uses the `web_search` tool for live, cited research |
| **Structured outputs** | The Reviewer returns a strict JSON verdict (score + issues + fixes) |
| **Streaming (SSE)** | Every token is streamed to the UI; you watch each agent "type" |
| **Guardrails** | Score threshold, capped revision cycles, input validation |
| **Human-in-the-loop** | The pipeline suspends at an approval gate until you approve or request changes |
| **Bilingual** | Full English / ქართული UI, plus content generation in either language |

---

## The pipeline

```
   ┌─────────────┐   ┌──────────┐   ┌──────────────┐
   │ 🔎 Researcher│──▶│ ✍️ Writer │──▶│ 🧐 Reviewer  │
   │ web search  │   │ drafts   │   │ scores 1-10  │
   └─────────────┘   └──────────┘   └──────┬───────┘
                                           │ score < 8 ?
                          ┌────────────────┘
                          ▼
                    ┌──────────┐        ┌───────────────┐        ┌──────────────┐
                    │ ✂️ Editor │──loop─▶│ 🧐 re-review  │──ok──▶│ ✋ Human gate │
                    └──────────┘        └───────────────┘        └──────┬───────┘
                          ▲  reject + feedback                          │ approve
                          └─────────────────────────────────────┐      ▼
                                                                 │ ┌──────────────┐
                                                                 └─│ 🚀 Publisher │
                                                                   └──────────────┘
```

1. **Researcher** — runs 2-4 web searches, produces cited research notes.
2. **Writer** — drafts the post from the notes in the requested tone & length.
3. **Reviewer** — scores the draft 1-10 and lists concrete, fixable issues.
4. **Editor** — revises against the critique. Loops until the score clears the
   threshold or the revision cap is hit.
5. **Human gate** — you approve, or request changes (which routes back to the Editor).
6. **Publisher** — final polish, meta description, and tags.

---

## Quick start

```bash
# 1. Run it — no key needed (offline demo mode works out of the box)
./run.sh            # creates a venv, installs deps, starts the server
```

Open **http://localhost:8000**, enter a topic, and hit **Run pipeline**.

### Real generation — add ONE free key

Offline mode produces placeholder content. For real posts, add a **free** key:

```bash
cp .env.example .env
```

| Provider | Free? | Get a key | Notes |
|---|---|---|---|
| **Groq** | ✅ free, no card | [console.groq.com/keys](https://console.groq.com/keys) | Fast (Llama 3.3 70B) — recommended |
| **Gemini** | ✅ free | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Great for Georgian |
| **Claude** | paid | [console.anthropic.com](https://console.anthropic.com/) | Best quality + live web search |

Paste it into `.env` and restart. Free providers are auto-detected first, so
adding a Groq key "just works". The UI shows which provider is active.

> **Language:** toggle the UI between **English / ქართული** (top-right), and pick
> the **Content language** so the agents write the post in English or Georgian.

<details>
<summary>Manual setup (without run.sh)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # or put it in .env
uvicorn backend.main:app --reload --port 8000
```
</details>

---

## How it works

The app is **stateless** — each phase is one HTTP request that streams its own
Server-Sent Events, and the browser holds the draft between phases. There's no
server-side session, so it runs identically on an always-on host or on
serverless (Vercel).

| Endpoint | Streams |
|---|---|
| `POST /api/generate` | research → write → review → (bounded edit loop) → `awaiting_approval` |
| `POST /api/revise` | edit against your feedback → re-review → `awaiting_approval` |
| `POST /api/publish` | final polish → `done` |

The human gate works without server state: when `generate` ends it streams the
draft back; the UI shows **Approve** / **Request changes**; approving POSTs the
draft to `/api/publish`, rejecting POSTs it (plus feedback) to `/api/revise`,
which loops back to the gate. Each request runs the pipeline task and drains its
event queue inside a single response — serverless-safe.

```
api/
  index.py         Vercel entrypoint (exposes the ASGI app)
backend/
  main.py          FastAPI routes; turns each phase into an SSE stream
  orchestrator.py  the three stateless phases: generate / revise / publish
  agents.py        the five agents (web search, streaming, structured output)
  llm.py           async Anthropic client, streaming helper, friendly errors
  models.py        request schemas + the per-request event Emitter
  config.py        model, guardrails, and every agent's prompt
frontend/
  index.html       self-contained animated UI (no build step, no CDN)
```

## Tech stack

**Python · FastAPI · asyncio · Server-Sent Events · Anthropic Claude (Opus 4.8)**
· web-search tool use · structured outputs · a self-contained animated frontend
(cohesive design system, SVG line-icons, light/dark theme, fully responsive —
no build step, no CDN).

## Deploy

**Vercel** (config included — `vercel.json` + `api/index.py`):

```bash
vercel --prod
```

Then add `ANTHROPIC_API_KEY` in **Project → Settings → Environment Variables**.

> ⚠️ Serverless functions have a max duration (60s on Vercel Hobby). A full
> multi-agent generation with web search can approach that, so for heavy use run
> it locally or on an always-on host (**Railway / Render / Fly** — works as-is,
> no changes needed). The live Vercel URL is perfect for showing the UI and
> short-length runs.

## Configuration

Tune everything in `backend/config.py`: the model, the `SCORE_THRESHOLD`,
`MAX_REVISIONS`, the length map, and each agent's system prompt.

---

Built as a portfolio piece demonstrating production-shaped agent orchestration.
