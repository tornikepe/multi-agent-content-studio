#!/usr/bin/env bash
# One-command setup + run. Creates a venv, installs deps, starts the server.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment…"
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "→ Installing dependencies…"
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  echo "⚠  No .env found. Copy .env.example to .env and add your ANTHROPIC_API_KEY."
  echo "   cp .env.example .env"
fi

echo "→ Starting server at http://localhost:8000"
exec uvicorn backend.main:app --reload --port 8000
