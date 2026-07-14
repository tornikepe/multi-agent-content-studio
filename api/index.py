"""Vercel Python entrypoint — exposes the FastAPI ASGI app.

Vercel's @vercel/python runtime serves the `app` object found here. We add the
project root to sys.path so the `backend` package imports the same way it does
locally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app  # noqa: E402

__all__ = ["app"]
