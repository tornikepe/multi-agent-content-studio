"""Force the offline provider before anything imports the backend.

`backend.llm` selects a provider at import time. Setting this here (which pytest
imports before the test modules) guarantees the whole suite runs with zero API
keys — so CI is green without any secrets.
"""

import os

os.environ["LLM_PROVIDER"] = "offline"
