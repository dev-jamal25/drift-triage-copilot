"""Platform model service.

Ensures the repo root is on ``sys.path`` so ``from shared.contracts import ...``
works for local ``uv run uvicorn`` (the Dockerfile already sets ``PYTHONPATH=/app``
for compose).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
