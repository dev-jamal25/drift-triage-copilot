"""Retry policy — backoff and exhaustion decisions for Step 5.

Encapsulates ``backoff_base_seconds`` and ``retry_max_backoff_seconds``
from ``WorkerSettings`` so the loop can ask the policy for the next
delay without depending on the Settings object directly.

Backoff formula matches plan §5:

    delay = min(backoff_base_seconds * (2 ** attempt),
                retry_max_backoff_seconds)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff parameters wired from ``WorkerSettings``.

    ``attempt`` in ``delay_for_attempt`` is the *most recent* failed
    attempt, 0-indexed. So a fresh action that just failed for the
    first time has ``attempt=0`` and gets ``delay = base * 1``; the
    second failure (``attempt=1``) gets ``base * 2``; and so on, capped
    at ``retry_max_backoff_seconds``.
    """

    backoff_base_seconds: float
    retry_max_backoff_seconds: float

    def delay_for_attempt(self, attempt: int) -> float:
        return min(
            self.backoff_base_seconds * (2**attempt),
            self.retry_max_backoff_seconds,
        )
