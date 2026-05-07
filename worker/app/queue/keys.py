"""Redis key namespace for the worker's reliable queue.

Centralised so producers (the agent) and consumers (this worker) cannot
drift on naming.
"""

from __future__ import annotations

QUEUE_PREFIX = "worker:queue"

READY: str = f"{QUEUE_PREFIX}:ready"
PROCESSING: str = f"{QUEUE_PREFIX}:processing"
RETRY: str = f"{QUEUE_PREFIX}:retry"
DLQ: str = f"{QUEUE_PREFIX}:dlq"
