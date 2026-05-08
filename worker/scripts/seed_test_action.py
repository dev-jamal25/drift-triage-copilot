"""Push a sample QueuedAction onto worker:queue:ready for manual testing.

Usage (from worker/):
    uv run python scripts/seed_test_action.py replay_test
    uv run python scripts/seed_test_action.py retrain
    uv run python scripts/seed_test_action.py rollback

Optional arguments:
    --target-version (default: v3)
    --model-name (default: bank-marketing-classifier)

Reads REDIS_URL from environment / .env (WorkerSettings).
Generates a fresh investigation_id each invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import UTC, datetime

from redis.asyncio import Redis
from shared.contracts import QueuedAction

from app.core.config import WorkerSettings
from app.queue.keys import READY


async def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue a sample action onto the worker queue")
    parser.add_argument(
        "action_type",
        choices=["replay_test", "retrain", "rollback"],
        help="Type of action to enqueue",
    )
    parser.add_argument(
        "--target-version",
        default="v3",
        help="Target model version (default: v3)",
    )
    parser.add_argument(
        "--model-name",
        default="bank-marketing-classifier",
        help="Model name (default: bank-marketing-classifier)",
    )
    args = parser.parse_args()

    investigation_id = str(uuid.uuid4())
    action = QueuedAction(
        idempotency_key=f"{investigation_id}:{args.action_type}:{args.target_version}",
        investigation_id=investigation_id,
        model_name=args.model_name,
        action_type=args.action_type,  # type: ignore[arg-type]
        target_version=args.target_version,
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime.now(UTC),
    )

    settings = WorkerSettings()
    client = Redis.from_url(settings.redis_url, decode_responses=False)
    try:
        payload = action.model_dump_json().encode("utf-8")
        await client.lpush(READY, payload)
        print(
            f"✓ Enqueued: idempotency_key={action.idempotency_key} "
            f"action_type={args.action_type} target_version={args.target_version}"
        )
    except Exception as exc:
        print(f"✗ Error: {exc}")
        raise
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
