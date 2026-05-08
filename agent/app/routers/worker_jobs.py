"""Read-only view of the worker_action_jobs table for the dashboard.

The worker owns this table. The agent only reads it for demo visibility and
never writes to it. Keeping it read-only avoids cross-track coupling.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from agent.app.database import get_session
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

SessionDep = Annotated[AsyncSession, Depends(get_session)]

log = structlog.get_logger()

router = APIRouter()


@router.get("/worker-jobs")
async def list_worker_jobs(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Return the most recently updated worker_action_jobs rows."""
    log.info("worker_jobs.list", limit=limit)

    sql = text(
        """
        SELECT idempotency_key,
               investigation_id,
               model_name,
               action_type,
               target_version,
               status,
               attempt,
               max_attempts,
               last_error,
               started_at,
               finished_at,
               updated_at,
               created_at
        FROM worker_action_jobs
        ORDER BY updated_at DESC
        LIMIT :limit
        """
    )

    try:
        result = await session.execute(sql, {"limit": limit})
    except Exception as exc:
        log.warning("worker_jobs.unavailable", error=str(exc))
        return {"jobs": []}

    rows: list[dict[str, Any]] = []
    for row in result.mappings():
        rows.append(
            {
                "idempotency_key": row["idempotency_key"],
                "investigation_id": row["investigation_id"],
                "model_name": row["model_name"],
                "action_type": row["action_type"],
                "target_version": row["target_version"],
                "status": row["status"],
                "attempt": row["attempt"],
                "max_attempts": row["max_attempts"],
                "last_error": row["last_error"],
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
        )

    return {"jobs": rows}
