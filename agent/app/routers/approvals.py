"""Approvals router: POST /approve/{investigation_id}."""

from datetime import UTC, datetime
from typing import Annotated

import structlog
from agent.app.database import get_session
from agent.app.models import HilApproval, Investigation
from agent.app.nodes.action import generate_idempotency_key
from agent.app.queue.client import QueueClient
from fastapi import APIRouter, Depends, HTTPException, Request
from shared.contracts import QueuedAction
from sqlalchemy.ext.asyncio import AsyncSession

SessionDep = Annotated[AsyncSession, Depends(get_session)]

log = structlog.get_logger()

router = APIRouter()


async def get_queue_client(request: Request) -> QueueClient:
    """Get the queue client from app state."""
    return request.app.state.queue_client


QueueClientDep = Annotated[QueueClient, Depends(get_queue_client)]


@router.post("/approve/{investigation_id}", status_code=200)
async def approve_investigation(
    investigation_id: str,
    session: SessionDep,
    queue_client: QueueClientDep,
) -> dict[str, str | bool]:
    """
    Approve a pending HIL investigation and enqueue the recommended action.

    Checks if approval is pending, marks it approved, then enqueues the
    action (retrain/rollback) if gated behind approval. Enqueues with the
    same idempotency_key so the worker de-dupes if retried.
    """
    log.info("approval.request", investigation_id=investigation_id)

    # Load approval record
    approval = await session.get(HilApproval, investigation_id)
    if not approval:
        log.error("approval.not_found", investigation_id=investigation_id)
        raise HTTPException(status_code=404, detail="Approval not found")

    # Check status
    if approval.superseded_by is not None:
        log.warning(
            "approval.superseded",
            investigation_id=investigation_id,
            superseded_by=approval.superseded_by,
        )
        raise HTTPException(
            status_code=409,
            detail=f"This approval has been superseded by {approval.superseded_by}",
        )

    if approval.status == "stale":
        log.warning("approval.stale", investigation_id=investigation_id)
        raise HTTPException(status_code=409, detail="This approval is stale and cannot be used")

    if approval.status != "pending":
        log.warning(
            "approval.not_pending",
            investigation_id=investigation_id,
            status=approval.status,
        )
        raise HTTPException(
            status_code=409,
            detail=f"Approval is already {approval.status}; cannot approve again",
        )

    # Load the investigation to get model_version
    investigation = await session.get(Investigation, investigation_id)
    if not investigation:
        log.error(
            "approval.investigation_not_found",
            investigation_id=investigation_id,
        )
        raise HTTPException(status_code=404, detail="Investigation not found")

    # Mark approval as approved
    approval.status = "approved"
    session.add(approval)

    try:
        await session.commit()
    except Exception as e:
        log.error("approval.commit_error", investigation_id=investigation_id, error=str(e))
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to approve investigation") from e

    # Enqueue the recommended action
    # Replay_test is enqueued immediately in the webhook; retrain/rollback are
    # enqueued here after approval.
    action_type = approval.recommended_action
    idempotency_key = generate_idempotency_key(
        investigation_id,
        action_type,
        investigation.model_version,
    )

    queued_action = QueuedAction(
        idempotency_key=idempotency_key,
        investigation_id=investigation_id,
        model_name=approval.model_name,
        action_type=action_type,
        target_version=investigation.model_version,
        payload={},
        attempt=0,
        max_attempts=3,
        created_at=datetime.now(UTC),
    )

    try:
        await queue_client.enqueue(queued_action)
        log.info(
            "approval.action_enqueued",
            investigation_id=investigation_id,
            action_type=action_type,
            idempotency_key=idempotency_key,
        )
    except Exception as e:
        log.error(
            "approval.enqueue_error",
            investigation_id=investigation_id,
            action_type=action_type,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue action after approval",
        ) from e

    log.info(
        "approval.approved",
        investigation_id=investigation_id,
        action_enqueued=True,
    )

    return {
        "investigation_id": investigation_id,
        "status": "approved",
        "action_enqueued": True,
    }
