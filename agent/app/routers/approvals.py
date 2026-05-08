"""Approvals router: POST /approve/{investigation_id}."""

from typing import Annotated

import structlog
from agent.app.database import get_session
from agent.app.models import HilApproval
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

SessionDep = Annotated[AsyncSession, Depends(get_session)]

log = structlog.get_logger()

router = APIRouter()


@router.post("/approve/{investigation_id}", status_code=200)
async def approve_investigation(
    investigation_id: str,
    session: SessionDep,
) -> dict[str, str]:
    """
    Approve a pending HIL investigation and resume LangGraph graph.

    Checks if approval is still pending, then resumes graph from checkpoint.
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

    # Mark approval as approved
    approval.status = "approved"
    session.add(approval)

    try:
        await session.commit()
    except Exception as e:
        log.error("approval.commit_error", investigation_id=investigation_id, error=str(e))
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to approve investigation") from e

    # TODO: Resume LangGraph from checkpoint using investigation_id

    log.info("approval.approved", investigation_id=investigation_id)

    return {"investigation_id": investigation_id, "status": "approved"}
