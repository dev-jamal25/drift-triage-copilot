"""Investigations router: GET endpoints for investigations."""

from typing import Annotated

import structlog
from agent.app.database import get_session
from agent.app.models import HilApproval, Investigation
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

SessionDep = Annotated[AsyncSession, Depends(get_session)]

log = structlog.get_logger()

router = APIRouter()


@router.get("/investigations")
async def list_investigations(
    session: SessionDep,
) -> dict[str, list]:
    """
    Get all open and resolved investigations.

    Returns investigations with their associated HIL approvals if any.
    """
    log.info("investigations.list")

    # Fetch all investigations
    result = await session.execute(select(Investigation))
    investigations = result.scalars().all()

    # Fetch all approvals
    approval_result = await session.execute(select(HilApproval))
    approvals = {a.investigation_id: a for a in approval_result.scalars().all()}

    investigations_list = []
    for inv in investigations:
        inv_dict = {
            "investigation_id": inv.investigation_id,
            "model_name": inv.model_name,
            "model_version": inv.model_version,
            "severity": inv.severity,
            "status": inv.status,
            "created_at": inv.created_at.isoformat(),
            "resolved_at": inv.resolved_at.isoformat() if inv.resolved_at else None,
        }
        if inv.investigation_id in approvals:
            approval = approvals[inv.investigation_id]
            inv_dict["approval"] = {
                "status": approval.status,
                "recommended_action": approval.recommended_action,
                "summary": approval.summary,
                "created_at": approval.created_at.isoformat(),
            }
        investigations_list.append(inv_dict)

    return {"investigations": investigations_list}


@router.get("/investigations/{investigation_id}")
async def get_investigation(
    investigation_id: str,
    session: SessionDep,
) -> dict:
    """Get a specific investigation with details."""
    log.info("investigations.get", investigation_id=investigation_id)

    investigation = await session.get(Investigation, investigation_id)
    if not investigation:
        return {"error": "Investigation not found"}

    inv_dict = {
        "investigation_id": investigation.investigation_id,
        "model_name": investigation.model_name,
        "model_version": investigation.model_version,
        "severity": investigation.severity,
        "status": investigation.status,
        "created_at": investigation.created_at.isoformat(),
        "resolved_at": investigation.resolved_at.isoformat() if investigation.resolved_at else None,
    }

    # Fetch associated approval if any
    approval = await session.get(HilApproval, investigation_id)
    if approval:
        inv_dict["approval"] = {
            "status": approval.status,
            "recommended_action": approval.recommended_action,
            "summary": approval.summary,
            "created_at": approval.created_at.isoformat(),
        }

    return inv_dict
