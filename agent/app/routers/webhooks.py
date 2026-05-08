"""Webhooks router: POST /webhooks/drift."""

import uuid
from datetime import datetime
from typing import Annotated

import structlog
from agent.app.database import get_session
from agent.app.graph import create_graph
from agent.app.models import Investigation
from agent.app.queue import get_queue_client
from fastapi import APIRouter, Depends, HTTPException
from shared.contracts import DriftEvent, QueuedAction
from sqlalchemy.ext.asyncio import AsyncSession

SessionDep = Annotated[AsyncSession, Depends(get_session)]

log = structlog.get_logger()

router = APIRouter()


@router.post("/webhooks/drift", status_code=202)
async def receive_drift_event(
    event: DriftEvent,
    session: SessionDep,
) -> dict[str, str]:
    """
    Receive drift event webhook from platform.

    Opens a new LangGraph investigation thread for the event.
    Returns investigation_id for tracking.
    """
    log.info("webhook.drift.received", event_id=event.event_id, model_name=event.model_name)

    # Create investigation record
    investigation_id = str(uuid.uuid4())
    thread_id = f"investigation:{investigation_id}"

    investigation = Investigation(
        investigation_id=investigation_id,
        model_name=event.model_name,
        model_version=event.model_version,
        severity=event.severity,
        status="open",
        thread_id=thread_id,
        created_at=datetime.utcnow(),
    )

    session.add(investigation)
    try:
        await session.commit()
    except Exception as e:
        log.error("webhook.drift.db_error", investigation_id=investigation_id, error=str(e))
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to create investigation") from e

    # Initialize graph and run investigation
    graph = create_graph()

    initial_state = {
        "investigation_id": investigation_id,
        "model_name": event.model_name,
        "model_version": event.model_version,
        "drift_event": event.model_dump(),
    }

    try:
        # Run the graph
        final_state = graph.invoke(initial_state)
        log.info(
            "webhook.drift.graph_complete",
            investigation_id=investigation_id,
            final_state_keys=list(final_state.keys()),
        )

        # Handle post-graph actions based on final state
        if final_state.get("action_type"):
            # Enqueue the action
            queue_client = get_queue_client()
            action_type = final_state.get("action_type")
            idempotency_key = final_state.get("idempotency_key")

            queued_action = QueuedAction(
                idempotency_key=idempotency_key,
                investigation_id=investigation_id,
                model_name=event.model_name,
                action_type=action_type,
                target_version=event.model_version,
                payload=event.model_dump(),
                attempt=0,
                max_attempts=3,
                created_at=datetime.utcnow(),
            )
            await queue_client.enqueue(queued_action)
            log.info(
                "webhook.drift.action_enqueued",
                investigation_id=investigation_id,
                action_type=action_type,
            )

    except Exception as e:
        log.error("webhook.drift.graph_error", investigation_id=investigation_id, error=str(e))
        investigation.status = "failed"
        session.add(investigation)
        await session.commit()
        raise HTTPException(status_code=500, detail="Investigation failed") from e

    log.info(
        "webhook.drift.investigation_created",
        investigation_id=investigation_id,
        event_id=event.event_id,
    )

    return {"investigation_id": investigation_id, "status": "accepted"}
