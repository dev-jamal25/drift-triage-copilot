"""Tiny mock /webhooks/drift receiver for live Step 5 smoke tests.

Logs every POST it receives so you can watch the platform's drift webhook
arrive in real time. Run from the ``platform/`` directory:

    uv run uvicorn scripts.mock_webhook_receiver:app --port 9999 --log-level warning
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from fastapi import FastAPI, Request

app = FastAPI(title="Mock drift webhook receiver")


@app.post("/webhooks/drift")
async def receive(request: Request) -> dict[str, str]:
    payload = await request.json()
    line = json.dumps(
        {
            "received_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "event_id": payload.get("event_id"),
            "model_version": payload.get("model_version"),
            "severity": payload.get("severity"),
            "previous_severity": payload.get("previous_severity"),
            "sample_size": payload.get("drift_report", {}).get("sample_size"),
            "overall_severity": payload.get("drift_report", {}).get("overall_severity"),
        }
    )
    print(line, flush=True, file=sys.stderr)
    return {"ok": "received"}
