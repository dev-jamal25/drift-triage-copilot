"""Platform-internal response schema for ``GET /drift``.

This wraps ``shared.contracts.DriftReport`` rather than reusing it directly
so we can decorate the response with platform-only metadata (which model
version produced it, when the scheduler last refreshed) without touching
the cross-service contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from shared.contracts import DriftReport


class DriftReportResponse(BaseModel):
    """Latest cached drift report plus serving-side provenance."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    drift: DriftReport
    model_name: str
    model_version: str
    last_refreshed_at: datetime
