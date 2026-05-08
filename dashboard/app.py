"""Drift Triage Co-Pilot — demo dashboard.

A read-only Streamlit dashboard that makes the full pipeline visible during
a live demo:

    Platform drift -> Agent investigation -> HIL approval ->
    Redis queue / worker action -> Final worker job status

Calls existing service APIs only:
    platform: GET /healthz, /readyz, /drift
    agent:    GET /healthz, /investigations, /worker-jobs
              POST /approve/{investigation_id}
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

PLATFORM_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8000").rstrip("/")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8001").rstrip("/")
HTTP_TIMEOUT = 5.0

st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling — small, demo-readable on a projector.
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp { background-color: #0e1117; }
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #0e1117;
        }
        .badge-green  { background-color: #2ecc71; }
        .badge-yellow { background-color: #f1c40f; }
        .badge-red    { background-color: #e74c3c; color: #fff; }
        .badge-grey   { background-color: #7f8c8d; color: #fff; }
        .card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .metric-label { color: #8b949e; font-size: 0.8rem; }
        .metric-value { font-size: 1.4rem; font-weight: 600; color: #e6edf3; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# HTTP helpers — server-side calls only; CORS not needed.
# ----------------------------------------------------------------------------


def _get(url: str) -> tuple[int, Any]:
    try:
        r = httpx.get(url, timeout=HTTP_TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def _post(url: str) -> tuple[int, Any]:
    try:
        r = httpx.post(url, timeout=HTTP_TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def _badge(status: str) -> str:
    s = (status or "").lower()
    if s in {"ok", "ready", "succeeded", "approved", "green", "healthy"}:
        cls = "badge-green"
    elif s in {"pending", "running", "open", "retry_scheduled", "yellow"}:
        cls = "badge-yellow"
    elif s in {
        "red",
        "failed",
        "dead_lettered",
        "blocked",
        "denied",
        "stale",
        "superseded",
        "not_ready",
    }:
        cls = "badge-red"
    else:
        cls = "badge-grey"
    return f'<span class="badge {cls}">{status or "unknown"}</span>'


def _fmt_dt(value: str | None) -> str:
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


# ----------------------------------------------------------------------------
# Header + controls
# ----------------------------------------------------------------------------

st.title("📡 Drift Triage Co-Pilot")

ctrl1, ctrl2, _ = st.columns([1, 1, 6])
with ctrl1:
    if st.button(":arrows_counterclockwise: Refresh"):
        st.rerun()
with ctrl2:
    auto = st.toggle("Auto-refresh 5s", value=False)

if auto:
    import time

    time.sleep(5)
    st.rerun()


# ----------------------------------------------------------------------------
# 1. System Status
# ----------------------------------------------------------------------------

st.header("1. System Status")

platform_ready_code, platform_ready = _get(f"{PLATFORM_URL}/readyz")
platform_health_code, _ = _get(f"{PLATFORM_URL}/healthz")
agent_health_code, _ = _get(f"{AGENT_URL}/healthz")
worker_jobs_code, _ = _get(f"{AGENT_URL}/worker-jobs?limit=1")

c1, c2, c3, c4 = st.columns(4)


def _status_card(col, title: str, ok: bool, detail: str) -> None:
    state = "ready" if ok else "down"
    col.markdown(
        f'<div class="card"><div class="metric-label">{title}</div>'
        f'<div class="metric-value">{_badge(state)}</div>',
        unsafe_allow_html=True,
    )


platform_ok = (
    platform_ready_code == 200
    and isinstance(platform_ready, dict)
    and (platform_ready.get("status") == "ready")
)
platform_detail = (
    f"model: {platform_ready.get('model_version', '?')}"
    if isinstance(platform_ready, dict) and platform_ready.get("status") == "ready"
    else "model not loaded"
    if platform_health_code == 200
    else "unreachable"
)
_status_card(c1, "Platform / model-service", platform_ok, platform_detail)

agent_ok = agent_health_code == 200
_status_card(c2, "Agent (LangGraph)", agent_ok, "/healthz" if agent_ok else "unreachable")

worker_via_db_ok = worker_jobs_code == 200
_status_card(
    c3,
    "Worker (via DB)",
    worker_via_db_ok,
    "worker_action_jobs reachable" if worker_via_db_ok else "table unreachable",
)

services_ok = platform_ok and agent_ok and worker_via_db_ok
_status_card(
    c4,
    "Redis / Postgres",
    services_ok,
    "through service checks" if services_ok else "see service cards",
)


# ----------------------------------------------------------------------------
# 2. Latest Drift
# ----------------------------------------------------------------------------

st.header("2. Latest Drift")

drift_code, drift_data = _get(f"{PLATFORM_URL}/drift")
if drift_code != 200 or not isinstance(drift_data, dict):
    st.info("No drift report yet — call `/predict` until the scheduler emits one.")
else:
    report = drift_data.get("drift", {})
    severity = report.get("overall_severity", "unknown")
    sample = report.get("sample_size", 0)
    output_psi = report.get("output_drift_psi", 0.0)

    a, b, c, d = st.columns(4)
    a.markdown(
        f'<div class="card"><div class="metric-label">Overall severity</div>'
        f'<div class="metric-value">{_badge(severity)}</div></div>',
        unsafe_allow_html=True,
    )
    b.markdown(
        f'<div class="card"><div class="metric-label">Sample size</div>'
        f'<div class="metric-value">{sample}</div></div>',
        unsafe_allow_html=True,
    )
    c.markdown(
        f'<div class="card"><div class="metric-label">Output drift PSI</div>'
        f'<div class="metric-value">{output_psi:.4f}</div></div>',
        unsafe_allow_html=True,
    )
    d.markdown(
        f'<div class="card"><div class="metric-label">Model</div>'
        f'<div class="metric-value">{drift_data.get("model_name", "?")}'
        f'<span class="metric-label"> v{drift_data.get("model_version", "?")}</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    feature_drifts = report.get("feature_drifts", [])
    if feature_drifts:
        df = pd.DataFrame(feature_drifts)
        df = df.sort_values("value", ascending=False).head(8)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No per-feature drift rows in this report.")


# ----------------------------------------------------------------------------
# 3. Investigations
# ----------------------------------------------------------------------------

st.header("3. Investigations")

inv_code, inv_data = _get(f"{AGENT_URL}/investigations")
investigations = inv_data.get("investigations", []) if isinstance(inv_data, dict) else []

if inv_code != 200:
    st.warning("Agent /investigations is not reachable.")
elif not investigations:
    st.info("No investigations yet. Send drifted predictions to the platform.")
else:
    rows = []
    for inv in investigations:
        approval = inv.get("approval") or {}
        rows.append(
            {
                "investigation_id": inv.get("investigation_id"),
                "model_name": inv.get("model_name"),
                "model_version": inv.get("model_version"),
                "severity": inv.get("severity"),
                "status": inv.get("status"),
                "approval_status": approval.get("status", "-"),
                "recommended_action": approval.get("recommended_action", "-"),
                "summary": (approval.get("summary") or "")[:120],
                "created_at": _fmt_dt(inv.get("created_at")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# 4. HIL Approval Panel
# ----------------------------------------------------------------------------

st.header("4. HIL Approval Panel")

pending = [inv for inv in investigations if (inv.get("approval") or {}).get("status") == "pending"]

if not pending:
    st.info("No approvals waiting. Pending investigations will appear here.")
else:
    for inv in pending:
        approval = inv["approval"]
        inv_id = inv["investigation_id"]
        with st.container():
            st.markdown(
                f'<div class="card">'
                f"<b>{inv_id}</b> &nbsp; {_badge(inv.get('severity', '?'))} "
                f"&nbsp; action: <code>{approval.get('recommended_action', '?')}</code>"
                f'<br/><span class="metric-label">{approval.get("summary", "")}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns([1, 5])
            if cols[0].button("Approve", key=f"approve-{inv_id}"):
                code, body = _post(f"{AGENT_URL}/approve/{inv_id}")
                if code == 200:
                    st.success(f"Approved {inv_id}.")
                else:
                    st.error(f"Approve failed ({code}): {body}")
                st.rerun()


# ----------------------------------------------------------------------------
# 5. Worker Queue / Action Jobs
# ----------------------------------------------------------------------------

st.header("5. Worker Queue / Action Jobs")

jobs_code, jobs_data = _get(f"{AGENT_URL}/worker-jobs?limit=25")
jobs = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []

if jobs_code != 200:
    st.warning("/worker-jobs is not reachable on the agent.")
elif not jobs:
    st.info("No worker jobs yet. Approve an investigation to enqueue one.")
else:
    df = pd.DataFrame(jobs)
    display_cols = [
        "idempotency_key",
        "investigation_id",
        "action_type",
        "target_version",
        "status",
        "attempt",
        "max_attempts",
        "updated_at",
        "last_error",
    ]
    df = df[[c for c in display_cols if c in df.columns]]
    if "updated_at" in df.columns:
        df["updated_at"] = df["updated_at"].map(_fmt_dt)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# 6. Demo Runbook
# ----------------------------------------------------------------------------

st.header("6. Runbook")

with st.expander("Scenario 1 — Trigger drift in live demo", expanded=False):
    st.markdown(
        """
1. Send prediction requests to platform `/predict` using drifted input.
2. Wait for the platform drift scheduler, or call `/drift` directly.
3. Confirm **Latest Drift** turns red.
4. Confirm a new investigation appears in **Investigations**.
5. Confirm a row appears in **HIL Approval Panel**.
6. Click **Approve**.
7. Confirm **Worker Queue** shows `replay_test` / `retrain` -> `succeeded`.
"""
    )

with st.expander("Scenario 2 — Recover from a stuck investigation", expanded=False):
    st.markdown(
        """
1. Look up the investigation row above and note the `investigation_id`.
2. Cross-check **Worker Queue** rows for the same `investigation_id`.
3. If a job is stuck in `running`, check Redis queue counts:
   `docker compose exec redis redis-cli LLEN <queue>`
4. If the worker is wedged, restart it: `docker compose restart worker`.
5. Re-enqueue only if the idempotency key is safe — duplicates are de-duped
   by the worker's `ON CONFLICT DO NOTHING` insert.
6. Confirm status flips to `succeeded` / `dead_lettered` in this dashboard.
"""
    )
