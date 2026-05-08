# Runbook

Three operational scenarios for the drift-triage copilot. The dashboard at http://localhost:8501
exposes the same scenarios in expandable panels; this file is the authoritative copy.

## Scenario 1: Trigger a drift alert in the live demo

**Goal:** Send prediction requests with shifted distribution, observe drift detection, approve remediation, and watch the action execute.

**Steps:**

1. **Send drifted predictions** to trigger the drift detector:
   - Open http://localhost:8000/docs (Platform FastAPI)
   - POST `/predict` with a batch of requests where key features deviate from training distribution
   - E.g., for Bank Marketing, shift `age` much higher, or invert `month` values
   - Alternatively, use a scripted batch (`scripts/drift_payload.py` if available)

2. **Wait for drift scheduler or force-check:**
   - The platform recomputes drift every 60 seconds
   - Or manually trigger: GET http://localhost:8000/drift
   - Look for response: drift metrics (PSI, chi²) for each feature

3. **Observe in dashboard** (http://localhost:8501):
   - **Latest Drift** card should turn red (severity > 0.25 PSI or p<0.001 chi²)
   - **Top Drifted Features** table shows which columns shifted most
   - **Investigations** table shows a new row with `severity=red`, `model_name=bank-marketing-classifier`, `status=pending`

4. **Monitor agent investigation:**
   - Expect `status` to move from `pending` → `triage_done` → `action_decided`
   - If action requires HIL approval (most actions do), status pauses at `awaiting_approval`
   - **HIL Approval Panel** shows a pending approval row with recommended action (e.g., `retrain`)

5. **Approve the action:**
   - Click **Approve** button in HIL Approval Panel
   - Row status changes to `approved_at` with timestamp

6. **Watch worker execute:**
   - Expect the Agent to enqueue `QueuedAction` to Redis
   - **Worker Queue / Action Jobs** table shows a new row with `action_type` (e.g., `retrain`), `status=running`
   - Within 30–60s, status should flip to `status=succeeded`
   - Once succeeded, Agent sends final comms message (logged in **Comms Log**)

7. **Verify in MLflow:**
   - http://localhost:5000 → Experiments → `bank-marketing-platform`
   - New run appears (if retrain was the action) with updated metrics

**What to look for:** No errors in logs, all statuses flow `pending` → `approved` → `succeeded`.

---

## Scenario 2: Recover from a stuck investigation (Redis resilience demo)

**Goal:** Show that the agent can be restarted mid-investigation without losing state, and the worker queue is durable.

**Steps:**

1. **Trigger a drift as in Scenario 1**, but let it reach `awaiting_approval` status.
   - Note the `investigation_id` (e.g., `inv_abc123def456`)

2. **Simulate worker failure: stop the worker** to demonstrate queue resilience:
   - From Docker Desktop or CLI: `docker compose stop worker`
   - Alternatively, kill the worker process in your terminal if running locally

3. **Verify the action is queued but not executing:**
   - **Worker Queue / Action Jobs** shows the action row with `status=running` (or `pending` if not yet claimed)
   - Check Redis queue depth (optional): `docker compose exec redis redis-cli LLEN drift_queue`
   - Action is *in* Redis, waiting to be processed

4. **Restart the worker:**
   - `docker compose start worker`
   - Or restart from Docker Desktop, or press Ctrl+C and `uv run python -m app.main` in the worker terminal

5. **Observe automatic retry & recovery:**
   - Worker reconnects to Redis
   - Claims the queued action by idempotency key (deduplicates automatically)
   - Executes the action (calls `model-service /promote`)
   - Status flips to `succeeded`
   - Check logs: `docker compose logs -f worker | grep idempotency_key` to see dedup in action

6. **Verify dashboard reflects the outcome:**
   - **Worker Queue / Action Jobs** shows `status=succeeded`
   - **Investigations** row shows `status=completed`
   - No duplicate jobs created on restart (idempotency key prevented replay)

**What to observe:** Queue survived worker downtime, no data loss, automatic dedup on retry.

---

## Scenario 3: Manual emergency promotion (operator bypass)

**Goal:** Show how an operator can manually promote a model without going through the agent, for true emergencies.

**⚠️ Emergency only.** Bypasses the agent investigation & HIL approval workflow. Requires explicit operator approval and Postgres idempotency key validation.

**When to use:**
- Agent is broken/stuck and you need immediate action
- A candidate model is already validated offline and needs to go live now
- Never use for routine remediation (use Scenario 1–2 instead)

**Steps:**

1. **Verify the candidate model exists in MLflow:**
   - http://localhost:5000 → Experiments → `bank-marketing-platform`
   - Pick a run with good metrics, note its version number (e.g., v42)

2. **Manually promote via MLflow CLI or UI:**
   - **Via UI:** Click the run, click "Register Model" (if not already), then set alias to `production`
   - **Via CLI:** `mlflow models alias set -m bank-marketing-classifier -a production -v <version>`

3. **Restart the model-service to load the new alias:**
   - `docker compose restart model-service`
   - Or: `GET http://localhost:8000/readyz` will show old alias; after restart, shows new alias

4. **Verify new model is loaded:**
   - http://localhost:8000/docs → GET `/readyz` → response shows `model_alias: production`
   - Predictions now use the new model

5. **Audit the decision:**
   - Append a line to this runbook with timestamp, approver, and version number
   - E.g., "2026-05-09 14:30 UTC — Jamal approved emergency promo of v42 (trained offline)"
   - Post a note in a shared channel (Slack, etc.) documenting why

**Safety gates:**
- **Platform validates** the alias exists in MLflow before loading
- **Idempotency key** in `worker_action_jobs` prevents double-execution if an agent action races this
- **No one-click button** in the dashboard — manual CLI or MLflow UI prevents accidental clicks
- **Always log the decision** — auditing is required for emergency flow

**What to avoid:**
- Do NOT bypass `docker compose restart model-service`; model must reload the new alias
- Do NOT promote to `production` if you haven't tested the model offline
- Do NOT use this for routine actions — that's what the agent's approval workflow is for
