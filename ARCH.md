# Architecture

## System diagram

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Postgres (all state: predictions, drift, approvals, logs)  │
   │  + LangGraph checkpoints (investigation resumption)         │
   └─┬──────────────────────────────────────────────────────────┘
     │
     ├─ predictions_log  (platform)
     ├─ drift_events, investigations  (agent)
     ├─ hil_approvals, actions_log  (agent, worker, dashboard)
     └─ worker_action_jobs  (worker)

   ┌───────────────────┐         ┌───────────────────┐
   │  MLflow Registry  │         │   Redis Queue     │
   │  (model artifacts,│         │  (QueuedAction)   │
   │   metrics, runs)  │         │                   │
   └───────────────────┘         └──────────┬────────┘
                                           │
┌──────────────────────┐  webhook ┌────────▼──────────────┐   dequeue/execute
│  Model Service       │ ────────▶│  Agent (LangGraph)   │─────────────┐
│  (FastAPI)           │          │  Supervisor          │             │
│                      │          │                      │             │
│  • /predict          │◀─────────│  • triage sub-agent  │             │
│  • /drift            │ /promote │  • action sub-agent  │             │
│  • /promote          │          │  • comms sub-agent   │             │
│                      │          │                      │             │
│  Rolling drift       │          │  ↓ (on action)       │             │
│  detector (60s       │          │  enqueue QueuedAction│             │
│  window = 1000 preds)│          │  to Redis            │             │
└──────────────────────┘          └─────────────────────┘             │
                                                                       │
                                   ┌───────────────────────────────────▼┐
                                   │  Worker (Redis consumer)            │
                                   │                                     │
                                   │  Idempotent execution:              │
                                   │  • replay_test                      │
                                   │  • retrain                          │
                                   │  • rollback                         │
                                   │  (call back to model-service)       │
                                   └──────────────────────────────────┬──┘
                                                                      │
┌──────────────────────────────────────────────────────────────────────────────┐
│  Dashboard (Streamlit, read Postgres)                                        │
│  • Model registry (MLflow)                                                   │
│  • Live investigations (drift events, agent state)                           │
│  • HIL approval inbox (hil_approvals table)                                  │
│  • Worker queue & action logs                                                │
│  • Drift visualizations (features, severity trend)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Services & responsibilities

**`postgres`** — Centralized database (no sharding).
- **platform** owns `predictions_log`, drift computation state
- **agent** owns `drift_events`, `investigations`, `hil_approvals`, `langgraph_checkpoints`
- **worker** owns `worker_action_jobs`, appends to `actions_log`
- **dashboard** reads all tables, updates `hil_approvals.status` on user approval

**`redis`** — Transient queue for slow operations.
- **agent** enqueues `QueuedAction` with idempotency key
- **worker** consumes, dedupes, executes, logs outcome

**`mlflow`** — Model artifact & run metadata repository.
- **platform** registers trained models with alias (`staging`, `production`)
- **worker** fetches versioned model URI on replay/retrain/rollback

**`platform`** (model-service, port 8000)
- **HTTP-driven**, stateless except for loaded model in memory
- **POST /predict** → validate input (reject `duration`) → apply `pdays` sentinel → `predict_proba` → log to DB
- **GET /drift** → compute PSI + chi² over last 1000 predictions → POST webhook to agent if severity changes
- **POST /promote** → validate idempotency key + approval token → update MLflow alias → reload model on next drift check
- **GET /readyz** → model load status (used for k8s-style probes)

**`agent`** (LangGraph, port 8001)
- **Event-driven**, long-running, **stateful** (LangGraph checkpoints in Postgres)
- **POST /webhooks/drift** → parse `DriftEvent` → write to DB → kick off graph
- **Supervisor topology:**
  - `triage` sub-agent: LLM classifies severity → "ok", "investigate", "urgent"
  - `action` sub-agent: LLM decides remediation → "none", "replay", "retrain", "rollback"
  - `comms` sub-agent: LLM summarizes findings & action
- **if action is production-touching:**
  - Pause, write `HilApproval` row, wait for human approval
  - Resume, enqueue `QueuedAction` to Redis
- **Never executes slow work** — only enqueues; resilient to agent restart

**`worker`** (Redis consumer, background process)
- **Queue-driven**, idempotent, **dumb** executor
- Claim `QueuedAction` from Redis via idempotency key (`{investigation_id}:{action_type}:{target_version}`)
- Execute: call `model-service /promote` with approval token
- Log outcome to `actions_log`; on failure → retry with backoff → DLQ after max_retries

**`dashboard`** (Streamlit, port 8501)
- **Read-only** over Postgres; HTTP-only writes (HilApproval updates)
- Poll `hil_approvals` for pending rows → display inbox
- Show live investigations (joined with `drift_events`, `actions_log`)
- Show queue depth, failed jobs, model registry status
- Never touches MLflow or Redis directly

## Data flow narrative

1. **Prediction incoming:** Client POSTs to `platform /predict` → FastAPI validates input (extra="forbid") → applies `pdays==999` sentinel → calls fitted model → logs to `predictions_log` + response cache for drift computation
2. **Drift detection:** Background task on `platform` (every 60s) computes PSI (numeric) + chi² (categorical) over rolling window of 1000 predictions. If any feature or output severity increases → POST `DriftEvent` webhook to `agent /webhooks/drift`
3. **Agent investigates:** Agent receives webhook → writes `drift_events` row → creates `investigation_id` → spawns LangGraph supervisor:
   - `triage` runs with DriftEvent details → outputs severity classification
   - `action` runs (conditioned on triage) → outputs action decision (none/replay/retrain/rollback)
   - `comms` runs (conditioned on action) → summarizes findings
4. **HIL gate (if production-touching):** If action is non-trivial, agent writes `HilApproval` row (status=`pending`). Agent **suspends** here. Dashboard shows row in HIL inbox. Human reviews & clicks Approve → Dashboard updates row to `approved` → Agent resumes next tick
5. **Queue & execute:** Agent validates approval is not superseded, enqueues `QueuedAction` to Redis with idempotency key. Worker claims job, calls `platform /promote` with approval token, logs result to `actions_log`. Platform validates approval + token, updates MLflow alias (e.g., `staging` → `production`), reloads model on next drift computation.
6. **Dashboard reflects:** Streamlit polls `hil_approvals`, `investigations`, `actions_log`, `worker_action_jobs`. Shows investigation state, action outcome, new model version loaded.

## Infrastructure notes

- All services share one **Postgres 16** instance (no sharding). Migrations split by owner (platform/worker; agent auto-creates via SQLAlchemy).
- **Redis** is ephemeral queue only — no persistence required. Worker dedupes via Postgres idempotency key, not Redis TTL.
- **MLflow** backed by Postgres (same DB). Artifacts stored on local filesystem (`/mlflow/artifacts`).
- **Docker Compose** defines eight services: `postgres`, `redis`, `mlflow`, `platform-migrate`, `model-bootstrap`, `model-service`, `agent`, `worker`, `dashboard`.
  - `model-bootstrap` (one-shot) trains + registers initial model before `model-service` starts.
  - `worker-migrate` ensures worker tables exist before worker consumes.
  - Agent auto-creates tables via SQLAlchemy on first run.

## Boundaries & constraints

- **Model service** is HTTP-request-driven; all calls timeout & retry (tenacity).
- **Agent** is soft state per investigation; restarts resume from checkpoint without re-running slow work (already in Redis).
- **Worker** is single-process (documented scaling limitation for sprint); prevents race conditions on idempotency key dedup.
- **Dashboard** never calls platform or worker directly — reads Postgres, writes HilApproval rows only.
- **All external calls** have timeout + tenacity retry. **No silent failures.**
