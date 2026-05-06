# Architecture

## System diagram

```
                        ┌─────────────────────────┐
                        │  Streamlit Dashboard    │
                        │  (registry, HIL inbox,  │
                        │   queue depth, DLQ)     │
                        └──────────┬──────────────┘
                                   │ reads
                                   ▼
   ┌──────────────┐   webhook    ┌──────────────┐   queue   ┌──────────────┐
   │ Model Service├─────────────▶│ Triage Agent │──────────▶│ Redis Queue  │
   │ (FastAPI)    │              │ (LangGraph   │           │ + Worker     │
   │              │◀─────────────┤  Supervisor) │           │ + DLQ        │
   │  /predict    │   /promote   │              │           └──────┬───────┘
   │  /drift      │              │  triage ─┐   │                  │
   │  /promote    │              │  action ─┼──▶│ tools            │ executes
   │  (gated)     │              │  comms  ─┘   │                  │ replay_test
   └──────┬───────┘              └──────┬───────┘                  │ retrain
          │                             │                          │ rollback
          │ MLflow                      │ checkpoints              │
          ▼                             ▼                          ▼
   ┌──────────────┐              ┌──────────────┐          (calls back to
   │   MLflow +   │              │  Postgres    │           Model Service
   │   Postgres   │              │ (langgraph_  │           /promote)
   │   backend    │              │  checkpoints,│
   └──────────────┘              │  app data)   │
                                 └──────────────┘
```

Seven docker-compose services: `postgres`, `redis`, `mlflow`, `model-service`, `agent`, `worker`, `dashboard`. One Postgres instance backs MLflow, LangGraph checkpoints, and application tables.

## Data flow

**Prediction request.** Client → `model-service /predict` → Pydantic-validated input (`platform/app/schemas/prediction.py`, `extra="forbid"` so `duration` is rejected at the edge) → `pdays==999` sentinel transformation via `apply_pdays_sentinel` → fitted pipeline in `app.state.model_bundle` → `predict_proba` dispatched on a worker thread → response written, `predictions_log` row appended. The model is loaded from MLflow once during the FastAPI lifespan (`models:/<name>@<alias>`); the threshold travels with it as a run param. `/healthz` reports liveness; `/readyz` reports model-readiness. Latency target: <100ms.

**Drift event.** Background task on `model-service` recomputes drift over the last 1000 predictions every 60s. On severity change → POST `DriftEvent` to agent's `/webhooks/drift` (timeout=5s, tenacity retry x3). Agent writes the event to `drift_events`, creates an `investigation_id`, kicks off the LangGraph supervisor on a fresh thread. Supervisor runs triage → action → (if Production-touching) HIL pause → comms.

**HIL approval.** Action sub-agent writes a row to `hil_approvals` with status=`pending`. Dashboard polls this table, surfaces pending rows in HIL inbox. Human clicks Approve → dashboard updates row to `approved` with `resolved_by` set. Agent's interrupted graph resumes on next tick, validates the approval is not superseded, and enqueues a `QueuedAction` to Redis. Worker dedupes via idempotency key, executes by calling `model-service /promote` with `approval_token`, writes result to `actions_log`. Dashboard reflects the outcome.

## Why these boundaries

**`platform` (model service)** is HTTP-driven, request/response, fast. Owns the model artifacts, the registry, the drift computation. Stateless across requests except for the loaded model in `app.state`.

**`agent`** is event-driven, long-running, stateful per investigation via LangGraph checkpoints. Decides *what to do*. Never executes slow operations itself — only enqueues them.

**`worker`** is queue-driven, idempotent, dumb. *Executes* the slow operations (replay test, retrain, rollback) by calling back into the platform's HTTP API. Separating it from the agent is what makes "kill the agent mid-investigation, restart, resume from checkpoint" actually work — the slow work is already in the queue or already running, independent of the agent's liveness.

**`dashboard`** is read-only over the database plus HTTP-only writes to the agent (HIL approvals via `hil_approvals` table updates). Never talks to MLflow or the queue directly — those are owned by their respective services.
