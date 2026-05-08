# Drift Triage Co-Pilot

**Week 5 AIE Bootcamp project** — a self-healing MLOps stack with human-in-the-loop (HIL) approval. A binary classifier trained on UCI Bank Marketing data is served via FastAPI with rolling drift detection (PSI + chi²). A LangGraph supervisor agent consumes drift alerts, decides remediation, and dispatches actions (replay, retrain, rollback) through Redis to a worker. All state persists across restarts, and the agent pauses for human approval before any production change.

**Authors:** Jamal Hamd, Hadi Kanaan
**Model:** `bank-marketing-classifier@staging`
**Framework:** FastAPI + LangGraph + SQLAlchemy + Redis + MLflow

## Architecture

**Data flow:** Platform detects drift → webhook to Agent → LangGraph triage/action/comms → enqueue to Redis → Worker executes → dashboard shows state

See [ARCH.md](./ARCH.md) for the full system diagram and service boundaries.

## Quick start

**Prerequisites:** Docker, Docker Compose (or `uv` + Python 3.12 for local dev).

```bash
git clone <repo>
cd drift-triage-copilot
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required), customize others if desired
docker compose up --build
```

**When everything is ready:**
- Model service: http://localhost:8000/docs (FastAPI interactive docs)
- Agent: http://localhost:8001/docs
- Dashboard: http://localhost:8501 (Streamlit)
- MLflow: http://localhost:5000 (model registry)

On first run, `model-bootstrap` auto-trains and registers the model. This may take 1–2 minutes.

## Environment setup

Copy `.env.example` to `.env` and fill in required secrets:
- `ANTHROPIC_API_KEY` — required for LLM inference (triage, action, comms)
- `LANGSMITH_API_KEY` — optional; for observability

All other variables have sensible defaults (localhost for local dev, compose hostnames for Docker).

For a detailed breakdown, see [.env.example](./.env.example).

## Metadata

| Component | Value |
|-----------|-------|
| **Dataset** | UCI Bank Marketing (45,211 rows) |
| **Target** | Subscription binary (15.64% positive) |
| **Features** | 20 (14 numeric + 6 categorical after `pdays` sentinel + unknown preservation) |
| **Model family** | Logistic Regression (best by 5-fold CV F1) |
| **Test AUC** | 0.801 |
| **Test F1** | 0.586 |
| **Operating threshold** | 0.38 (maximizes precision for recall ≥ 0.75) |
| **Drift thresholds** | PSI: <0.1 green, 0.1–0.25 yellow, >0.25 red; Chi² p<0.01 yellow, <0.001 red |
| **LLM models** | Haiku 4.5 (triage + comms), Sonnet 4.6 (action) |
| **Fallback mode** | Deterministic action selection if LLM unavailable |

## Local dev (no Docker)

Each service is a standalone `uv` project. From the repo root:

```bash
# Platform (model service + drift detection)
cd platform
uv sync --all-extras
uv run pytest -q                    # tests
uv run ruff check . && uv run ruff format --check .  # lint
uv run python -m ml.train --data data/raw/bank-additional-full.csv --out data/processed/
uv run uvicorn app.main:app --reload   # http://localhost:8000/docs

# Agent
cd ../agent
uv sync --all-extras
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run uvicorn app.main:app --reload   # http://localhost:8001/docs

# Worker
cd ../worker
uv sync --all-extras
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python -m app.main           # consume Redis queue

# Dashboard
cd ../dashboard
uv sync --all-extras
uv run streamlit run app.py         # http://localhost:8501
```

The UCI Bank Marketing dataset is not included (`.gitignore`). Download from
https://archive.ics.uci.edu/dataset/222/bank+marketing and save to `platform/data/raw/bank-additional-full.csv`.

## Useful links

- **API / Interactive docs**
  - Platform: http://localhost:8000/docs (predictions, drift, promotion)
  - Agent: http://localhost:8001/docs (webhooks, approval callbacks)
- **Health checks**
  - Platform liveness: http://localhost:8000/healthz
  - Platform model readiness: http://localhost:8000/readyz
- **UI & Observability**
  - Dashboard: http://localhost:8501 (investigation state, HIL inbox, queue depth)
  - MLflow: http://localhost:5000 (model registry, run artifacts, metrics)

## Documentation

- [ARCH.md](./ARCH.md) — system diagram, service responsibilities, data flow, infrastructure
- [DECISIONS.md](./DECISIONS.md) — design choices (model selection, thresholds, LLM tiers, drift bins, etc.)
- [RUNBOOK.md](./RUNBOOK.md) — three operational scenarios: trigger drift, recover from stuck investigation, manual promotion
- [platform/README.md](./platform/README.md) — dataset prep, model bake-off, threshold rule, MLflow flow

## Project structure

```
platform/      FastAPI model service: /predict, /drift, /promote endpoints
               Training pipeline, MLflow registry, drift detection
agent/         LangGraph supervisor: triage/action/comms sub-agents
               Webhook consumer, HIL approval state, Redis enqueue
worker/        Redis queue consumer
               Executes slow operations: replay, retrain, rollback
dashboard/     Streamlit dashboard
               Investigation state, HIL inbox, action queue, drift charts
shared/        Pydantic schemas
               All cross-service wire contracts (DriftEvent, PromotionRequest, HilApproval, etc.)
```
