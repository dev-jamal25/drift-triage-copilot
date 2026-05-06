# Drift Triage Co-Pilot

Week 5 AIE Bootcamp project. A self-healing MLOps stack: a binary classifier
trained on the UCI Bank Marketing dataset, served via FastAPI with rolling
drift detection (PSI + chi²); a LangGraph supervisor agent that consumes drift
alerts, decides what to do, dispatches slow consequences (replay, retrain,
rollback) through a Redis queue, persists state across restarts, and pauses
for human approval before any change to Production.

**Authors:** Jamal Hamd, Hadi Kanaan
**Tag:** `v0.1.0-week5`

## Architecture

See [ARCH.md](./ARCH.md) for the full diagram and data flow.

## Decisions

See [DECISIONS.md](./DECISIONS.md) for design choices and rationale.

## Operations

See [RUNBOOK.md](./RUNBOOK.md) for how to drift the system, approve actions,
and recover from a stuck investigation.

## Quick start

Prerequisites: Docker, Docker Compose, `uv` (for local dev outside containers).

```bash
git clone https://github.com/YOUR_USERNAME/drift-triage-copilot.git
cd drift-triage-copilot
cp .env.example .env
# Fill in ANTHROPIC_API_KEY in .env
docker compose up --build
```

Services:
- Model service: http://localhost:8000 (FastAPI docs at `/docs`)
- Agent: http://localhost:8001
- Dashboard: http://localhost:8501 (Streamlit)
- MLflow: http://localhost:5000

## Project structure

```
platform/    Model service: training, MLflow registry, drift, /predict endpoint
agent/       LangGraph supervisor: triage + action + comms sub-agents
worker/      Redis queue consumer: executes slow tools (replay/retrain/rollback)
dashboard/   Streamlit UI: registry, HIL inbox, queue depth, drift charts
shared/      Pydantic contracts shared by all services
```
