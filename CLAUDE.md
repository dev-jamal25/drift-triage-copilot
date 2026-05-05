# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is
Week 5 AIE Bootcamp pair project. A self-healing MLOps stack: a binary
classifier on UCI Bank Marketing, served via FastAPI with rolling drift
detection (PSI + chi²); a LangGraph supervisor agent that reacts to drift
alerts, dispatches slow consequences through a Redis queue, and pauses
for HIL approval before changing Production.

Read these in order before doing any task:
1. README.md
2. ARCH.md
3. DECISIONS.md
4. shared/contracts.py

## Commands

Each package (`platform/`, `agent/`, `worker/`, `dashboard/`) is an independent `uv` project. Run all commands from within the package directory.

```bash
# Install deps (including dev extras)
uv sync --all-extras

# Lint + format check (mirrors CI)
uv run ruff check .
uv run ruff format --check .

# Auto-fix formatting
uv run ruff format .

# Run all tests
uv run pytest -q

# Run a single test file
uv run pytest tests/test_smoke.py -q

# Run a single test by name
uv run pytest -k "test_healthz" -q
```

Start the full stack (from repo root):
```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY
docker compose up --build
docker compose logs -f agent  # tail a specific service
```

Services when running:
- `http://localhost:8000` — platform (FastAPI `/docs` for interactive API)
- `http://localhost:8001` — agent
- `http://localhost:8501` — dashboard (Streamlit)
- `http://localhost:5000` — MLflow UI

## Architecture & data flow

```
[client] --POST /predict--> platform (FastAPI)
                                |
                         rolling drift window
                         PSI (numeric) + chi² (categorical)
                                |
                     severity change? --webhook--> agent (LangGraph)
                                                       |
                                              triage sub-agent (LLM)
                                              action sub-agent (LLM)
                                              comms sub-agent (LLM)
                                                       |
                                              enqueue QueuedAction
                                              to Redis (NOT execute)
                                                       |
                                     worker consumes + executes
                                     (replay_test / retrain / rollback)
                                                       |
                                        HIL approval needed?
                                        dashboard surfaces HilApproval row
                                        human approves/denies
                                                       |
                                    agent sends PromotionRequest to platform
                                    platform validates + promotes in MLflow
```

**Shared state:** Postgres holds investigation state, HIL approvals, and action log. The agent uses `langgraph-checkpoint-postgres` for durable graph state across restarts.

**Key contracts in `shared/contracts.py`** (all `extra="forbid"`):
- `DriftEvent` — platform → agent webhook payload
- `QueuedAction` — agent → Redis → worker; idempotency key = `{investigation_id}:{action_type}:{target_version}`
- `ActionResult` — worker → DB log on completion
- `PromotionRequest` / `PromotionResult` — agent → platform after HIL
- `HilApproval` — DB row surfaced in dashboard, status: `pending | approved | denied | superseded`

**Three LLM sub-agents** (each uses its own model env var — `LLM_TRIAGE_MODEL`, `LLM_ACTION_MODEL`, `LLM_COMMS_MODEL`). Prompts live in `agent/app/prompts/*.md`, never as inline strings.

## Repository layout
- `platform/`  FastAPI model service, MLflow, drift detection
- `agent/`     LangGraph supervisor (triage + action + comms sub-agents)
- `worker/`    Redis queue consumer (slow tools: replay, retrain, rollback)
- `dashboard/` Streamlit
- `shared/`    Pydantic contracts shared by all services — schema changes here are breaking changes, both authors must approve

## Hard rules — never violate
- All routes async. No `requests`, no `time.sleep`, no blocking I/O in request paths. Use `httpx.AsyncClient`.
- No globals at module level for the model, db engine, redis client, or LLM client. All shared resources go in lifespan handlers and are injected via FastAPI `Depends()`.
- Pydantic models at every external boundary (HTTP request/response, agent tool inputs, LLM structured outputs).
- Type hints on every function signature.
- `structlog` (or stdlib logging), never `print()`.
- Every external call has timeout + tenacity retry. No silent failures.
- Prompts are markdown files in `agent/app/prompts/`, never inline strings.
- Tools enqueue to Redis; they do NOT execute slow work in the agent process.
- Idempotency keys on every queued action.
- All `Settings` must use `pydantic-settings`; never call `os.getenv()` outside the `Settings` class.

## Do not modify without confirming first
- `shared/contracts.py` — both partners must approve changes
- `docker-compose.yml` service topology
- `.github/workflows/` — CI definitions

## Stop and ask if you're about to:
- Change a Pydantic schema in `shared/`
- Add a new top-level dependency
- Touch CI configuration
- Skip writing tests "for now"
- Catch a broad `Exception`
- Use `os.getenv()` outside the `Settings` class

## When making decisions
Append a one-paragraph entry to `DECISIONS.md`. The format is: the choice, alternatives considered, reason. No exceptions — every architectural choice gets a paragraph or it doesn't get made.

## When stuck
- Reach for Context7 for library docs (LangGraph, MLflow, Pydantic v2) before guessing.
- For runtime errors: bring traceback + what you tried + your hypothesis back to Claude AI for the architectural conversation.
