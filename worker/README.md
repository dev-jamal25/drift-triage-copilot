# Worker Service

`worker/` is the background job executor for the drift triage system.

The agent decides what should happen. The worker only executes queued actions safely.

It consumes `QueuedAction` messages from Redis, validates them with `shared.contracts.QueuedAction`, dispatches the matching handler, retries transient failures, sends terminal failures to the DLQ, and records job state in Postgres.

## System role

```text
platform  -> emits drift webhook
agent     -> investigates drift and enqueues QueuedAction
redis     -> shared queue
worker    -> consumes and executes QueuedAction
postgres  -> stores worker job status and idempotency state
dashboard -> later displays queue/job state
```

Redis and Postgres are shared infrastructure from the root `docker-compose.yml`.
There is no `redis/` folder and no worker-only Postgres container.

## Message contract

The worker expects messages using the shared contract:

```python
from shared.contracts import QueuedAction
```

Required fields:

```text
idempotency_key
investigation_id
model_name
action_type
target_version
payload
attempt
max_attempts
created_at
```

Supported actions:

```text
replay_test
retrain
rollback
```

`shared/contracts.py` must not be edited by the worker.

## Queue flow

Redis keys use the `worker:queue:` prefix:

```text
worker:queue:ready       # waiting jobs
worker:queue:processing  # claimed jobs
worker:queue:retry       # delayed retries
worker:queue:dlq         # terminal failures
```

Flow:

```text
ready -> processing -> handler

success            -> ack and remove from processing
retryable failure  -> schedule retry and remove from processing
terminal failure   -> push DLQ envelope and remove from processing
validation failure -> push DLQ envelope and remove from processing
```

The queue adapter follows a copy-first-then-remove rule:

```text
copy to retry/DLQ first
then LREM the original message from processing
```

This avoids message loss if the worker crashes mid-transition.

## Handlers

| Action | Current behaviour |
|---|---|
| `replay_test` | Safe stub success |
| `retrain` | Safe stub success; no MLflow writes and no promotion |
| `rollback` | Blocked and sent to DLQ until HIL + promotion gate exist |

The worker does not contain HIL logic and does not call platform `/promote`.

## Persistence

Worker state is stored in Postgres table:

```text
worker_action_jobs
```

Main columns:

```text
idempotency_key
investigation_id
model_name
action_type
target_version
status
attempt
max_attempts
payload
result
last_error
created_at
started_at
finished_at
updated_at
```

Worker migrations use a separate Alembic version table:

```text
worker_alembic_version
```

This avoids collisions with platform migrations.

Statuses:

```text
pending           # created, not running yet
running           # dispatch in progress
succeeded         # completed successfully
retry_scheduled   # waiting for retry
dead_lettered     # terminal failure or max attempts exhausted
blocked           # intentionally blocked, e.g. rollback before HIL gate
skipped_duplicate # duplicate of an already completed job
```

## Idempotency

The worker assumes **at-least-once delivery**, not exactly-once delivery.

Duplicate protection uses two layers:

1. Postgres unique `idempotency_key`
2. Redis runtime lock:

```text
worker:lock:{idempotency_key}
```

Behaviour:

```text
already succeeded -> ack and skip dispatch
lock busy         -> requeue with short delay, do not increment attempt
lock acquired     -> mark running and dispatch
```

The lock has a TTL so crashes recover automatically.

## Running locally

From `worker/`:

```bash
uv sync --extra dev
uv run pytest -q                     # Unit tests only (67 passed, 2 skipped)
uv run ruff check .
uv run ruff format --check .
```

To run **all tests including E2E** (requires Docker services):

```bash
docker compose up -d redis postgres
uv run alembic upgrade head          # Apply database schema
uv run pytest -q -m requires_docker  # Run E2E integration tests
```

To run **specific tests**:

```bash
uv run pytest -q -m "not requires_docker"  # Explicit: skip Docker tests
uv run pytest -q tests/test_loop_dispatch.py -v  # Single file
```

### Database migrations

**Local execution** (if PostgreSQL is running on localhost:5432):

```bash
cd worker
uv run alembic upgrade head
```

**Docker execution** (recommended for consistency):

```bash
docker compose run --rm worker alembic upgrade head
```

This uses the DATABASE_URL from `.env` (postgres:5432 inside the network).

### Start the full worker stack

From repo root:

```bash
docker compose up -d redis postgres worker
docker compose logs -f worker
```

## Manual test action

From `worker/`:

```bash
uv run python scripts/seed_test_action.py replay_test
```

Other test actions:

```bash
uv run python scripts/seed_test_action.py retrain
uv run python scripts/seed_test_action.py rollback
```

Expected results:

```text
replay_test -> worker_action_jobs.status = succeeded
retrain     -> worker_action_jobs.status = succeeded
rollback    -> worker_action_jobs.status = blocked and DLQ entry created
```

## Integration tests

Default tests do not require Docker (67 unit tests):

```bash
cd worker
uv run pytest -q
```

Docker-based E2E tests are skipped by default. To run them:

```bash
# Step 1: Start Docker services (from repo root)
docker compose up -d redis postgres

# Step 2: Apply migrations (from repo root)
docker compose run --rm worker alembic upgrade head

# Step 3: Run E2E tests (from worker/)
uv run pytest -q -m requires_docker
```

Or run all tests at once:

```bash
uv run pytest -q  # Skips Docker tests
uv run pytest -q -m "not requires_docker"  # Explicit skip
uv run pytest -q -m requires_docker  # Docker tests only
```

## Inspect worker state

Recent jobs:

```sql
SELECT idempotency_key, action_type, status, attempt, updated_at
FROM worker_action_jobs
ORDER BY updated_at DESC
LIMIT 20;
```

DLQ entries:

```bash
docker compose exec redis redis-cli LRANGE worker:queue:dlq 0 -1
```

## Troubleshooting

### `uv run pytest -q` fails with "Access is denied" on Windows

**Symptom:** Error like `PermissionError: [WinError 5] Access is denied: '.venv\\Lib\\...\\lib64'`

**Cause:** Docker created a Linux-style `.venv` on the Windows host via bind mount. The `.venv/lib64` directory doesn't exist on Windows, and paths are incompatible.

**Solution:**

```powershell
# From repo root
Remove-Item ./worker/.venv -Recurse -Force -ErrorAction SilentlyContinue
docker compose down
# Rebuild with the anonymous volume fix
docker compose build --no-cache worker
```

This should not happen again after the docker-compose.yml fix (anonymous volume for `.venv`).

### Docker tests pass, but `uv run pytest -q` runs them anyway

**Symptom:** E2E tests run even though you didn't intend them to.

**Cause:** `RUN_DOCKER_TESTS=1` environment variable left set in your PowerShell session.

**Solution:**

```powershell
# Check if it's set
$env:RUN_DOCKER_TESTS

# Clear it
Remove-Item env:RUN_DOCKER_TESTS -ErrorAction SilentlyContinue

# Verify tests skip again
uv run pytest -q  # Should show 2 skipped, not running Docker tests
```

Alternatively, always use pytest markers instead:

```powershell
uv run pytest -q -m "not requires_docker"  # Always skips Docker tests
uv run pytest -q -m requires_docker        # Always runs Docker tests
```

### Migration fails: "password authentication failed for user drift_user"

**Symptom:** `asyncpg.exceptions.InvalidPasswordError` when running `uv run alembic upgrade head`

**Cause:** `.env` has Docker hostnames ("postgres"), not localhost. Local execution needs localhost override, or use Docker to run migrations.

**Solution (local):**

```powershell
cd worker
$env:DATABASE_URL = "postgresql+asyncpg://jamal:pass1234@localhost:5432/drift_triage"
uv run alembic upgrade head
```

**Solution (recommended — use Docker):**

```powershell
docker compose run --rm worker alembic upgrade head
```

## Rules

Do not:

```text
edit shared/contracts.py
create a redis/ folder
create a worker-only Postgres container
put HIL logic in the worker
call MLflow from retrain stub
call platform /promote from the worker
implement rollback before HIL + promotion gate exist
overwrite .env
```

The worker is an executor, not a decision-maker.
