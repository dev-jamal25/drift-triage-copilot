# Platform — Model Service

The platform-track owns training, the MLflow registry, drift detection, and
the FastAPI prediction service. This README is the reference for engineers
working inside `platform/` only — see the [top-level README](../README.md)
for the full system and [ARCH.md](../ARCH.md) for cross-service flow.

## Layout

```
platform/
  app/                    FastAPI service
    core/                 Settings (pydantic-settings), structlog config
    routers/              /predict (future: /promote, /drift)
    schemas/              Pydantic request/response (platform-internal)
    services/             Model loader, MLflow client wrappers
    dependencies.py       Depends() providers (settings, model bundle)
    main.py               Lifespan + /healthz + /readyz
  ml/                     Reusable training + registry code
    data.py               Load, validate, clean, split (60/20/20 stratified)
    features.py           ColumnTransformer (StandardScaler + OneHotEncoder)
    train.py              CV bake-off + threshold tuning + CLI entry point
    evaluate.py           tune_threshold (recall floor), evaluate_at_threshold
    registry.py           MLflow log + register + load + fidelity check
    model_card.md         Markdown template, rendered at log time
  notebooks/
    train.ipynb           Exploratory EDA + bake-off narrative (not in CI)
  tests/                  pytest suite (see "Testing" below)
  data/raw/               <- bank-additional-full.csv goes here (gitignored)
  data/processed/         Generated splits + model.joblib + metrics.json + registry.json
  mlruns/                 Local MLflow file store (gitignored)
```

## Run commands

```bash
# Install (Python 3.12+, uv)
uv sync --all-extras

# Bring up the project's Postgres (required — no SQLite fallback)
docker compose up -d postgres

# Apply migrations (run once after pulling new revisions)
uv run alembic upgrade head

# Train + register the staging model (writes data/processed/* + mlruns/)
uv run python -m ml.train --data data/raw/bank-additional-full.csv --out data/processed/

# Skip MLflow registration (faster iteration during development)
uv run python -m ml.train --data data/raw/bank-additional-full.csv --out data/processed/ --no-register

# Serve the registered staging model
uv run uvicorn app.main:app --reload
# -> http://127.0.0.1:8000/docs    interactive OpenAPI
# -> http://127.0.0.1:8000/healthz liveness (process up)
# -> http://127.0.0.1:8000/readyz  readiness (model loaded)
# -> http://127.0.0.1:8000/predict POST a single client (see schema below)

# Lint + format (mirrors CI)
uv run ruff check .
uv run ruff format --check .

# Tests
uv run pytest -q
uv run pytest -q tests/test_predict.py            # one file
uv run pytest -q -k "test_threshold_rule"          # by name
```

## Database & migrations

Postgres-only — there's no SQLite fallback. Local dev requires the
`postgres` service from [docker-compose.yml](../docker-compose.yml)
running on `localhost:5432`. The compose service reads credentials from
[.env](../.env.example).

Schema is managed with Alembic. Migrations are **explicit** — never run
from the FastAPI lifespan.

```bash
# Bring up Postgres
docker compose up -d postgres

# Apply all migrations
uv run alembic upgrade head

# Create a new migration (manual stub; we don't autogenerate yet)
uv run alembic revision -m "<short slug>"

# Roll back one revision
uv run alembic downgrade -1
```

Tables in scope today:

| Table             | Purpose                                                  |
| ----------------- | -------------------------------------------------------- |
| `predictions_log` | One row per `POST /predict` (features, score, label, …)  |

If you `uvicorn app.main:app` against an empty database, the first
`/predict` will fail loudly — that's by design. Run `alembic upgrade head`.

The platform's Alembic uses `version_table="platform_alembic_version"`
(see [alembic/env.py](alembic/env.py)) because MLflow shares this Postgres
instance and owns the default `alembic_version` table.

## Configuration

All runtime config flows through `Settings` in
[app/core/config.py](app/core/config.py); env vars use the `PLATFORM_`
prefix and a local `.env` is honoured. `os.getenv` is forbidden anywhere
else (CLAUDE.md hard rule).

| Setting                          | Default                     | Purpose                                                          |
| -------------------------------- | --------------------------- | ---------------------------------------------------------------- |
| `PLATFORM_MLFLOW_TRACKING_URI`   | `file:./mlruns`             | MLflow tracking + registry backend                               |
| `PLATFORM_MODEL_NAME`            | `bank-marketing-classifier` | Registered model name                                            |
| `PLATFORM_MODEL_ALIAS`           | `staging`                   | Alias resolved at startup (`models:/<name>@<alias>`)             |
| `PLATFORM_LOAD_MODEL_ON_STARTUP` | `true`                      | Set to `false` for tests/CI to skip MLflow                       |
| `PLATFORM_LOG_LEVEL`             | `INFO`                      | structlog level                                                  |
| `DATABASE_URL` / `PLATFORM_DATABASE_URL` | `postgresql+asyncpg://drift_user:change_me_locally@localhost:5432/drift_triage` | Async Postgres URL. `DATABASE_URL` is the project-wide convention used by docker-compose. |

## Dataset — UCI Bank Marketing

`bank-additional-full.csv` (~41,188 rows × 21 columns, semicolon-separated).
Fetch from <https://archive.ics.uci.edu/dataset/222/bank+marketing> and
place at `data/raw/bank-additional-full.csv`. The full variant is preferred
over `bank-additional.csv` because PSI/chi² drift detection in later
platform work needs reference distributions with enough categorical mass.

**Cleaning rules locked by [tests/test_data.py](tests/test_data.py):**

| Rule                                                                                                          | Why                                                |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Drop `duration`                                                                                               | Target leakage — known per the UCI data card       |
| Map `y` → `target` int 0/1                                                                                    | Binary classification                              |
| Keep `"unknown"` as a literal category                                                                        | "Unknown" is itself signal in this domain          |
| `pdays == 999` → `was_previously_contacted` (0/1) + `pdays_clean` (0 if sentinel); drop `pdays`               | Sentinel encoding without poisoning the numeric    |
| Stratified 60/20/20 split, `random_state=42`                                                                  | Reproducible, balanced splits                      |

The same `apply_pdays_sentinel` runs at training and inference time, so
training/serving skew on `pdays` is structurally impossible.

## Model bake-off

5-fold stratified CV on the train fold (mean ± std, sorted by `f1_macro_mean`):

| Model               | accuracy       | f1_macro          | roc_auc        |
| ------------------- | -------------- | ----------------- | -------------- |
| **logreg** (chosen) | 0.824 ± 0.004  | **0.670 ± 0.008** | 0.784 ± 0.005  |
| gb                  | 0.899 ± 0.001  | 0.648 ± 0.003     | 0.790 ± 0.007  |
| rf                  | 0.891 ± 0.001  | 0.642 ± 0.007     | 0.769 ± 0.012  |
| majority baseline   | 0.887 ± 0.000  | 0.470             | 0.500          |

`logreg` wins on macro F1 — the right metric here because the dataset is
~11% positive and macro F1 weights both classes equally. Random forest and
gradient boosting both look better on raw accuracy, but that's the
majority-class trap; all three trounce the baseline on F1 and ROC-AUC.

## Threshold rule

Pick the **highest** decision threshold whose validation recall is ≥ 0.75.
Implemented in [ml/evaluate.tune_threshold](ml/evaluate.py). Rationale:
missing buyers is the dominant cost on the marketing-call use case, so we
fix a recall floor and maximise precision under it. Configurable via
`--min-recall` on the CLI.

Held-out test results at the locked threshold (single shot, no peeking):

| metric    | value |
| --------- | ----- |
| threshold | 0.38  |
| accuracy  | 0.705 |
| macro F1  | 0.586 |
| ROC-AUC   | 0.801 |
| precision | 0.241 |
| recall    | 0.754 |

Val and test are essentially identical (recall 0.754 in both) — no
overfitting to validation.

## MLflow flow

Each `python -m ml.train` run logs **one MLflow run** carrying:

1. **Model binary** — fitted `sklearn.Pipeline` via
   `mlflow.sklearn.log_model`, inferred signature, 5-row input example,
   auto-registered under `bank-marketing-classifier`.
2. **`schema.json`** — input columns + categorical levels (pulled from the
   fitted `OneHotEncoder`) + output shape + class labels.
3. **`model_card.md`** — rendered from [ml/model_card.md](ml/model_card.md)
   with concrete metrics + dataset hash.

Plus run-level provenance:

- `dataset_hash` tag — sha256 of cleaned-frame parquet bytes (deterministic
  per source CSV; surfaces silent input drift between training runs).
- `env.python`, `env.sklearn`, `env.mlflow`, `env.pandas`, `env.numpy`,
  `env.platform` tags — interpreter and library versions.
- `threshold` param — the locked decision threshold (the FastAPI lifespan
  reads it back; no separate config file).

After registration, alias `staging` is set on the new version. The service
loads `models:/bank-marketing-classifier@staging`. To promote to
production:

```python
from mlflow.tracking import MlflowClient
MlflowClient().set_registered_model_alias(
    "bank-marketing-classifier", "production", version
)
```

A load-back fidelity check runs at the end of every CLI run — predictions
from the loaded model must match the in-memory pipeline within `1e-9`.
Any nonzero diff is a hard error.

## Prediction API

```
POST /predict
Content-Type: application/json
```

Request body matches one cleaned UCI row with raw column names (excluding
`duration` and `y`). `populate_by_name=True` so snake_case alternatives
work too. `extra="forbid"` ensures `duration` is rejected at the edge.

Response:

```json
{
  "score": 0.548,
  "label": 1,
  "threshold": 0.38,
  "model_name": "bank-marketing-classifier",
  "model_version": "2",
  "predicted_at": "2026-05-06T23:05:08.773299Z"
}
```

`label = 1` iff `score >= threshold`. Threshold and `model_version` come
from the bundle loaded at startup — they don't ship per-request.

## Testing

```
tests/
  test_smoke.py       /healthz
  test_data.py        Data prep + split rules (synthetic, no real CSV needed)
  test_threshold.py   Threshold rule edge cases
  test_registry.py    MLflow log/register/load + fidelity (file:// in tmp_path)
  test_schemas.py     Pydantic request/response validation
  test_predict.py     Route + lifespan, plus a real-MLflow round-trip in tmp_path
```

`uv run pytest -q` should be green from a fresh `uv sync --all-extras`.
No test depends on the real dataset, on a network, or on a shared MLflow
server — every MLflow-touching test isolates to its own `file://` URI
under `tmp_path`.
