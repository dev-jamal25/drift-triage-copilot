FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY shared/ ./shared/
COPY worker/pyproject.toml ./worker/
WORKDIR /app/worker
RUN uv sync --no-install-project

COPY worker/ ./

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-m", "app.main"]
