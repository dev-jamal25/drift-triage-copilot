FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY shared/ ./shared/
COPY agent/pyproject.toml ./agent/
WORKDIR /app/agent
RUN uv sync --no-install-project

COPY agent/ ./

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
