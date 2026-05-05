FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy shared first (for layer caching)
COPY shared/ ./shared/

# Copy platform deps and install
COPY platform/pyproject.toml ./platform/
WORKDIR /app/platform
RUN uv sync --no-install-project

# Copy source
COPY platform/ ./

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
