FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY shared/ ./shared/
COPY dashboard/pyproject.toml ./dashboard/
WORKDIR /app/dashboard
RUN uv sync --no-install-project

COPY dashboard/ ./

ENV PYTHONPATH=/app
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
