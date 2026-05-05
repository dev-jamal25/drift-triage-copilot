"""Model service entry point. Filled in on Day 1."""

from fastapi import FastAPI

app = FastAPI(title="Drift Triage Co-Pilot — Model Service")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
