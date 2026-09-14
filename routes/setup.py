"""First-run wizard endpoints. Thin wrappers over app/setup.py."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import setup

router = APIRouter(prefix="/setup", tags=["setup"])


class PullRequest(BaseModel):
    models: list[str]


class CompleteRequest(BaseModel):
    mode: str  # "local" | "api"


@router.get("/status", response_model=setup.EnvironmentReport)
def status() -> setup.EnvironmentReport:
    return setup.detect_environment()


@router.post("/ollama-start")
def ollama_start() -> dict[str, object]:
    return setup.start_ollama()


@router.post("/ollama-pull")
def ollama_pull(body: PullRequest) -> dict[str, object]:
    models = [m.strip() for m in body.models if m and m.strip()]
    if not models:
        raise HTTPException(status_code=422, detail="No models given.")
    return setup.start_pull(models).view()


@router.get("/ollama-pull/{job_id}")
def ollama_pull_status(job_id: str) -> dict[str, object]:
    job = setup.get_pull(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown pull job.")
    return job.view()


@router.post("/complete")
def complete(body: CompleteRequest) -> dict[str, object]:
    mode = (body.mode or "").strip().lower()
    if mode not in ("local", "api"):
        raise HTTPException(status_code=422, detail="mode must be 'local' or 'api'.")
    settings = setup.complete_setup(mode)
    return {"ok": True, "mode": mode, "setup_complete": settings.setup_complete}
