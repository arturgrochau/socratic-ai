"""First-run setup: detect what this machine has, recommend local models that
fit its memory, and pull them from Ollama with progress.

Everything here is read by the Welcome page (frontend/pages/welcome.py) through
routes/setup.py. No LLM calls; only local probes and the Ollama daemon."""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import BaseModel

import config
from app import ollama_probe

logger = logging.getLogger(__name__)

OLLAMA_APP = Path("/Applications/Ollama.app")
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
EMBEDDING_MODEL = config.LOCAL_EMBEDDING_MODEL_DEFAULT


@dataclass(frozen=True)
class ModelTier:
    name: str
    min_ram_gb: float
    generation_model: str
    download_gb: float
    num_ctx: int
    note: str = ""


# Ordered smallest first; the last tier whose min_ram_gb fits wins. Sizes are
# the q4_K_M downloads. The embedding model (~270 MB) rides along with every tier.
MODEL_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        "compact", 0, "qwen3:4b", 2.6, 8192,
        note="Under 12 GB of memory: local mode works but the API mode gives noticeably better packs.",
    ),
    ModelTier("standard", 12, "qwen3:8b", 5.2, 8192),
    ModelTier("large", 24, "qwen3:14b", 9.3, 16384),
    ModelTier("full", 32, "qwen3:30b-a3b-instruct-2507-q4_K_M", 18.6, 16384,
              note="30B-class quality at 4B speed on Apple Silicon."),
)


def recommend_tier(ram_gb: float | None) -> ModelTier:
    if ram_gb is None:
        return MODEL_TIERS[1]
    chosen = MODEL_TIERS[0]
    for tier in MODEL_TIERS:
        if ram_gb >= tier.min_ram_gb:
            chosen = tier
    return chosen


def detect_ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
            return round(int(out.stdout.strip()) / 1024**3, 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 1)
    except Exception:  # noqa: BLE001 — unknown hardware is a valid answer
        return None


class ToolStatus(BaseModel):
    found: bool
    path: str | None = None
    version: str | None = None


class OllamaStatus(BaseModel):
    installed: bool
    running: bool
    host: str
    models: list[str] = []
    detail: str = ""


class TierView(BaseModel):
    name: str
    generation_model: str
    embedding_model: str
    download_gb: float
    num_ctx: int
    note: str = ""


class EnvironmentReport(BaseModel):
    platform: str
    arch: str
    ram_gb: float | None
    ffmpeg: ToolStatus
    ollama: OllamaStatus
    mlx_available: bool
    recommended: TierView
    missing_models: list[str]
    setup_complete: bool
    mode: str


def _ffmpeg_status() -> ToolStatus:
    path = shutil.which("ffmpeg")
    if not path:
        return ToolStatus(found=False)
    version = None
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=3)
        version = out.stdout.split("\n", 1)[0].replace("ffmpeg version", "").strip().split(" ")[0]
    except Exception:  # noqa: BLE001
        pass
    return ToolStatus(found=True, path=path, version=version)


def _ollama_status(host: str) -> OllamaStatus:
    installed = bool(shutil.which("ollama")) or OLLAMA_APP.exists()
    try:
        models = ollama_probe.list_models(host, timeout=1.5)
        return OllamaStatus(installed=True, running=True, host=host, models=models)
    except Exception as exc:  # noqa: BLE001
        return OllamaStatus(installed=installed, running=False, host=host, detail=str(exc)[:200])


def _mlx_available() -> bool:
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        return False
    import importlib.util

    return importlib.util.find_spec("mlx") is not None


def _has_model(pulled: list[str], wanted: str) -> bool:
    names = set(pulled) | {n.removesuffix(":latest") for n in pulled}
    return wanted in names or f"{wanted}:latest" in names


def detect_environment() -> EnvironmentReport:
    settings = config.get_settings()
    ram = detect_ram_gb()
    tier = recommend_tier(ram)
    ollama = _ollama_status(settings.ollama_host)
    # Once setup is complete the user's own choices are what must be present,
    # not the recommendation.
    wanted = (
        [settings.local_generation_model, settings.local_retrieval_model]
        if settings.setup_complete
        else [tier.generation_model, EMBEDDING_MODEL]
    )
    missing = [m for m in dict.fromkeys(wanted) if not _has_model(ollama.models, m)] if ollama.running else wanted
    if settings.setup_complete:
        # Re-running the wizard: describe what is configured, not what we would pick.
        shown = TierView(
            name="current",
            generation_model=settings.local_generation_model,
            embedding_model=settings.local_retrieval_model,
            download_gb=0.0,
            num_ctx=config.ollama_num_ctx(),
        )
    else:
        shown = TierView(
            name=tier.name,
            generation_model=tier.generation_model,
            embedding_model=EMBEDDING_MODEL,
            download_gb=round(tier.download_gb + 0.3, 1),
            num_ctx=tier.num_ctx,
            note=tier.note,
        )
    return EnvironmentReport(
        platform=sys.platform,
        arch=platform.machine(),
        ram_gb=ram,
        ffmpeg=_ffmpeg_status(),
        ollama=ollama,
        mlx_available=_mlx_available(),
        recommended=shown,
        missing_models=missing,
        setup_complete=settings.setup_complete,
        mode=config.current_mode(),
    )


def start_ollama() -> dict[str, object]:
    """Try to start the daemon the way the user installed it."""
    if OLLAMA_APP.exists():
        subprocess.Popen(["open", "-a", str(OLLAMA_APP)])
        return {"started": True, "how": "app"}
    cli = shutil.which("ollama")
    if cli:
        subprocess.Popen([cli, "serve"], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"started": True, "how": "cli"}
    return {"started": False, "how": None, "hint": OLLAMA_DOWNLOAD_URL}


# ── Model pulls ──────────────────────────────────────────────────────────────


@dataclass
class PullJob:
    job_id: str
    models: list[str]
    current: str | None = None
    status: str = "queued"  # queued | pulling | done | failed
    completed: int = 0
    total: int = 0
    percent: float = 0.0
    finished: list[str] = field(default_factory=list)
    error: str | None = None

    def view(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "models": self.models,
            "current": self.current,
            "status": self.status,
            "completed": self.completed,
            "total": self.total,
            "percent": round(self.percent, 1),
            "finished": self.finished,
            "error": self.error,
            "done": self.status in ("done", "failed"),
        }


_jobs: dict[str, PullJob] = {}
_jobs_lock = threading.Lock()


def _run_pull(job: PullJob, host: str, client: httpx.Client | None = None) -> None:
    base = host.rstrip("/")
    http = client or httpx.Client(timeout=httpx.Timeout(None, connect=10))
    try:
        for name in job.models:
            job.current, job.status, job.percent, job.completed, job.total = name, "pulling", 0.0, 0, 0
            with http.stream("POST", f"{base}/api/pull", json={"name": name, "stream": True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    event = httpx.Response(200, content=line).json()
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    total = int(event.get("total") or 0)
                    completed = int(event.get("completed") or 0)
                    if total:
                        job.total, job.completed = total, completed
                        job.percent = min(100.0, 100.0 * completed / total)
                    if event.get("status") == "success":
                        job.percent = 100.0
            job.finished.append(name)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 — reported through the job view
        job.status = "failed"
        job.error = str(exc)[:300]
        logger.warning("model pull failed: %s", exc)


def start_pull(models: list[str], host: str | None = None) -> PullJob:
    settings = config.get_settings()
    job = PullJob(job_id=uuid.uuid4().hex[:12], models=[m for m in dict.fromkeys(models) if m])
    with _jobs_lock:
        _jobs[job.job_id] = job
    threading.Thread(target=_run_pull, args=(job, host or settings.ollama_host), daemon=True).start()
    return job


def get_pull(job_id: str) -> PullJob | None:
    return _jobs.get(job_id)


def complete_setup(mode: str, *, tier: ModelTier | None = None) -> config.Settings:
    """Persist the chosen mode (and, for a fresh local install, the tier)."""
    from dataclasses import replace

    settings = config.settings_for_mode(mode)
    if mode == "local" and not config.get_settings().setup_complete:
        tier = tier or recommend_tier(detect_ram_gb())
        settings = replace(
            settings,
            local_generation_model=tier.generation_model,
            local_chat_model=tier.generation_model,
            local_aux_model=tier.generation_model,
            local_retrieval_model=EMBEDDING_MODEL,
            local_num_ctx=tier.num_ctx,
        )
    return config.apply_settings(replace(settings, setup_complete=True))
