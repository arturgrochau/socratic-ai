import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from app.cost_logging import ensure_cost_logging_tables
from app.generation import ensure_generation_tables
from app.ingestion import ensure_ingestion_tables
from app.interaction import ensure_interaction_tables
from config import run_startup_checks
from routes.interaction import router as interaction_router
from routes.settings import router as settings_router
from routes.setup import router as setup_router
from routes.upload import router as upload_router
from routes.workflow import router as workflow_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    run_startup_checks()
    ensure_cost_logging_tables()
    ensure_ingestion_tables()
    ensure_interaction_tables()
    ensure_generation_tables()
    yield
    # The native app owns its daemon usage: when the window closes, evict the
    # local models instead of leaving ~19 GB resident for the keep_alive
    # window. Terminal users may want the model warm, so only in native mode.
    if os.getenv("SOCRATIC_NATIVE") == "1" and config.current_mode() == "local":
        from app.ollama_probe import unload_models

        unload_models(config.get_settings().ollama_host, config.local_models_in_use())


app = FastAPI(title="Socratic AI", lifespan=lifespan)
app.include_router(upload_router)
app.include_router(interaction_router)
app.include_router(workflow_router)
app.include_router(settings_router)
app.include_router(setup_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Mount the NiceGUI frontend onto this same FastAPI app so the whole product
# runs in one uvicorn process (no separate Streamlit server / port 8501).
from nicegui import ui  # noqa: E402  (import after app is defined)

from frontend.ui import init_ui  # noqa: E402

init_ui()
ui.run_with(
    app,
    title="Socratic AI",
    favicon="🦉",
    # None = follow the OS appearance (prefers-color-scheme), which is what a
    # native window is expected to do.
    dark=None,
    storage_secret=config.storage_secret(),
    # Tolerate brief disconnects (alt-tab, sleep) without purging per-tab state.
    reconnect_timeout=float(os.getenv("SOCRATIC_RECONNECT_TIMEOUT", "20")),
)
