import os

from fastapi import FastAPI

from app.cost_logging import ensure_cost_logging_tables
from app.generation import ensure_generation_tables
from app.ingestion import ensure_ingestion_tables
from app.interaction import ensure_interaction_tables
from config import run_startup_checks
from routes.interaction import router as interaction_router
from routes.settings import router as settings_router
from routes.upload import router as upload_router
from routes.workflow import router as workflow_router


app = FastAPI(title="Study Assistant API")
app.include_router(upload_router)
app.include_router(interaction_router)
app.include_router(workflow_router)
app.include_router(settings_router)


@app.on_event("startup")
def on_startup() -> None:
    run_startup_checks()
    ensure_cost_logging_tables()
    ensure_ingestion_tables()
    ensure_interaction_tables()
    ensure_generation_tables()


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
    storage_secret=os.getenv("SOCRATIC_STORAGE_SECRET", "socratic-ai-local-secret"),
)
