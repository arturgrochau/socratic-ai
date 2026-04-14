from fastapi import FastAPI

from app.cost_logging import ensure_cost_logging_tables
from app.generation import ensure_generation_tables
from app.ingestion import ensure_ingestion_tables
from app.interaction import ensure_interaction_tables
from app.linking import ensure_linking_tables
from app.processing import ensure_processing_tables
from config import run_startup_checks
from routes.interaction import router as interaction_router
from routes.upload import router as upload_router
from routes.workflow import router as workflow_router


app = FastAPI(title="Study Assistant API")
app.include_router(upload_router)
app.include_router(interaction_router)
app.include_router(workflow_router)


@app.on_event("startup")
def on_startup() -> None:
    run_startup_checks()
    ensure_cost_logging_tables()
    ensure_ingestion_tables()
    ensure_processing_tables()
    ensure_linking_tables()
    ensure_interaction_tables()
    ensure_generation_tables()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}