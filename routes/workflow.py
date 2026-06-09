from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.auth import get_user_id
from app.generation import GenerationStageError, generate_tailored_learning
from app.models import (
    GenerateTailoredLearningRequest,
    GenerateTailoredLearningResponse,
)


router = APIRouter(tags=["workflow"])


@router.post("/generate-tailored-learning", response_model=GenerateTailoredLearningResponse)
async def generate_tailored_learning_endpoint(
    request: GenerateTailoredLearningRequest,
    http_request: Request,
) -> GenerateTailoredLearningResponse:
    try:
        user_id = get_user_id(http_request)
        # Generation is synchronous and slow (many LLM calls). Run it in a worker
        # thread so it never blocks the shared event loop that also serves the
        # NiceGUI websocket — otherwise the UI shows "not connected" mid-run.
        return await run_in_threadpool(
            generate_tailored_learning,
            request.video_source_id,
            request.document_source_ids,
            user_id,
        )
    except GenerationStageError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Generation stage failed after automatic retries.",
                "run_id": exc.run_id,
                "stage": exc.stage_name,
                "attempt": exc.attempt_number,
                "reason": exc.reason,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected generation workflow failure.") from exc
