from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

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
        return generate_tailored_learning(
            video_source_id=request.video_source_id,
            document_source_ids=request.document_source_ids,
            user_id=user_id,
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
