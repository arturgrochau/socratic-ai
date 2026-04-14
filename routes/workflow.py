from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.auth import get_user_id
from app.models import (
    GenerateTailoredLearningRequest,
    GenerateTailoredLearningResponse,
    ProcessPipelineRequest,
    ProcessPipelineResponse,
)
from app.workflow import run_processing_and_linking, run_tailored_learning_generation


router = APIRouter(tags=["workflow"])


@router.post("/process", response_model=ProcessPipelineResponse)
async def process_pipeline(
    request: ProcessPipelineRequest,
    http_request: Request,
) -> ProcessPipelineResponse:
    try:
        user_id = get_user_id(http_request)
        return run_processing_and_linking(
            video_source_id=request.video_source_id,
            document_source_ids=request.document_source_ids,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected workflow failure.") from exc


@router.post("/generate-tailored-learning", response_model=GenerateTailoredLearningResponse)
async def generate_tailored_learning(
    request: GenerateTailoredLearningRequest,
    http_request: Request,
) -> GenerateTailoredLearningResponse:
    try:
        user_id = get_user_id(http_request)
        return run_tailored_learning_generation(
            video_source_id=request.video_source_id,
            document_source_ids=request.document_source_ids,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected generation workflow failure.") from exc
