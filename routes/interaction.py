from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.auth import get_user_id
from app.interaction import handle_user_query
from app.models import AskRequest, AskResponse


router = APIRouter(tags=["interaction"])


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    http_request: Request,
) -> AskResponse:
    try:
        user_id = get_user_id(http_request)
        return handle_user_query(
            query=request.query,
            source_ids=request.source_ids,
            session_id=request.session_id,
            user_id=user_id,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected ask interaction failure.") from exc
