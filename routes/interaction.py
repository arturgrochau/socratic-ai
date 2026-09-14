from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.auth import get_user_id
from app.interaction import handle_user_query, stream_user_query
from app.models import AskRequest, AskResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["interaction"])


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    http_request: Request,
) -> AskResponse:
    try:
        user_id = get_user_id(http_request)
        # Offload the synchronous, LLM-bound handler to a worker thread so it
        # doesn't freeze the shared event loop (and the NiceGUI websocket).
        return await run_in_threadpool(
            handle_user_query,
            request.query,
            request.source_ids,
            request.session_id,
            user_id=user_id,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected ask interaction failure.") from exc


@router.post("/ask/stream")
async def ask_stream_endpoint(request: AskRequest, http_request: Request) -> StreamingResponse:
    """Same turn as /ask, streamed as NDJSON: {"delta": ...} lines, then one
    {"done": true, "session_id", "answer", "model_name"}; {"error": ...} on failure."""
    user_id = get_user_id(http_request)

    def events():
        try:
            for event in stream_user_query(
                request.query,
                request.source_ids,
                request.session_id,
                user_id=user_id,
                top_k=request.top_k,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except (ValueError, RuntimeError) as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
        except Exception:  # noqa: BLE001 — the stream is already open; report, don't 500
            logger.exception("ask/stream failed")
            yield json.dumps({"error": "Unexpected ask interaction failure."}) + "\n"

    # The generator is synchronous and LLM-bound; iterate it off the event loop.
    return StreamingResponse(iterate_in_threadpool(events()), media_type="application/x-ndjson")
