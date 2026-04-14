from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.auth import get_user_id
from app.ingestion import ingest_upload_bundle
from app.models import IngestionResponse


router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=IngestionResponse)
async def upload_content(
    request: Request,
    video: UploadFile = File(...),
    documents: list[UploadFile] = File(...),
) -> IngestionResponse:
    try:
        user_id = get_user_id(request)
        return await ingest_upload_bundle(
            video_file=video,
            document_files=documents,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected ingestion failure.") from exc
    finally:
        await video.close()
        for document in documents:
            await document.close()
