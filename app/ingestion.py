from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import ffmpeg
import fitz
from fastapi import UploadFile
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.cost_logging import log_api_usage
from app.models import (
    DocumentIngestionRecord,
    DocumentPageText,
    IngestionResponse,
    TranscriptPayload,
    TranscriptSegment,
    UploadRequestMeta,
    VideoIngestionRecord,
)
from config import db_engine, openai_client


UPLOAD_ROOT = Path("uploads")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
RAW_CHUNK_TARGET_CHARS = 2400
RAW_CHUNK_OVERLAP_CHARS = 320
TRANSCRIPT_TEXT_PLACEHOLDER = "[chunked transcript stored in source_text_chunks]"


def ensure_ingestion_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_type TEXT NOT NULL CHECK (source_type IN ('video', 'document')),
                    filename TEXT NOT NULL,
                    mime_type TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        source_columns = {
            row[1] for row in connection.execute(text("PRAGMA table_info(sources)")).fetchall()
        }
        if "user_id" not in source_columns:
            connection.execute(text("ALTER TABLE sources ADD COLUMN user_id TEXT DEFAULT 'legacy'"))
            connection.execute(text("UPDATE sources SET user_id = 'legacy' WHERE user_id IS NULL"))
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_sources_user_id
                ON sources (user_id)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    transcript_text TEXT NOT NULL,
                    segments_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    page_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS source_text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL CHECK (chunk_type IN ('transcript', 'document')),
                    chunk_text TEXT NOT NULL,
                    timestamp_start REAL,
                    timestamp_end REAL,
                    page_number INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(id),
                    UNIQUE (user_id, source_id, chunk_type, chunk_index)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_source_text_chunks_user_source
                ON source_text_chunks (user_id, source_id)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_source_text_chunks_chunk_type
                ON source_text_chunks (chunk_type)
                """
            )
        )


def validate_video_file(video_file: UploadFile) -> None:
    if not video_file.filename:
        raise ValueError("Video file is required.")

    suffix = Path(video_file.filename).suffix.lower()
    mime_type = video_file.content_type or ""
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS and not mime_type.startswith("video/"):
        raise ValueError("Video file must be a valid video format.")


def validate_document_files(document_files: list[UploadFile]) -> None:
    if not document_files:
        raise ValueError("At least one document file is required.")

    for document in document_files:
        if not document.filename:
            raise ValueError("Each document must include a filename.")

        suffix = Path(document.filename).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise ValueError(
                f"Unsupported document file type for {document.filename}. "
                "Allowed: .pdf, .txt, .md"
            )


def save_upload_file(upload: UploadFile, destination_dir: Path) -> Path:
    if not upload.filename:
        raise ValueError("Upload is missing a filename.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / Path(upload.filename).name

    upload.file.seek(0)
    with destination_path.open("wb") as destination:
        shutil.copyfileobj(upload.file, destination)

    if destination_path.stat().st_size == 0:
        destination_path.unlink(missing_ok=True)
        raise ValueError(f"Uploaded file is empty: {upload.filename}")

    upload.file.seek(0)
    return destination_path


def extract_audio_from_video(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), ac=1, ar=16000, format="wav")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else "Unknown ffmpeg error"
        raise RuntimeError(f"Audio extraction failed: {stderr}") from exc

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("Audio extraction produced no audio output.")


def transcribe_audio_with_whisper(audio_path: Path, user_id: str) -> TranscriptPayload:
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("Audio file is missing or empty.")

    with audio_path.open("rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    log_api_usage(
        response=transcription,
        user_id=user_id,
        call_stage="ingestion",
        model_name="whisper-1",
    )

    transcript_text = getattr(transcription, "text", None)
    segments_raw = getattr(transcription, "segments", None)

    if isinstance(transcription, dict):
        if transcript_text is None:
            transcript_text = transcription.get("text")
        if segments_raw is None:
            segments_raw = transcription.get("segments")

    text_value = (transcript_text or "").strip()
    if not text_value:
        raise RuntimeError("Transcription returned empty text.")

    segments: list[TranscriptSegment] = []
    for segment in segments_raw or []:
        if isinstance(segment, dict):
            start_value = float(segment.get("start", 0.0))
            end_value = float(segment.get("end", 0.0))
            segment_text = str(segment.get("text", "")).strip()
        else:
            start_value = float(getattr(segment, "start", 0.0))
            end_value = float(getattr(segment, "end", 0.0))
            segment_text = str(getattr(segment, "text", "")).strip()

        segments.append(
            TranscriptSegment(start=start_value, end=end_value, text=segment_text)
        )

    if not segments:
        raise RuntimeError("Transcription did not include segment timestamps.")

    return TranscriptPayload(text=text_value, segments=segments)


def extract_pdf_text_with_pages(pdf_path: Path) -> list[DocumentPageText]:
    try:
        with fitz.open(pdf_path) as document:
            pages: list[DocumentPageText] = []
            for index in range(document.page_count):
                page = document.load_page(index)
                page_text = (page.get_text("text") or "").strip()
                pages.append(DocumentPageText(page_number=index + 1, text=page_text))
    except Exception as exc:  # pragma: no cover - external parser exceptions vary
        raise ValueError(f"Unable to parse PDF file: {pdf_path.name}") from exc

    if not pages:
        raise ValueError(f"PDF has no readable pages: {pdf_path.name}")

    return pages


def extract_note_text(note_path: Path) -> list[DocumentPageText]:
    if note_path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError(f"Unsupported note file type: {note_path.name}")

    try:
        note_text = note_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Note file must be UTF-8 encoded: {note_path.name}") from exc

    text_value = note_text.strip()
    if not text_value:
        raise ValueError(f"Note file is empty: {note_path.name}")

    return [DocumentPageText(page_number=1, text=text_value)]


def parse_document_text(document_path: Path) -> list[DocumentPageText]:
    if document_path.suffix.lower() == ".pdf":
        return extract_pdf_text_with_pages(document_path)
    return extract_note_text(document_path)


def _split_paragraphs(text_value: str) -> list[str]:
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n+", text_value) if value.strip()]
    if paragraphs:
        return paragraphs
    return [value.strip() for value in text_value.splitlines() if value.strip()]


def _chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_chars = 0

    for paragraph in paragraphs:
        paragraph_chars = len(paragraph) + 1
        if current_parts and current_chars + paragraph_chars > RAW_CHUNK_TARGET_CHARS:
            chunks.append("\n\n".join(current_parts).strip())

            overlap_parts: list[str] = []
            overlap_chars = 0
            for prior_paragraph in reversed(current_parts):
                overlap_parts.insert(0, prior_paragraph)
                overlap_chars += len(prior_paragraph) + 1
                if overlap_chars >= RAW_CHUNK_OVERLAP_CHARS:
                    break

            current_parts = overlap_parts
            current_chars = sum(len(value) + 1 for value in current_parts)

        current_parts.append(paragraph)
        current_chars += paragraph_chars

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def build_transcript_chunks(transcript: TranscriptPayload) -> list[dict[str, float | int | str | None]]:
    chunks: list[dict[str, float | int | str | None]] = []
    current_segments: list[TranscriptSegment] = []
    current_chars = 0

    for segment in transcript.segments:
        segment_text = segment.text.strip()
        if not segment_text:
            continue

        segment_chars = len(segment_text) + 1
        if current_segments and current_chars + segment_chars > RAW_CHUNK_TARGET_CHARS:
            chunk_text = " ".join(
                value.text.strip() for value in current_segments if value.text.strip()
            ).strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_type": "transcript",
                        "chunk_text": chunk_text,
                        "timestamp_start": float(current_segments[0].start),
                        "timestamp_end": float(current_segments[-1].end),
                        "page_number": None,
                    }
                )

            overlap_segments: list[TranscriptSegment] = []
            overlap_chars = 0
            for prior_segment in reversed(current_segments):
                prior_text = prior_segment.text.strip()
                if not prior_text:
                    continue
                overlap_segments.insert(0, prior_segment)
                overlap_chars += len(prior_text) + 1
                if overlap_chars >= RAW_CHUNK_OVERLAP_CHARS:
                    break

            current_segments = overlap_segments
            current_chars = sum(
                len(value.text.strip()) + 1 for value in current_segments if value.text.strip()
            )

        current_segments.append(segment)
        current_chars += segment_chars

    if current_segments:
        chunk_text = " ".join(
            value.text.strip() for value in current_segments if value.text.strip()
        ).strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_type": "transcript",
                    "chunk_text": chunk_text,
                    "timestamp_start": float(current_segments[0].start),
                    "timestamp_end": float(current_segments[-1].end),
                    "page_number": None,
                }
            )

    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_index"] = index

    return chunks


def build_document_chunks(pages: list[DocumentPageText]) -> list[dict[str, float | int | str | None]]:
    chunks: list[dict[str, float | int | str | None]] = []

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        page_chunks = _chunk_paragraphs(_split_paragraphs(page_text))
        for page_chunk in page_chunks:
            chunks.append(
                {
                    "chunk_type": "document",
                    "chunk_text": page_chunk,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "page_number": int(page.page_number),
                }
            )

    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_index"] = index

    return chunks


def _insert_source_row(
    connection: Connection,
    user_id: str,
    source_type: str,
    filename: str,
    mime_type: str | None,
) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO sources (user_id, source_type, filename, mime_type)
            VALUES (:user_id, :source_type, :filename, :mime_type)
            """
        ),
        {
            "user_id": user_id,
            "source_type": source_type,
            "filename": filename,
            "mime_type": mime_type,
        },
    )

    source_id = result.lastrowid
    if source_id is None:
        raise RuntimeError("Failed to create source metadata record.")

    return int(source_id)


def _store_source_chunks(
    connection: Connection,
    user_id: str,
    source_id: int,
    chunks: list[dict[str, float | int | str | None]],
) -> None:
    for chunk in chunks:
        connection.execute(
            text(
                """
                INSERT INTO source_text_chunks (
                    user_id,
                    source_id,
                    chunk_index,
                    chunk_type,
                    chunk_text,
                    timestamp_start,
                    timestamp_end,
                    page_number
                ) VALUES (
                    :user_id,
                    :source_id,
                    :chunk_index,
                    :chunk_type,
                    :chunk_text,
                    :timestamp_start,
                    :timestamp_end,
                    :page_number
                )
                ON CONFLICT(user_id, source_id, chunk_type, chunk_index) DO UPDATE SET
                    chunk_text = excluded.chunk_text,
                    timestamp_start = excluded.timestamp_start,
                    timestamp_end = excluded.timestamp_end,
                    page_number = excluded.page_number
                """
            ),
            {
                "user_id": user_id,
                "source_id": source_id,
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_type": str(chunk["chunk_type"]),
                "chunk_text": str(chunk["chunk_text"]),
                "timestamp_start": chunk["timestamp_start"],
                "timestamp_end": chunk["timestamp_end"],
                "page_number": chunk["page_number"],
            },
        )


def store_video_transcript(
    user_id: str,
    filename: str,
    mime_type: str | None,
    transcript: TranscriptPayload,
) -> VideoIngestionRecord:
    transcript_chunks = build_transcript_chunks(transcript)
    segments_json = json.dumps(
        [segment.model_dump() for segment in transcript.segments],
        ensure_ascii=True,
    )

    with db_engine.begin() as connection:
        source_id = _insert_source_row(connection, user_id, "video", filename, mime_type)
        connection.execute(
            text(
                """
                INSERT INTO transcripts (source_id, transcript_text, segments_json)
                VALUES (:source_id, :transcript_text, :segments_json)
                """
            ),
            {
                "source_id": source_id,
                "transcript_text": TRANSCRIPT_TEXT_PLACEHOLDER,
                "segments_json": segments_json,
            },
        )
        _store_source_chunks(
            connection=connection,
            user_id=user_id,
            source_id=source_id,
            chunks=transcript_chunks,
        )

    return VideoIngestionRecord(
        source_id=source_id,
        filename=filename,
        mime_type=mime_type,
        transcript=transcript,
    )


def store_document_pages(
    user_id: str,
    filename: str,
    mime_type: str | None,
    pages: list[DocumentPageText],
) -> DocumentIngestionRecord:
    document_chunks = build_document_chunks(pages)
    with db_engine.begin() as connection:
        source_id = _insert_source_row(connection, user_id, "document", filename, mime_type)
        for page in pages:
            connection.execute(
                text(
                    """
                    INSERT INTO document_pages (source_id, page_number, page_text)
                    VALUES (:source_id, :page_number, :page_text)
                    """
                ),
                {
                    "source_id": source_id,
                    "page_number": page.page_number,
                    "page_text": page.text,
                },
            )
        _store_source_chunks(
            connection=connection,
            user_id=user_id,
            source_id=source_id,
            chunks=document_chunks,
        )

    return DocumentIngestionRecord(
        source_id=source_id,
        filename=filename,
        mime_type=mime_type,
        pages=pages,
    )


async def ingest_upload_bundle(
    user_id: str,
    video_file: UploadFile,
    document_files: list[UploadFile],
) -> IngestionResponse:
    ensure_ingestion_tables()
    validate_video_file(video_file)
    validate_document_files(document_files)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    run_root = UPLOAD_ROOT / run_id
    video_dir = run_root / "video"
    documents_dir = run_root / "documents"
    audio_dir = run_root / "audio"

    # 1) Upload save
    video_path = save_upload_file(video_file, video_dir)
    saved_documents: list[tuple[UploadFile, Path]] = []
    for document_file in document_files:
        document_path = save_upload_file(document_file, documents_dir)
        saved_documents.append((document_file, document_path))

    # 2) Audio extraction
    audio_path = audio_dir / f"{video_path.stem}.wav"
    extract_audio_from_video(video_path, audio_path)

    # 3) Transcription
    transcript = transcribe_audio_with_whisper(audio_path, user_id)

    # 4) PDF or note parsing
    parsed_documents: list[tuple[UploadFile, list[DocumentPageText]]] = []
    for document_file, document_path in saved_documents:
        pages = parse_document_text(document_path)
        parsed_documents.append((document_file, pages))

    # 5) Storage
    video_record = store_video_transcript(
        user_id=user_id,
        filename=Path(video_file.filename or video_path.name).name,
        mime_type=video_file.content_type,
        transcript=transcript,
    )

    document_records: list[DocumentIngestionRecord] = []
    for document_file, pages in parsed_documents:
        document_record = store_document_pages(
            user_id=user_id,
            filename=Path(document_file.filename or "document").name,
            mime_type=document_file.content_type,
            pages=pages,
        )
        document_records.append(document_record)

    return IngestionResponse(
        message="Ingestion completed successfully.",
        request=UploadRequestMeta(
            video_filename=Path(video_file.filename or video_path.name).name,
            document_filenames=[
                Path(document_file.filename or "document").name
                for document_file in document_files
            ],
        ),
        video=video_record,
        documents=document_records,
    )
