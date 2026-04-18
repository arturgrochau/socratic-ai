from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
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
from config import (
    INGESTION_VIDEO_STEP_TIMEOUT_SECONDS,
    WHISPER_TRANSCRIPTION_MAX_RETRIES,
    WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS,
    db_engine,
    openai_client,
)


UPLOAD_ROOT = Path("uploads")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
SUPPORTED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "www.youtu.be",
}
RAW_CHUNK_TARGET_CHARS = 2400
RAW_CHUNK_OVERLAP_CHARS = 320
TRANSCRIPT_TEXT_PLACEHOLDER = "[chunked transcript stored in source_text_chunks]"
WHISPER_MAX_REQUEST_BYTES = 24 * 1024 * 1024
WHISPER_CHUNK_SECONDS = 540


logger = logging.getLogger(__name__)


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
    for document in document_files:
        if not document.filename:
            raise ValueError("Each document must include a filename.")

        suffix = Path(document.filename).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise ValueError(
                f"Unsupported document file type for {document.filename}. "
                "Allowed: .pdf, .txt, .md"
            )


def validate_youtube_url(video_url: str) -> str:
    raw = video_url.strip()
    if not raw:
        raise ValueError("YouTube URL cannot be empty.")

    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
    except Exception as exc:
        raise ValueError("Only single-video YouTube URLs are supported.") from exc

    host = (parsed.netloc or "").lower()
    if host.startswith("www.") and host not in {"www.youtube.com", "www.youtube-nocookie.com"}:
        host = host[4:]
    if host not in SUPPORTED_YOUTUBE_HOSTS:
        raise ValueError("Only single-video YouTube URLs are supported.")

    path = (parsed.path or "").strip()
    if host == "youtu.be":
        if path.strip("/"):
            return candidate
        raise ValueError("Only single-video YouTube URLs are supported.")

    normalized_path = path.rstrip("/")
    if normalized_path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [""])[0].strip()
        if video_id:
            return candidate
        raise ValueError("Only single-video YouTube URLs are supported.")

    for prefix in ("/shorts/", "/embed/", "/live/"):
        if normalized_path.startswith(prefix):
            tail = normalized_path[len(prefix):].strip("/")
            if tail:
                return candidate

    raise ValueError("Only single-video YouTube URLs are supported.")


def _safe_filename(raw_value: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value).strip("._-")
    if not safe:
        return fallback
    return safe[:120]


def _normalize_downloaded_video_path(
    *,
    downloaded_path: Path,
    source_title: str,
) -> tuple[Path, str, str | None]:
    if not downloaded_path.exists() or downloaded_path.stat().st_size == 0:
        raise RuntimeError("YouTube download produced an empty media file.")

    suffix = downloaded_path.suffix or ".mp4"
    target_name = f"{_safe_filename(source_title, 'youtube_video')}{suffix}"
    normalized_path = downloaded_path.with_name(target_name)

    if downloaded_path != normalized_path:
        candidate_path = normalized_path
        collision_index = 1
        while candidate_path.exists():
            candidate_path = normalized_path.with_name(
                f"{normalized_path.stem}_{collision_index}{normalized_path.suffix}"
            )
            collision_index += 1
        downloaded_path.rename(candidate_path)
        normalized_path = candidate_path

    mime_type = mimetypes.guess_type(normalized_path.name)[0]
    return normalized_path, normalized_path.name, mime_type


def _download_youtube_video_with_cli(video_url: str, destination_dir: Path) -> tuple[Path, str, str | None]:
    yt_dlp_binary = shutil.which("yt-dlp")
    if not yt_dlp_binary:
        raise RuntimeError(
            "yt-dlp is required for YouTube ingestion. Install it with '.venv/bin/pip install -r requirements.txt'."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(destination_dir / "%(id)s.%(ext)s")

    try:
        metadata_result = subprocess.run(
            [
                yt_dlp_binary,
                "--no-playlist",
                "--quiet",
                "--dump-single-json",
                video_url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(metadata_result.stdout)
    except Exception as exc:
        raise RuntimeError(f"YouTube metadata lookup failed: {exc}") from exc

    try:
        subprocess.run(
            [
                yt_dlp_binary,
                "-f",
                "bestaudio/best",
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "-o",
                output_template,
                video_url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or "").strip()
        raise RuntimeError(f"YouTube download failed: {stderr_text or exc}") from exc

    video_id = str(metadata.get("id") or "").strip()
    source_title = str(metadata.get("title") or video_id or "youtube_video")
    downloaded_candidates = list(destination_dir.glob(f"{video_id}.*")) if video_id else []
    if not downloaded_candidates:
        downloaded_candidates = sorted(
            destination_dir.glob("*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not downloaded_candidates:
        raise RuntimeError("YouTube download produced no output file.")

    return _normalize_downloaded_video_path(
        downloaded_path=downloaded_candidates[0],
        source_title=source_title,
    )


def download_youtube_video(video_url: str, destination_dir: Path) -> tuple[Path, str, str | None]:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        # Fallback for environments where the CLI exists but Python module is unavailable.
        return _download_youtube_video_with_cli(video_url, destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(destination_dir / "%(id)s.%(ext)s")
    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_options) as ydl:
            metadata = ydl.extract_info(video_url, download=True)
            if not isinstance(metadata, dict):
                raise RuntimeError("yt-dlp did not return video metadata.")

            if metadata.get("entries"):
                first_entry = metadata["entries"][0]
                if isinstance(first_entry, dict):
                    metadata = first_entry

            downloaded_path = Path(ydl.prepare_filename(metadata))
    except Exception as exc:  # pragma: no cover - external downloader exceptions vary
        raise RuntimeError(f"YouTube download failed: {exc}") from exc

    source_title = str(metadata.get("title") or metadata.get("id") or "youtube_video")
    return _normalize_downloaded_video_path(
        downloaded_path=downloaded_path,
        source_title=source_title,
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


def _probe_audio_duration_seconds(audio_path: Path) -> float:
    try:
        probe_data = ffmpeg.probe(str(audio_path))
    except ffmpeg.Error:
        return 0.0

    format_data = probe_data.get("format", {}) if isinstance(probe_data, dict) else {}
    raw_duration = format_data.get("duration")
    try:
        return max(0.0, float(raw_duration or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _parse_whisper_transcription(
    transcription: object,
    *,
    segment_offset_seconds: float,
) -> TranscriptPayload:
    transcript_text = getattr(transcription, "text", None)
    segments_raw = getattr(transcription, "segments", None)

    if isinstance(transcription, dict):
        if transcript_text is None:
            transcript_text = transcription.get("text")
        if segments_raw is None:
            segments_raw = transcription.get("segments")

    text_value = (transcript_text or "").strip()

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
            TranscriptSegment(
                start=start_value + segment_offset_seconds,
                end=end_value + segment_offset_seconds,
                text=segment_text,
            )
        )

    if not text_value and segments:
        text_value = " ".join(segment.text for segment in segments if segment.text.strip()).strip()

    if not text_value:
        raise RuntimeError("Transcription returned empty text.")

    if not segments:
        raise RuntimeError("Transcription did not include segment timestamps.")

    return TranscriptPayload(text=text_value, segments=segments)


def _transcribe_whisper_file(
    *,
    audio_path: Path,
    user_id: str,
    segment_offset_seconds: float,
) -> TranscriptPayload:
    last_error: Exception | None = None
    transcription: object | None = None
    for attempt in range(1, WHISPER_TRANSCRIPTION_MAX_RETRIES + 1):
        try:
            with audio_path.open("rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    timeout=WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS,
                )
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Whisper attempt %s/%s failed for %s: %s",
                attempt,
                WHISPER_TRANSCRIPTION_MAX_RETRIES,
                audio_path.name,
                exc,
            )
            if attempt >= WHISPER_TRANSCRIPTION_MAX_RETRIES:
                raise RuntimeError(
                    "Whisper transcription failed after "
                    f"{WHISPER_TRANSCRIPTION_MAX_RETRIES} attempt(s): {exc}"
                ) from exc
            time.sleep(min(2.0, 0.4 * attempt))

    if transcription is None:
        raise RuntimeError(f"Whisper transcription failed: {last_error}") from last_error

    log_api_usage(
        response=transcription,
        user_id=user_id,
        call_stage="ingestion",
        model_name="whisper-1",
    )
    return _parse_whisper_transcription(
        transcription,
        segment_offset_seconds=segment_offset_seconds,
    )


def transcribe_audio_with_whisper(audio_path: Path, user_id: str) -> TranscriptPayload:
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("Audio file is missing or empty.")

    audio_size = audio_path.stat().st_size
    if audio_size <= WHISPER_MAX_REQUEST_BYTES:
        return _transcribe_whisper_file(
            audio_path=audio_path,
            user_id=user_id,
            segment_offset_seconds=0.0,
        )

    with tempfile.TemporaryDirectory(prefix="whisper_chunks_") as tmp_dir:
        chunk_dir = Path(tmp_dir)
        chunk_pattern = str(chunk_dir / "chunk_%03d.wav")
        try:
            (
                ffmpeg.input(str(audio_path))
                .output(
                    chunk_pattern,
                    f="segment",
                    segment_format="wav",
                    segment_time=WHISPER_CHUNK_SECONDS,
                    reset_timestamps=1,
                    ac=1,
                    ar=16000,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else "Unknown ffmpeg error"
            raise RuntimeError(f"Audio chunking failed: {stderr}") from exc

        chunk_paths = sorted(chunk_dir.glob("chunk_*.wav"))
        if not chunk_paths:
            raise RuntimeError("Audio chunking produced no chunks for transcription.")

        all_segments: list[TranscriptSegment] = []
        text_parts: list[str] = []
        running_offset_seconds = 0.0

        for chunk_path in chunk_paths:
            if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                continue

            chunk_payload = _transcribe_whisper_file(
                audio_path=chunk_path,
                user_id=user_id,
                segment_offset_seconds=running_offset_seconds,
            )
            if chunk_payload.text.strip():
                text_parts.append(chunk_payload.text.strip())
            all_segments.extend(chunk_payload.segments)

            chunk_duration = _probe_audio_duration_seconds(chunk_path)
            if chunk_duration <= 0.0 and chunk_payload.segments:
                max_chunk_end = max(segment.end for segment in chunk_payload.segments)
                chunk_duration = max(0.0, max_chunk_end - running_offset_seconds)
            if chunk_duration <= 0.0:
                chunk_duration = float(WHISPER_CHUNK_SECONDS)

            running_offset_seconds += chunk_duration

    if not all_segments:
        raise RuntimeError("Transcription did not include segment timestamps.")

    combined_text = " ".join(text_parts).strip()
    if not combined_text:
        combined_text = " ".join(segment.text for segment in all_segments if segment.text.strip()).strip()
    if not combined_text:
        raise RuntimeError("Transcription returned empty text.")

    return TranscriptPayload(text=combined_text, segments=all_segments)


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
    video_file: UploadFile | None,
    video_url: str | None,
    document_files: list[UploadFile],
) -> IngestionResponse:
    ensure_ingestion_tables()

    normalized_documents = document_files or []
    if video_file is not None and video_url and video_url.strip():
        raise ValueError("Provide either a video file or a YouTube URL, not both.")

    if video_file is None and not (video_url and video_url.strip()) and not normalized_documents:
        raise ValueError("Provide at least one source: video upload, YouTube URL, or document.")

    if video_file is not None:
        validate_video_file(video_file)
    normalized_video_url = validate_youtube_url(video_url) if video_url and video_url.strip() else None
    if normalized_documents:
        validate_document_files(normalized_documents)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    run_root = UPLOAD_ROOT / run_id
    video_dir = run_root / "video"
    documents_dir = run_root / "documents"
    audio_dir = run_root / "audio"

    # 1) Upload save / URL download
    video_path: Path | None = None
    video_display_name: str | None = None
    video_mime_type: str | None = None

    if video_file is not None:
        video_path = save_upload_file(video_file, video_dir)
        video_display_name = Path(video_file.filename or video_path.name).name
        video_mime_type = video_file.content_type
    elif normalized_video_url is not None:
        video_path, video_display_name, video_mime_type = download_youtube_video(
            normalized_video_url,
            video_dir,
        )

    saved_documents: list[tuple[UploadFile, Path]] = []
    for document_file in normalized_documents:
        document_path = save_upload_file(document_file, documents_dir)
        saved_documents.append((document_file, document_path))

    # 2) Audio extraction + 3) Transcription
    transcript: TranscriptPayload | None = None
    if video_path is not None:
        audio_path = audio_dir / f"{video_path.stem}.wav"
        try:
            start_time = time.perf_counter()
            await asyncio.wait_for(
                asyncio.to_thread(extract_audio_from_video, video_path, audio_path),
                timeout=INGESTION_VIDEO_STEP_TIMEOUT_SECONDS,
            )
            extraction_seconds = time.perf_counter() - start_time
            logger.info(
                "Video audio extraction completed in %.2fs for %s",
                extraction_seconds,
                video_path.name,
            )

            start_time = time.perf_counter()
            transcript = await asyncio.wait_for(
                asyncio.to_thread(transcribe_audio_with_whisper, audio_path, user_id),
                timeout=INGESTION_VIDEO_STEP_TIMEOUT_SECONDS,
            )
            transcription_seconds = time.perf_counter() - start_time
            logger.info(
                "Whisper transcription completed in %.2fs for %s",
                transcription_seconds,
                audio_path.name,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Video ingestion exceeded timeout budget during extraction/transcription. "
                f"Increase INGESTION_VIDEO_STEP_TIMEOUT_SECONDS (current={INGESTION_VIDEO_STEP_TIMEOUT_SECONDS}) "
                "or use documents-only mode for faster validation."
            ) from exc

    # 4) PDF or note parsing
    parsed_documents: list[tuple[UploadFile, list[DocumentPageText]]] = []
    for document_file, document_path in saved_documents:
        pages = parse_document_text(document_path)
        parsed_documents.append((document_file, pages))

    # 5) Storage
    video_record: VideoIngestionRecord | None = None
    if transcript is not None and video_display_name is not None:
        video_record = store_video_transcript(
            user_id=user_id,
            filename=video_display_name,
            mime_type=video_mime_type,
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
            video_filename=video_display_name,
            video_url=normalized_video_url,
            document_filenames=[
                Path(document_file.filename or "document").name
                for document_file in normalized_documents
            ],
        ),
        video=video_record,
        documents=document_records,
    )
