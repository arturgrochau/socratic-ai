from __future__ import annotations

import json
from typing import Literal

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.models import ConceptExtractionPayload, ProcessSourceResponse
from config import CACHE_PROCESSED_SOURCES, PROCESSING_MODEL, db_engine, openai_client
from prompts.concept_extraction import (
    CONCEPT_EXTRACTION_JSON_SCHEMA,
    CONCEPT_EXTRACTION_SYSTEM_PROMPT,
)
from prompts.source_summary import SOURCE_SUMMARY_SYSTEM_PROMPT


TRANSCRIPT_CHUNK_SIZE = 3200
DOCUMENT_CHUNK_SIZE = 2800
CHUNK_OVERLAP = 250


def ensure_processing_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS concept_extractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_id INTEGER NOT NULL UNIQUE,
                    source_type TEXT NOT NULL CHECK (source_type IN ('video', 'document')),
                    model_name TEXT NOT NULL,
                    extraction_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
                """
            )
        )
        extraction_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(concept_extractions)")).fetchall()
        }
        if "user_id" not in extraction_columns:
            connection.execute(
                text("ALTER TABLE concept_extractions ADD COLUMN user_id TEXT DEFAULT 'legacy'")
            )
            connection.execute(
                text(
                    """
                    UPDATE concept_extractions
                    SET user_id = (
                        SELECT sources.user_id FROM sources WHERE sources.id = concept_extractions.source_id
                    )
                    WHERE user_id IS NULL
                    """
                )
            )
            connection.execute(
                text(
                    """
                    UPDATE concept_extractions
                    SET user_id = 'legacy'
                    WHERE user_id IS NULL OR user_id = ''
                    """
                )
            )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_concept_extractions_user_id
                ON concept_extractions (user_id)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS source_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_id INTEGER NOT NULL UNIQUE,
                    source_type TEXT NOT NULL CHECK (source_type IN ('video', 'document')),
                    model_name TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
                """
            )
        )
        summary_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(source_summaries)")).fetchall()
        }
        if "user_id" not in summary_columns:
            connection.execute(
                text("ALTER TABLE source_summaries ADD COLUMN user_id TEXT DEFAULT 'legacy'")
            )
            connection.execute(
                text(
                    """
                    UPDATE source_summaries
                    SET user_id = (
                        SELECT sources.user_id FROM sources WHERE sources.id = source_summaries.source_id
                    )
                    WHERE user_id IS NULL
                    """
                )
            )
            connection.execute(
                text(
                    """
                    UPDATE source_summaries
                    SET user_id = 'legacy'
                    WHERE user_id IS NULL OR user_id = ''
                    """
                )
            )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_source_summaries_user_id
                ON source_summaries (user_id)
                """
            )
        )


def get_source_metadata(
    source_id: int,
    user_id: str,
) -> tuple[Literal["video", "document"], str, str | None]:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT source_type, filename, mime_type
                FROM sources
                WHERE id = :source_id AND user_id = :user_id
                """
            ),
            {"source_id": source_id, "user_id": user_id},
        ).mappings().first()

    if row is None:
        raise ValueError(f"Source {source_id} was not found.")

    source_type = row["source_type"]
    if source_type not in {"video", "document"}:
        raise ValueError(f"Unsupported source type for source {source_id}: {source_type}")

    return source_type, row["filename"], row["mime_type"]


def load_transcript_text(source_id: int, user_id: str) -> str:
    get_source_metadata(source_id, user_id)
    with db_engine.connect() as connection:
        chunk_rows = connection.execute(
            text(
                """
                SELECT chunk_text
                FROM source_text_chunks
                WHERE source_id = :source_id
                  AND user_id = :user_id
                  AND chunk_type = 'transcript'
                ORDER BY chunk_index ASC
                """
            ),
            {"source_id": source_id, "user_id": user_id},
        ).mappings().all()

    if chunk_rows:
        transcript_text = "\n\n".join(
            str(row["chunk_text"]).strip() for row in chunk_rows if str(row["chunk_text"]).strip()
        ).strip()
        if transcript_text:
            return transcript_text

    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT transcript_text
                FROM transcripts
                WHERE source_id = :source_id
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"source_id": source_id},
        ).mappings().first()

    if row is None:
        raise ValueError(f"No transcript found for source {source_id}.")

    transcript_text = (row["transcript_text"] or "").strip()
    if not transcript_text:
        raise ValueError(f"Transcript text is empty for source {source_id}.")

    return transcript_text


def load_document_text(source_id: int, user_id: str) -> str:
    get_source_metadata(source_id, user_id)
    with db_engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT page_number, page_text
                FROM document_pages
                WHERE source_id = :source_id
                ORDER BY page_number ASC, id ASC
                """
            ),
            {"source_id": source_id},
        ).mappings().all()

    if not rows:
        raise ValueError(f"No document pages found for source {source_id}.")

    page_blocks: list[str] = []
    for row in rows:
        page_text = str(row["page_text"] or "").strip()
        page_number = int(row["page_number"])
        if page_text:
            page_blocks.append(f"Page {page_number}\n{page_text}")

    if not page_blocks:
        raise ValueError(f"Document text is empty for source {source_id}.")

    return "\n\n".join(page_blocks)


def _chunk_text(text_value: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = text_value.strip()
    if not normalized:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            split_at = normalized.rfind("\n", start, end)
            if split_at == -1:
                split_at = normalized.rfind(" ", start, end)
            if split_at > start + (chunk_size // 2):
                end = split_at

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_transcript_text(transcript_text: str) -> list[str]:
    return _chunk_text(
        text_value=transcript_text,
        chunk_size=TRANSCRIPT_CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )


def chunk_document_text(document_text: str) -> list[str]:
    return _chunk_text(
        text_value=document_text,
        chunk_size=DOCUMENT_CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )


def build_source_context(source_type: str, chunks: list[str]) -> str:
    if not chunks:
        raise ValueError("No chunks were generated for extraction.")

    chunk_blocks = [f"[Chunk {index}]\n{value}" for index, value in enumerate(chunks, start=1)]
    return f"Source Type: {source_type}\n\n" + "\n\n".join(chunk_blocks)


def extract_concepts_once(source_context: str, user_id: str) -> ConceptExtractionPayload:
    completion = openai_client.chat.completions.create(
        model=PROCESSING_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": CONCEPT_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract concepts from this source context and return schema-valid JSON.\n\n"
                    f"{source_context}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": CONCEPT_EXTRACTION_JSON_SCHEMA,
        },
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="processing",
        model_name=PROCESSING_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty extraction content.")

    return ConceptExtractionPayload.model_validate_json(content)


def generate_source_summary(
    source_type: Literal["video", "document"],
    source_context: str,
    user_id: str,
) -> str:
    completion = openai_client.chat.completions.create(
        model=PROCESSING_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SOURCE_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source type: {source_type}\n\n"
                    "Write a concise source summary from this context:\n\n"
                    f"{source_context}"
                ),
            },
        ],
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="processing",
        model_name=PROCESSING_MODEL,
    )

    content = completion.choices[0].message.content
    summary_text = (content or "").strip()
    if not summary_text:
        raise ValueError("Model returned empty source summary content.")
    return summary_text


def store_concept_extraction(
    source_id: int,
    source_type: Literal["video", "document"],
    extraction_payload: ConceptExtractionPayload,
    user_id: str,
) -> None:
    serialized_payload = json.dumps(
        extraction_payload.model_dump(),
        ensure_ascii=True,
        sort_keys=True,
    )

    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO concept_extractions (
                    user_id,
                    source_id,
                    source_type,
                    model_name,
                    extraction_json
                ) VALUES (
                    :user_id,
                    :source_id,
                    :source_type,
                    :model_name,
                    :extraction_json
                )
                ON CONFLICT(source_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    source_type = excluded.source_type,
                    model_name = excluded.model_name,
                    extraction_json = excluded.extraction_json,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "source_id": source_id,
                "source_type": source_type,
                "model_name": PROCESSING_MODEL,
                "extraction_json": serialized_payload,
            },
        )


def store_source_summary(
    source_id: int,
    source_type: Literal["video", "document"],
    summary_text: str,
    user_id: str,
) -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO source_summaries (
                    user_id,
                    source_id,
                    source_type,
                    model_name,
                    summary_text
                ) VALUES (
                    :user_id,
                    :source_id,
                    :source_type,
                    :model_name,
                    :summary_text
                )
                ON CONFLICT(source_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    source_type = excluded.source_type,
                    model_name = excluded.model_name,
                    summary_text = excluded.summary_text,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "source_id": source_id,
                "source_type": source_type,
                "model_name": PROCESSING_MODEL,
                "summary_text": summary_text,
            },
        )


def load_existing_concept_extraction(
    source_id: int,
    user_id: str,
) -> ConceptExtractionPayload | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT extraction_json
                FROM concept_extractions
                WHERE source_id = :source_id AND user_id = :user_id
                """
            ),
            {"source_id": source_id, "user_id": user_id},
        ).mappings().first()

    if row is None:
        return None
    return ConceptExtractionPayload.model_validate_json(str(row["extraction_json"]))


def load_existing_source_summary(source_id: int, user_id: str) -> str | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT summary_text
                FROM source_summaries
                WHERE source_id = :source_id AND user_id = :user_id
                """
            ),
            {"source_id": source_id, "user_id": user_id},
        ).mappings().first()

    if row is None:
        return None
    summary_text = str(row["summary_text"] or "").strip()
    return summary_text or None


def has_concept_extraction(source_id: int, user_id: str) -> bool:
    return load_existing_concept_extraction(source_id, user_id) is not None


def process_source(source_id: int, user_id: str) -> ProcessSourceResponse:
    ensure_processing_tables()

    source_type, _, _ = get_source_metadata(source_id, user_id)

    existing_payload = load_existing_concept_extraction(source_id, user_id)
    existing_summary = load_existing_source_summary(source_id, user_id)

    if CACHE_PROCESSED_SOURCES:
        if existing_payload is not None and existing_summary is not None:
            return ProcessSourceResponse(
                source_id=source_id,
                source_type=source_type,
                model_name=PROCESSING_MODEL,
                chunk_count=0,
                extraction=existing_payload,
                source_summary=existing_summary,
            )

    if source_type == "video":
        raw_text = load_transcript_text(source_id, user_id)
        chunks = chunk_transcript_text(raw_text)
    else:
        raw_text = load_document_text(source_id, user_id)
        chunks = chunk_document_text(raw_text)

    source_context = build_source_context(source_type, chunks)
    extraction_payload = existing_payload or extract_concepts_once(source_context, user_id)
    summary_text = existing_summary or generate_source_summary(source_type, source_context, user_id)

    if existing_payload is None:
        store_concept_extraction(
            source_id=source_id,
            source_type=source_type,
            extraction_payload=extraction_payload,
            user_id=user_id,
        )
    if existing_summary is None:
        store_source_summary(
            source_id=source_id,
            source_type=source_type,
            summary_text=summary_text,
            user_id=user_id,
        )

    return ProcessSourceResponse(
        source_id=source_id,
        source_type=source_type,
        model_name=PROCESSING_MODEL,
        chunk_count=len(chunks),
        extraction=extraction_payload,
        source_summary=summary_text,
    )
