from __future__ import annotations

import json
import time
import uuid
from typing import Any

from sqlalchemy import text

from app.cost_logging import log_api_usage, log_generation_stage_event
from app.json_reliability import parse_json_object, safe_json_loads
from app.linking import link_source_pair
from app.models import (
    ApplicationScenario,
    AttributedSentence,
    CombinedInsightSection,
    CombinedQuizSection,
    GenerateTailoredLearningResponse,
    InsightIntersection,
    KeyTermExplanation,
    QuizQuestion,
    ReflectionPoint,
    SourceLearningSection,
)
from app.processing import get_source_metadata, load_existing_source_summary, process_source
from config import CACHE_PROCESSED_SOURCES, GENERATION_MODEL, db_engine, openai_client
from prompts.combined_insights import (
    COMBINED_INSIGHTS_JSON_SCHEMA,
    COMBINED_INSIGHTS_SYSTEM_PROMPT,
)
from prompts.combined_quiz import COMBINED_QUIZ_JSON_SCHEMA, COMBINED_QUIZ_SYSTEM_PROMPT
from prompts.comparative_analysis import (
    COMPARATIVE_ANALYSIS_JSON_SCHEMA,
    COMPARATIVE_ANALYSIS_SYSTEM_PROMPT,
)
from prompts.application_scenarios import (
    APPLICATION_SCENARIOS_JSON_SCHEMA,
    APPLICATION_SCENARIOS_SYSTEM_PROMPT,
)
from prompts.socratic_reflection import (
    SOCRATIC_REFLECTION_JSON_SCHEMA,
    SOCRATIC_REFLECTION_SYSTEM_PROMPT,
)
from prompts.source_deep_dive import (
    SOURCE_DEEP_DIVE_JSON_SCHEMA,
    SOURCE_DEEP_DIVE_SYSTEM_PROMPT,
)
from prompts.source_title import SOURCE_TITLE_JSON_SCHEMA, SOURCE_TITLE_SYSTEM_PROMPT
from prompts.under_surface import UNDER_SURFACE_JSON_SCHEMA, UNDER_SURFACE_SYSTEM_PROMPT


GENERATION_SCHEMA_VERSION = 5
MAX_GROUNDING_CHUNKS = 10
MAX_GROUNDING_CHARS = 9000
GENERATION_MAX_STAGE_ATTEMPTS = 3
MAX_STAGE_ERROR_CHARS = 700


class GenerationStageError(RuntimeError):
    def __init__(
        self,
        *,
        run_id: str,
        stage_name: str,
        attempt_number: int,
        reason: str,
    ) -> None:
        self.run_id = run_id
        self.stage_name = stage_name
        self.attempt_number = attempt_number
        self.reason = reason
        super().__init__(
            f"Generation failed at stage '{stage_name}' for run_id={run_id} "
            f"after attempt {attempt_number}: {reason}"
        )


def _trim_error_detail(error_text: str) -> str:
    compact = " ".join(str(error_text).split())
    if len(compact) <= MAX_STAGE_ERROR_CHARS:
        return compact
    return compact[: MAX_STAGE_ERROR_CHARS - 3].rstrip() + "..."


def _run_structured_generation_step(
    *,
    run_id: str,
    stage_name: str,
    user_id: str,
    user_prompt: str,
    system_prompt: str,
    response_schema: dict[str, Any],
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    for attempt_number in range(1, GENERATION_MAX_STAGE_ATTEMPTS + 1):
        started = time.perf_counter()
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=stage_name,
            attempt_number=attempt_number,
            status="started",
        )

        try:
            completion = openai_client.chat.completions.create(
                model=GENERATION_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_schema", "json_schema": response_schema},
            )
            log_api_usage(
                response=completion,
                user_id=user_id,
                call_stage="generation",
                model_name=GENERATION_MODEL,
            )

            content = completion.choices[0].message.content
            payload = parse_json_object(content or "", stage_name=f"{stage_name} attempt {attempt_number}")

            if required_keys:
                missing_keys = [key for key in required_keys if key not in payload]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")

            duration_ms = int((time.perf_counter() - started) * 1000)
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=stage_name,
                attempt_number=attempt_number,
                status="succeeded",
                duration_ms=duration_ms,
            )
            return payload
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            error_detail = _trim_error_detail(str(exc))
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=stage_name,
                attempt_number=attempt_number,
                status="failed",
                duration_ms=duration_ms,
                error_message=error_detail,
            )
            if attempt_number >= GENERATION_MAX_STAGE_ATTEMPTS:
                raise GenerationStageError(
                    run_id=run_id,
                    stage_name=stage_name,
                    attempt_number=attempt_number,
                    reason=error_detail,
                ) from exc

    raise GenerationStageError(
        run_id=run_id,
        stage_name=stage_name,
        attempt_number=GENERATION_MAX_STAGE_ATTEMPTS,
        reason="Retry budget exhausted.",
    )


def ensure_generation_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS source_learning_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL CHECK (source_type IN ('video', 'document')),
                    generated_title TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    reflection_points_json TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, source_id),
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
                """
            )
        )
        source_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(source_learning_sections)")).fetchall()
        }
        if "deep_dive_text" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN deep_dive_text TEXT DEFAULT ''")
            )
        if "key_terms_json" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN key_terms_json TEXT DEFAULT '[]'")
            )
        if "schema_version" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN schema_version INTEGER DEFAULT 1")
            )
        if "source_name" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN source_name TEXT DEFAULT ''")
            )
        if "under_surface_text" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN under_surface_text TEXT DEFAULT ''")
            )
        if "diagnostic_checklist_json" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN diagnostic_checklist_json TEXT DEFAULT '[]'")
            )
        if "key_term_explanations_json" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN key_term_explanations_json TEXT DEFAULT '[]'")
            )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_source_learning_sections_user_source
                ON source_learning_sections (user_id, source_id)
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS combined_learning_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    video_source_id INTEGER NOT NULL,
                    document_source_ids_json TEXT NOT NULL,
                    parallels_json TEXT NOT NULL,
                    layman_bridge TEXT NOT NULL,
                    quiz_json TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, video_source_id, document_source_ids_json),
                    FOREIGN KEY (video_source_id) REFERENCES sources(id)
                )
                """
            )
        )
        combined_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(combined_learning_sections)")).fetchall()
        }
        if "synthesis_text" not in combined_columns:
            connection.execute(
                text("ALTER TABLE combined_learning_sections ADD COLUMN synthesis_text TEXT DEFAULT ''")
            )
        if "schema_version" not in combined_columns:
            connection.execute(
                text("ALTER TABLE combined_learning_sections ADD COLUMN schema_version INTEGER DEFAULT 1")
            )
        if "intersections_json" not in combined_columns:
            connection.execute(
                text("ALTER TABLE combined_learning_sections ADD COLUMN intersections_json TEXT DEFAULT '[]'")
            )
        if "comparative_analysis_text" not in combined_columns:
            connection.execute(
                text(
                    "ALTER TABLE combined_learning_sections ADD COLUMN comparative_analysis_text TEXT DEFAULT ''"
                )
            )
        if "application_scenarios_json" not in combined_columns:
            connection.execute(
                text(
                    "ALTER TABLE combined_learning_sections ADD COLUMN application_scenarios_json TEXT DEFAULT '[]'"
                )
            )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_combined_learning_sections_user_video
                ON combined_learning_sections (user_id, video_source_id)
                """
            )
        )


def _normalize_ids(video_source_id: int, document_source_ids: list[int]) -> tuple[int, list[int]]:
    if video_source_id <= 0:
        raise ValueError("video_source_id must be a positive integer.")

    normalized_document_ids = sorted(
        {source_id for source_id in document_source_ids if source_id > 0}
    )
    if not normalized_document_ids:
        raise ValueError("At least one valid document source_id is required.")

    return video_source_id, normalized_document_ids


def _serialize_ids(source_ids: list[int]) -> str:
    return json.dumps(source_ids, ensure_ascii=True, separators=(",", ":"))


def _load_relationship_insights(source_ids: list[int], user_id: str) -> list[str]:
    if len(source_ids) < 2:
        return []

    params: dict[str, int | str] = {"user_id": user_id}
    id_keys: list[str] = []
    for index, source_id in enumerate(source_ids):
        key = f"source_id_{index}"
        params[key] = source_id
        id_keys.append(f":{key}")

    in_clause = ", ".join(id_keys)
    query = text(
        f"""
        SELECT source_concept_id, target_concept_id, relation_type, confidence, explanation
        FROM concept_relationship_edges
        WHERE user_id = :user_id
          AND source_source_id IN ({in_clause})
          AND target_source_id IN ({in_clause})
        ORDER BY confidence DESC, updated_at DESC, id DESC
        LIMIT 18
        """
    )

    with db_engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return [
        (
            f"- {row['source_concept_id']} -> {row['target_concept_id']}: "
            f"{row['relation_type']} (confidence={float(row['confidence']):.2f}) | "
            f"{str(row['explanation'])}"
        )
        for row in rows
    ]


def _load_grounding_chunks(source_id: int, user_id: str) -> list[str]:
    with db_engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT chunk_type, chunk_index, chunk_text, timestamp_start, timestamp_end, page_number
                FROM source_text_chunks
                WHERE user_id = :user_id AND source_id = :source_id
                ORDER BY chunk_index ASC
                """
            ),
            {"user_id": user_id, "source_id": source_id},
        ).mappings().all()

    chunks: list[str] = []
    consumed_chars = 0
    for row in rows:
        chunk_text = str(row["chunk_text"] or "").strip()
        if not chunk_text:
            continue

        chunk_label = f"{row['chunk_type']} chunk {int(row['chunk_index'])}"
        if row["chunk_type"] == "transcript":
            start = row["timestamp_start"]
            end = row["timestamp_end"]
            if start is not None and end is not None:
                chunk_label += f" [{float(start):.1f}s-{float(end):.1f}s]"
        if row["chunk_type"] == "document":
            page_number = row["page_number"]
            if page_number is not None:
                chunk_label += f" [page {int(page_number)}]"

        entry = f"[{chunk_label}]\n{chunk_text}"
        if consumed_chars + len(entry) > MAX_GROUNDING_CHARS and chunks:
            break

        chunks.append(entry)
        consumed_chars += len(entry)
        if len(chunks) >= MAX_GROUNDING_CHUNKS:
            break

    return chunks


def _generate_source_title(summary_text: str, source_type: str, user_id: str, run_id: str) -> str:
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_title:{source_type}",
        user_id=user_id,
        system_prompt=SOURCE_TITLE_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            "Generate a concise title from this summary:\n\n"
            f"{summary_text}"
        ),
        response_schema=SOURCE_TITLE_JSON_SCHEMA,
        required_keys=["title"],
    )
    title = str(payload.get("title", "")).strip()
    if not title:
        raise ValueError("Model returned an empty source title.")
    return title


def _generate_source_deep_dive(
    summary_text: str,
    source_type: str,
    grounding_chunks: list[str],
    user_id: str,
    run_id: str,
) -> tuple[str, list[str]]:
    chunk_text = "\n\n".join(grounding_chunks) if grounding_chunks else "[No grounding chunks available]"
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_deep_dive:{source_type}",
        user_id=user_id,
        system_prompt=SOURCE_DEEP_DIVE_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Source summary:\n{summary_text}\n\n"
            "Grounding chunks:\n"
            f"{chunk_text}"
        ),
        response_schema=SOURCE_DEEP_DIVE_JSON_SCHEMA,
        required_keys=["deep_dive_text", "key_terms"],
    )
    deep_dive_text = str(payload.get("deep_dive_text", "")).strip()
    key_terms = [
        str(term).strip()
        for term in payload.get("key_terms", [])
        if str(term).strip()
    ]
    if not deep_dive_text:
        raise ValueError("Model returned an empty deep-dive section.")
    if len(key_terms) < 5:
        raise ValueError("Model returned too few key terms.")

    return deep_dive_text, key_terms


def _generate_reflection_points(
    summary_text: str,
    deep_dive_text: str,
    source_type: str,
    user_id: str,
    run_id: str,
) -> list[ReflectionPoint]:
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_reflection:{source_type}",
        user_id=user_id,
        system_prompt=SOCRATIC_REFLECTION_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Summary:\n{summary_text}\n\n"
            f"Deep dive:\n{deep_dive_text}\n\n"
            "Generate Socratic reflection points."
        ),
        response_schema=SOCRATIC_REFLECTION_JSON_SCHEMA,
        required_keys=["reflection_points"],
    )
    raw_points = payload.get("reflection_points", [])
    points: list[ReflectionPoint] = []
    for raw_point in raw_points:
        try:
            points.append(ReflectionPoint.model_validate(raw_point))
        except Exception:
            point_text = str(raw_point).strip()
            if point_text:
                points.append(
                    ReflectionPoint(
                        question=point_text,
                        explanation="Reflect on how this idea works in concrete situations.",
                        under_the_hood="Trace the mechanism that makes this idea true.",
                        depth_level="foundational",
                    )
                )

    if len(points) < 4:
        raise ValueError("Model returned too few reflection points.")

    return points


def _generate_under_surface_pack(
    *,
    summary_text: str,
    deep_dive_text: str,
    key_terms: list[str],
    source_type: str,
    grounding_chunks: list[str],
    user_id: str,
    run_id: str,
) -> tuple[str, list[str], list[KeyTermExplanation]]:
    chunk_text = "\n\n".join(grounding_chunks[:6]) if grounding_chunks else "[No grounding chunks available]"
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_under_surface:{source_type}",
        user_id=user_id,
        system_prompt=UNDER_SURFACE_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Summary:\n{summary_text}\n\n"
            f"Deep dive:\n{deep_dive_text}\n\n"
            f"Key terms: {', '.join(key_terms[:10])}\n\n"
            f"Grounding chunks:\n{chunk_text}"
        ),
        response_schema=UNDER_SURFACE_JSON_SCHEMA,
        required_keys=["under_surface_explainer", "diagnostic_checklist", "key_term_explanations"],
    )
    under_surface_explainer = str(payload.get("under_surface_explainer") or "").strip()
    diagnostic_checklist = [
        str(item).strip()
        for item in payload.get("diagnostic_checklist", [])
        if str(item).strip()
    ]
    key_term_explanations: list[KeyTermExplanation] = []
    for entry in payload.get("key_term_explanations", []):
        try:
            key_term_explanations.append(KeyTermExplanation.model_validate(entry))
        except Exception:
            continue

    if not under_surface_explainer:
        raise ValueError("Model returned empty under-surface explainer text.")
    if len(diagnostic_checklist) < 4:
        raise ValueError("Model returned too few diagnostic checklist points.")
    if len(key_term_explanations) < 4:
        raise ValueError("Model returned too few key-term explanations.")

    return under_surface_explainer, diagnostic_checklist, key_term_explanations


def _store_source_learning_section(section: SourceLearningSection, user_id: str) -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO source_learning_sections (
                    user_id,
                    source_id,
                    source_type,
                    source_name,
                    generated_title,
                    summary_text,
                    reflection_points_json,
                    deep_dive_text,
                    key_terms_json,
                    under_surface_text,
                    diagnostic_checklist_json,
                    key_term_explanations_json,
                    schema_version,
                    model_name
                ) VALUES (
                    :user_id,
                    :source_id,
                    :source_type,
                    :source_name,
                    :generated_title,
                    :summary_text,
                    :reflection_points_json,
                    :deep_dive_text,
                    :key_terms_json,
                    :under_surface_text,
                    :diagnostic_checklist_json,
                    :key_term_explanations_json,
                    :schema_version,
                    :model_name
                )
                ON CONFLICT(user_id, source_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    source_name = excluded.source_name,
                    generated_title = excluded.generated_title,
                    summary_text = excluded.summary_text,
                    reflection_points_json = excluded.reflection_points_json,
                    deep_dive_text = excluded.deep_dive_text,
                    key_terms_json = excluded.key_terms_json,
                    under_surface_text = excluded.under_surface_text,
                    diagnostic_checklist_json = excluded.diagnostic_checklist_json,
                    key_term_explanations_json = excluded.key_term_explanations_json,
                    schema_version = excluded.schema_version,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "source_id": section.source_id,
                "source_type": section.source_type,
                "source_name": section.source_name,
                "generated_title": section.generated_title,
                "summary_text": section.summary_text,
                "reflection_points_json": json.dumps(
                    [point.model_dump() for point in section.reflection_points],
                    ensure_ascii=True,
                ),
                "deep_dive_text": section.deep_dive_text,
                "key_terms_json": json.dumps(section.key_terms, ensure_ascii=True),
                "under_surface_text": section.under_surface_explainer,
                "diagnostic_checklist_json": json.dumps(section.diagnostic_checklist, ensure_ascii=True),
                "key_term_explanations_json": json.dumps(
                    [entry.model_dump() for entry in section.key_term_explanations],
                    ensure_ascii=True,
                ),
                "schema_version": section.schema_version,
                "model_name": section.model_name,
            },
        )


def _load_cached_source_learning_section(source_id: int, user_id: str) -> SourceLearningSection | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    source_id,
                    source_type,
                    source_name,
                    generated_title,
                    summary_text,
                    reflection_points_json,
                    deep_dive_text,
                    key_terms_json,
                    under_surface_text,
                    diagnostic_checklist_json,
                    key_term_explanations_json,
                    model_name,
                    schema_version
                FROM source_learning_sections
                WHERE user_id = :user_id AND source_id = :source_id
                LIMIT 1
                """
            ),
            {"user_id": user_id, "source_id": source_id},
        ).mappings().first()

    if row is None:
        return None

    if int(row.get("schema_version") or 1) != GENERATION_SCHEMA_VERSION:
        return None

    reflection_points_raw = safe_json_loads(row.get("reflection_points_json"), default=[])
    reflection_points: list[ReflectionPoint] = []
    for value in reflection_points_raw:
        if isinstance(value, str):
            value = value.strip()
            if value:
                reflection_points.append(
                    ReflectionPoint(
                        question=value,
                        explanation="Reflect on what this means in context.",
                        under_the_hood="Identify the underlying mechanism behind this idea.",
                        depth_level="foundational",
                    )
                )
        else:
            try:
                reflection_points.append(ReflectionPoint.model_validate(value))
            except Exception:
                continue

    key_terms_raw = safe_json_loads(row.get("key_terms_json"), default=[])
    key_terms = [str(term).strip() for term in key_terms_raw if str(term).strip()]

    diagnostic_checklist_raw = safe_json_loads(row.get("diagnostic_checklist_json"), default=[])
    diagnostic_checklist = [str(item).strip() for item in diagnostic_checklist_raw if str(item).strip()]

    key_term_explanations_raw = safe_json_loads(row.get("key_term_explanations_json"), default=[])
    key_term_explanations: list[KeyTermExplanation] = []
    for entry in key_term_explanations_raw:
        try:
            key_term_explanations.append(KeyTermExplanation.model_validate(entry))
        except Exception:
            continue

    under_surface_explainer = str(row.get("under_surface_text") or "").strip()

    if not reflection_points or not key_terms or not under_surface_explainer:
        return None

    return SourceLearningSection(
        source_id=int(row["source_id"]),
        source_type=str(row["source_type"]),
        source_name=str(row["source_name"] or row["generated_title"] or "Source").strip(),
        generated_title=str(row["generated_title"]),
        summary_text=str(row["summary_text"]),
        deep_dive_text=str(row["deep_dive_text"] or "").strip(),
        key_terms=key_terms,
        under_surface_explainer=under_surface_explainer,
        diagnostic_checklist=diagnostic_checklist,
        key_term_explanations=key_term_explanations,
        reflection_points=reflection_points,
        model_name=str(row["model_name"]),
        schema_version=int(row["schema_version"]),
    )


def _build_source_learning_section(source_id: int, user_id: str, run_id: str) -> SourceLearningSection:
    source_type, source_name, _ = get_source_metadata(source_id, user_id)

    if CACHE_PROCESSED_SOURCES:
        cached = _load_cached_source_learning_section(source_id, user_id)
        if cached is not None:
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=f"source_learning_cache:{source_id}",
                attempt_number=1,
                status="cache_hit",
                details=f"source_type={source_type}",
            )
            return cached
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=f"source_learning_cache:{source_id}",
            attempt_number=1,
            status="cache_miss",
            details=f"source_type={source_type}",
        )

    summary_text = load_existing_source_summary(source_id, user_id)
    if summary_text is None:
        process_result = process_source(source_id, user_id)
        summary_text = (process_result.source_summary or "").strip()

    if not summary_text:
        raise ValueError(f"Source {source_id} summary is unavailable.")

    grounding_chunks = _load_grounding_chunks(source_id, user_id)
    generated_title = _generate_source_title(summary_text, source_type, user_id, run_id)
    deep_dive_text, key_terms = _generate_source_deep_dive(
        summary_text,
        source_type,
        grounding_chunks,
        user_id,
        run_id,
    )
    reflection_points = _generate_reflection_points(
        summary_text,
        deep_dive_text,
        source_type,
        user_id,
        run_id,
    )
    under_surface_explainer, diagnostic_checklist, key_term_explanations = _generate_under_surface_pack(
        summary_text=summary_text,
        deep_dive_text=deep_dive_text,
        key_terms=key_terms,
        source_type=source_type,
        grounding_chunks=grounding_chunks,
        user_id=user_id,
        run_id=run_id,
    )

    section = SourceLearningSection(
        source_id=source_id,
        source_type=source_type,
        source_name=str(source_name).strip() or f"source-{source_id}",
        generated_title=generated_title,
        summary_text=summary_text,
        deep_dive_text=deep_dive_text,
        key_terms=key_terms,
        under_surface_explainer=under_surface_explainer,
        diagnostic_checklist=diagnostic_checklist,
        key_term_explanations=key_term_explanations,
        reflection_points=reflection_points,
        model_name=GENERATION_MODEL,
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    _store_source_learning_section(section, user_id)
    return section


def _normalize_attributed_sentence(
    raw_item: dict,
    known_sources: dict[int, str],
    fallback_source_id: int,
) -> AttributedSentence | None:
    source_id = int(raw_item.get("source_id", fallback_source_id))
    if source_id not in known_sources:
        source_id = fallback_source_id

    source_type = str(raw_item.get("source_type", known_sources[source_id]))
    if source_type not in {"video", "document"}:
        source_type = known_sources[source_id]

    emphasis_terms = [
        str(term).strip()
        for term in raw_item.get("emphasis_terms", [])
        if str(term).strip()
    ]
    if not emphasis_terms:
        emphasis_terms = ["key idea"]

    text_value = str(raw_item.get("text", "")).strip()
    if not text_value:
        return None

    return AttributedSentence(
        text=text_value,
        source_id=source_id,
        source_type=source_type,
        emphasis_terms=emphasis_terms,
    )


def _generate_combined_insights(
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
    relationship_insights: list[str],
    user_id: str,
    run_id: str,
) -> CombinedInsightSection:
    source_catalog = [
        {
            "source_id": video_section.source_id,
            "source_type": "video",
            "title": video_section.generated_title,
            "source_name": video_section.source_name,
        }
    ] + [
        {
            "source_id": section.source_id,
            "source_type": "document",
            "title": section.generated_title,
            "source_name": section.source_name,
        }
        for section in document_sections
    ]

    source_payload = {
        "source_catalog": source_catalog,
        "video": video_section.model_dump(),
        "documents": [section.model_dump() for section in document_sections],
        "relationship_insights": relationship_insights,
    }

    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name="combined_insights",
        user_id=user_id,
        system_prompt=COMBINED_INSIGHTS_SYSTEM_PROMPT,
        user_prompt=(
            "Generate cross-source attributed insights from this payload:\n\n"
            f"{json.dumps(source_payload, ensure_ascii=True)}"
        ),
        response_schema=COMBINED_INSIGHTS_JSON_SCHEMA,
        required_keys=["intersections", "layman_bridge", "synthesis_text"],
    )
    known_sources = {
        video_section.source_id: "video",
        **{section.source_id: "document" for section in document_sections},
    }

    raw_intersections = payload.get("intersections", [])
    intersections: list[InsightIntersection] = []
    parallels: list[AttributedSentence] = []

    for raw_intersection in raw_intersections:
        if not isinstance(raw_intersection, dict):
            continue

        raw_sentences = raw_intersection.get("attributed_sentences", [])
        attributed_sentences: list[AttributedSentence] = []
        for raw_sentence in raw_sentences:
            if not isinstance(raw_sentence, dict):
                continue
            normalized_sentence = _normalize_attributed_sentence(
                raw_sentence,
                known_sources,
                video_section.source_id,
            )
            if normalized_sentence is not None:
                attributed_sentences.append(normalized_sentence)

        if len(attributed_sentences) < 2:
            continue

        intersection_title = str(raw_intersection.get("intersection_title", "")).strip()
        why_it_matters = str(raw_intersection.get("why_it_matters", "")).strip()
        integrated_explanation = str(raw_intersection.get("integrated_explanation", "")).strip()
        if not intersection_title or not why_it_matters or not integrated_explanation:
            continue

        inferred_extension = raw_intersection.get("inferred_extension")
        inferred_extension_text = (
            str(inferred_extension).strip()
            if inferred_extension is not None
            else None
        )
        if inferred_extension_text == "":
            inferred_extension_text = None

        inference_label = raw_intersection.get("inference_label")
        inference_label_value = (
            "inferred_extension"
            if inferred_extension_text and str(inference_label).strip() == "inferred_extension"
            else None
        )

        intersections.append(
            InsightIntersection(
                intersection_title=intersection_title,
                why_it_matters=why_it_matters,
                integrated_explanation=integrated_explanation,
                attributed_sentences=attributed_sentences,
                inferred_extension=inferred_extension_text,
                inference_label=inference_label_value,
            )
        )
        parallels.extend(attributed_sentences)

    if not intersections:
        # Backward compatibility: handle legacy payload shape.
        raw_parallels = payload.get("parallels", [])
        legacy_sentences: list[AttributedSentence] = []
        for raw_item in raw_parallels:
            if not isinstance(raw_item, dict):
                continue
            normalized_sentence = _normalize_attributed_sentence(
                raw_item,
                known_sources,
                video_section.source_id,
            )
            if normalized_sentence is not None:
                legacy_sentences.append(normalized_sentence)

        if len(legacy_sentences) >= 3:
            intersections.append(
                InsightIntersection(
                    intersection_title="Core Cross-Source Intersection",
                    why_it_matters="This overlap captures the strongest shared mechanism across your sources.",
                    integrated_explanation="These attributed points connect what the video and documents reinforce, extend, or challenge.",
                    attributed_sentences=legacy_sentences,
                )
            )
            parallels.extend(legacy_sentences)

    layman_bridge = str(payload.get("layman_bridge", "")).strip()
    synthesis_text = str(payload.get("synthesis_text", "")).strip()
    if len(intersections) < 1:
        raise ValueError("Model returned too few integrated cross-source intersections.")
    if not layman_bridge:
        raise ValueError("Model returned an empty layman bridge.")
    if not synthesis_text:
        raise ValueError("Model returned an empty synthesis text.")

    return CombinedInsightSection(
        intersections=intersections,
        parallels=parallels,
        layman_bridge=layman_bridge,
        synthesis_text=synthesis_text,
        model_name=GENERATION_MODEL,
        schema_version=GENERATION_SCHEMA_VERSION,
    )


def _generate_combined_quiz(
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
    insights_section: CombinedInsightSection,
    relationship_insights: list[str],
    user_id: str,
    run_id: str,
) -> CombinedQuizSection:
    quiz_payload = {
        "video": video_section.model_dump(),
        "documents": [section.model_dump() for section in document_sections],
        "insights": insights_section.model_dump(),
        "relationship_insights": relationship_insights,
        "difficulty_mix": {
            "foundational": "2-3 questions",
            "intermediate": "2-4 questions",
            "advanced": "1-3 questions",
        },
        "style": "hard, high-discrimination trick questions with clear correctness",
        "distractor_policy": (
            "Use topic-adjacent plausible distractors, avoid giveaway absolutes such as solely/always/never/entirely/only "
            "unless directly grounded, and prefer distractors that can be true in nearby contexts but not for the asked scope"
        ),
        "under_the_hood_depth": (
            "Provide technical, first-principles reasoning with mechanism, assumptions, constraints, and tradeoffs"
        ),
    }

    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name="combined_quiz",
        user_id=user_id,
        system_prompt=COMBINED_QUIZ_SYSTEM_PROMPT,
        user_prompt=(
            "Generate overlap-focused advanced quiz questions from this payload:\n\n"
            f"{json.dumps(quiz_payload, ensure_ascii=True)}"
        ),
        response_schema=COMBINED_QUIZ_JSON_SCHEMA,
        required_keys=["questions", "study_advice"],
    )
    questions = [QuizQuestion.model_validate(item) for item in payload.get("questions", [])]
    if len(questions) < 6:
        raise ValueError("Model returned too few quiz questions.")

    study_advice = str(payload.get("study_advice", "")).strip()
    if not study_advice:
        raise ValueError("Model returned empty study advice.")

    return CombinedQuizSection(
        questions=questions,
        study_advice=study_advice,
        model_name=GENERATION_MODEL,
        schema_version=GENERATION_SCHEMA_VERSION,
    )


def _generate_comparative_analysis(
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
    insights_section: CombinedInsightSection,
    relationship_insights: list[str],
    user_id: str,
    run_id: str,
) -> str:
    payload = {
        "video": video_section.model_dump(),
        "documents": [section.model_dump() for section in document_sections],
        "insights": insights_section.model_dump(),
        "relationship_insights": relationship_insights,
    }

    result = _run_structured_generation_step(
        run_id=run_id,
        stage_name="comparative_analysis",
        user_id=user_id,
        system_prompt=COMPARATIVE_ANALYSIS_SYSTEM_PROMPT,
        user_prompt=(
            "Generate one comparative deepening analysis from this payload:\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        ),
        response_schema=COMPARATIVE_ANALYSIS_JSON_SCHEMA,
        required_keys=["comparative_analysis"],
    )
    comparative_analysis = str(result.get("comparative_analysis", "")).strip()
    if not comparative_analysis:
        raise ValueError("Model returned empty comparative analysis text.")

    return comparative_analysis


def _generate_application_scenarios(
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
    insights_section: CombinedInsightSection,
    relationship_insights: list[str],
    user_id: str,
    run_id: str,
) -> list[ApplicationScenario]:
    payload = {
        "video": video_section.model_dump(),
        "documents": [section.model_dump() for section in document_sections],
        "insights": insights_section.model_dump(),
        "relationship_insights": relationship_insights,
    }

    result = _run_structured_generation_step(
        run_id=run_id,
        stage_name="application_scenarios",
        user_id=user_id,
        system_prompt=APPLICATION_SCENARIOS_SYSTEM_PROMPT,
        user_prompt=(
            "Generate grounded application scenarios from this payload:\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        ),
        response_schema=APPLICATION_SCENARIOS_JSON_SCHEMA,
        required_keys=["application_scenarios"],
    )
    scenarios = [
        ApplicationScenario.model_validate(item)
        for item in result.get("application_scenarios", [])
    ]
    if len(scenarios) < 2:
        raise ValueError("Model returned too few application scenarios.")

    return scenarios


def _store_combined_learning_sections(
    *,
    user_id: str,
    video_source_id: int,
    document_source_ids: list[int],
    insights: CombinedInsightSection,
    quiz: CombinedQuizSection,
) -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO combined_learning_sections (
                    user_id,
                    video_source_id,
                    document_source_ids_json,
                    intersections_json,
                    parallels_json,
                    layman_bridge,
                    synthesis_text,
                    comparative_analysis_text,
                    application_scenarios_json,
                    quiz_json,
                    schema_version,
                    model_name
                ) VALUES (
                    :user_id,
                    :video_source_id,
                    :document_source_ids_json,
                    :intersections_json,
                    :parallels_json,
                    :layman_bridge,
                    :synthesis_text,
                    :comparative_analysis_text,
                    :application_scenarios_json,
                    :quiz_json,
                    :schema_version,
                    :model_name
                )
                ON CONFLICT(user_id, video_source_id, document_source_ids_json) DO UPDATE SET
                    intersections_json = excluded.intersections_json,
                    parallels_json = excluded.parallels_json,
                    layman_bridge = excluded.layman_bridge,
                    synthesis_text = excluded.synthesis_text,
                    comparative_analysis_text = excluded.comparative_analysis_text,
                    application_scenarios_json = excluded.application_scenarios_json,
                    quiz_json = excluded.quiz_json,
                    schema_version = excluded.schema_version,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "video_source_id": video_source_id,
                "document_source_ids_json": _serialize_ids(document_source_ids),
                "intersections_json": json.dumps(
                    [entry.model_dump() for entry in insights.intersections],
                    ensure_ascii=True,
                ),
                "parallels_json": json.dumps(
                    [entry.model_dump() for entry in insights.parallels],
                    ensure_ascii=True,
                ),
                "layman_bridge": insights.layman_bridge,
                "synthesis_text": insights.synthesis_text,
                "comparative_analysis_text": insights.comparative_analysis,
                "application_scenarios_json": json.dumps(
                    [entry.model_dump() for entry in insights.application_scenarios],
                    ensure_ascii=True,
                ),
                "quiz_json": json.dumps(quiz.model_dump(), ensure_ascii=True),
                "schema_version": GENERATION_SCHEMA_VERSION,
                "model_name": GENERATION_MODEL,
            },
        )


def _load_cached_combined_learning_sections(
    *,
    user_id: str,
    video_source_id: int,
    document_source_ids: list[int],
) -> tuple[CombinedInsightSection, CombinedQuizSection] | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    intersections_json,
                    parallels_json,
                    layman_bridge,
                    synthesis_text,
                    comparative_analysis_text,
                    application_scenarios_json,
                    quiz_json,
                    model_name,
                    schema_version
                FROM combined_learning_sections
                WHERE user_id = :user_id
                  AND video_source_id = :video_source_id
                  AND document_source_ids_json = :document_source_ids_json
                LIMIT 1
                """
            ),
            {
                "user_id": user_id,
                "video_source_id": video_source_id,
                "document_source_ids_json": _serialize_ids(document_source_ids),
            },
        ).mappings().first()

    if row is None:
        return None

    if int(row.get("schema_version") or 1) != GENERATION_SCHEMA_VERSION:
        return None

    raw_intersections = safe_json_loads(row.get("intersections_json"), default=[])
    intersections: list[InsightIntersection] = []
    for entry in raw_intersections:
        try:
            intersections.append(InsightIntersection.model_validate(entry))
        except Exception:
            continue

    raw_parallels = safe_json_loads(row.get("parallels_json"), default=[])
    parallels: list[AttributedSentence] = []
    for entry in raw_parallels:
        try:
            parallels.append(AttributedSentence.model_validate(entry))
        except Exception:
            continue

    if not intersections and parallels:
        intersections.append(
            InsightIntersection(
                intersection_title="Core Cross-Source Intersection",
                why_it_matters="This overlap captures the strongest shared mechanism across your sources.",
                integrated_explanation="These attributed points connect what the video and documents reinforce, extend, or challenge.",
                attributed_sentences=parallels,
            )
        )

    if not intersections:
        return None

    layman_bridge = str(row["layman_bridge"] or "").strip()
    synthesis_text = str(row["synthesis_text"] or "").strip()
    if not layman_bridge or not synthesis_text:
        return None

    comparative_analysis = str(row.get("comparative_analysis_text") or "").strip()
    raw_application_scenarios = safe_json_loads(row.get("application_scenarios_json"), default=[])
    application_scenarios: list[ApplicationScenario] = []
    for entry in raw_application_scenarios:
        try:
            application_scenarios.append(ApplicationScenario.model_validate(entry))
        except Exception:
            continue
    if not comparative_analysis or len(application_scenarios) < 2:
        return None

    quiz_payload = safe_json_loads(row.get("quiz_json"), default={})
    try:
        quiz_section = CombinedQuizSection.model_validate(quiz_payload)
    except Exception:
        return None

    insights_section = CombinedInsightSection(
        intersections=intersections,
        parallels=parallels,
        layman_bridge=layman_bridge,
        synthesis_text=synthesis_text,
        comparative_analysis=comparative_analysis,
        application_scenarios=application_scenarios,
        model_name=str(row["model_name"]),
        schema_version=int(row["schema_version"]),
    )
    return insights_section, quiz_section


def generate_tailored_learning(
    video_source_id: int,
    document_source_ids: list[int],
    user_id: str,
) -> GenerateTailoredLearningResponse:
    ensure_generation_tables()
    run_id = f"gen-{uuid.uuid4().hex[:12]}"

    normalized_video_id, normalized_document_ids = _normalize_ids(
        video_source_id,
        document_source_ids,
    )

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name="pipeline_start",
        attempt_number=1,
        status="started",
        details=(
            f"video_source_id={normalized_video_id}; "
            f"document_source_ids={','.join(str(value) for value in normalized_document_ids)}"
        ),
    )

    video_process_result = process_source(normalized_video_id, user_id)
    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name=f"process_source:{normalized_video_id}",
        attempt_number=1,
        status="cache_hit" if int(video_process_result.chunk_count) == 0 else "succeeded",
        details=f"source_type=video; chunk_count={video_process_result.chunk_count}",
    )
    for document_source_id in normalized_document_ids:
        process_result = process_source(document_source_id, user_id)
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=f"process_source:{document_source_id}",
            attempt_number=1,
            status="cache_hit" if int(process_result.chunk_count) == 0 else "succeeded",
            details=f"source_type=document; chunk_count={process_result.chunk_count}",
        )

    for document_source_id in normalized_document_ids:
        link_result = link_source_pair(normalized_video_id, document_source_id, user_id)
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=f"link_sources:{normalized_video_id}:{document_source_id}",
            attempt_number=1,
            status="cache_hit" if int(link_result.get("stored_edges", 0)) == 0 else "succeeded",
            details=(
                f"candidate_pairs={int(link_result.get('candidate_pairs', 0))}; "
                f"stored_edges={int(link_result.get('stored_edges', 0))}"
            ),
        )

    video_section = _build_source_learning_section(normalized_video_id, user_id, run_id)
    document_sections = [
        _build_source_learning_section(source_id, user_id, run_id)
        for source_id in normalized_document_ids
    ]

    cached_combined = None
    if CACHE_PROCESSED_SOURCES:
        cached_combined = _load_cached_combined_learning_sections(
            user_id=user_id,
            video_source_id=normalized_video_id,
            document_source_ids=normalized_document_ids,
        )
    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name="combined_learning_cache",
        attempt_number=1,
        status="cache_hit" if cached_combined is not None else "cache_miss",
    )

    relationship_insights = _load_relationship_insights(
        [normalized_video_id, *normalized_document_ids],
        user_id,
    )

    if cached_combined is not None:
        insights_section, quiz_section = cached_combined
    else:
        insights_section = _generate_combined_insights(
            video_section=video_section,
            document_sections=document_sections,
            relationship_insights=relationship_insights,
            user_id=user_id,
            run_id=run_id,
        )
        quiz_section = _generate_combined_quiz(
            video_section=video_section,
            document_sections=document_sections,
            insights_section=insights_section,
            relationship_insights=relationship_insights,
            user_id=user_id,
            run_id=run_id,
        )
        comparative_analysis = _generate_comparative_analysis(
            video_section=video_section,
            document_sections=document_sections,
            insights_section=insights_section,
            relationship_insights=relationship_insights,
            user_id=user_id,
            run_id=run_id,
        )
        application_scenarios = _generate_application_scenarios(
            video_section=video_section,
            document_sections=document_sections,
            insights_section=insights_section,
            relationship_insights=relationship_insights,
            user_id=user_id,
            run_id=run_id,
        )
        insights_section = insights_section.model_copy(
            update={
                "comparative_analysis": comparative_analysis,
                "application_scenarios": application_scenarios,
            }
        )
        _store_combined_learning_sections(
            user_id=user_id,
            video_source_id=normalized_video_id,
            document_source_ids=normalized_document_ids,
            insights=insights_section,
            quiz=quiz_section,
        )

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name="pipeline_end",
        attempt_number=1,
        status="succeeded",
        details=(
            f"video_source_id={normalized_video_id}; "
            f"document_count={len(normalized_document_ids)}"
        ),
    )

    return GenerateTailoredLearningResponse(
        status_message="Tailored Socratic learning generated successfully.",
        source_ids=[normalized_video_id, *normalized_document_ids],
        video=video_section,
        documents=document_sections,
        insights=insights_section,
        quiz=quiz_section,
    )
