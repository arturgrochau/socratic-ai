from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from difflib import SequenceMatcher
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
from prompts.source_progression_outline import (
    SOURCE_PROGRESSION_OUTLINE_JSON_SCHEMA,
    SOURCE_PROGRESSION_OUTLINE_SYSTEM_PROMPT,
)
from prompts.source_title import SOURCE_TITLE_JSON_SCHEMA, SOURCE_TITLE_SYSTEM_PROMPT
from prompts.under_surface import UNDER_SURFACE_JSON_SCHEMA, UNDER_SURFACE_SYSTEM_PROMPT


GENERATION_SCHEMA_VERSION = 8
MAX_GROUNDING_CHUNKS = 10
MAX_GROUNDING_CHARS = 9000
MAX_UNDER_SURFACE_GROUNDING_CHUNKS = 6
MAX_UNDER_SURFACE_GROUNDING_CHARS = 7000
MAX_REFLECTION_GROUNDING_CHUNKS = 6
MAX_REFLECTION_GROUNDING_CHARS = 7000
GENERATION_MAX_STAGE_ATTEMPTS = 3
MAX_STAGE_ERROR_CHARS = 700
DEDUPLICATION_SIMILARITY_THRESHOLD = 0.9
DEDUPLICATION_TOKEN_OVERLAP_THRESHOLD = 0.58
DEDUPLICATION_MIN_CHAR_RATIO = 0.35
DEDUPLICATION_MIN_TOKEN_LENGTH = 4
OUTLINE_MODEL_MIN_GROUNDING_ROWS = 12
MAX_EXPANSION_GROUNDING_CHARS = 8000
MAX_DEEP_DIVE_WORDS = 900
SECTION_REDUNDANCY_RATIO_THRESHOLD = 0.34
SECTION_SIMILARITY_THRESHOLD = 0.82
SECTION_TOKEN_OVERLAP_THRESHOLD = 0.50
SOURCE_SECTION_PAIR_OVERLAP_THRESHOLD = 0.32
CROSS_SECTION_PAIR_OVERLAP_THRESHOLD = 0.26

TEXT_EXPANSION_JSON_SCHEMA = {
    "name": "expanded_text",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "expanded_text": {"type": "string"},
        },
        "required": ["expanded_text"],
    },
}


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

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS generation_novelty_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    source_id INTEGER,
                    document_source_ids_json TEXT DEFAULT '',
                    section_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    similarity_to_prior REAL DEFAULT 0.0,
                    overlap_sentences_count INTEGER DEFAULT 0,
                    total_sentences_count INTEGER DEFAULT 0,
                    expansion_applied INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_novelty_ledger_user_run
                ON generation_novelty_ledger (user_id, run_id, section_type)
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS generation_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    section_type TEXT NOT NULL,
                    pair_key TEXT NOT NULL,
                    overlap_ratio REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_quality_metrics_run
                ON generation_quality_metrics (user_id, run_id, section_type)
                """
            )
        )


def _normalize_ids(
    video_source_id: int | None,
    document_source_ids: list[int],
) -> tuple[int | None, list[int]]:
    normalized_video_id = video_source_id if (video_source_id or 0) > 0 else None
    if video_source_id is not None and normalized_video_id is None:
        raise ValueError("video_source_id must be a positive integer.")

    normalized_document_ids = sorted(
        {source_id for source_id in document_source_ids if source_id > 0}
    )
    if normalized_video_id is None and not normalized_document_ids:
        raise ValueError("At least one valid source_id is required.")

    return normalized_video_id, normalized_document_ids


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


def _load_grounding_chunk_rows(source_id: int, user_id: str) -> list[dict[str, Any]]:
    with db_engine.connect() as connection:
        return list(
            connection.execute(
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
        )


def _format_grounding_entry(row: dict[str, Any]) -> str | None:
    chunk_text = str(row["chunk_text"] or "").strip()
    if not chunk_text:
        return None

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

    return f"[{chunk_label}]\n{chunk_text}"


def _evenly_sample_positions(positions: list[int], count: int) -> list[int]:
    if count <= 0 or not positions:
        return []
    if count >= len(positions):
        return list(positions)
    if count == 1:
        return [positions[len(positions) // 2]]

    step = (len(positions) - 1) / (count - 1)
    sampled = [positions[round(index * step)] for index in range(count)]

    deduped: list[int] = []
    seen: set[int] = set()
    for value in sampled:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _compute_distributed_row_positions(total_rows: int, max_chunks: int) -> list[int]:
    if total_rows <= 0 or max_chunks <= 0:
        return []
    if max_chunks >= total_rows:
        return list(range(total_rows))

    one_third = total_rows // 3
    two_thirds = (2 * total_rows) // 3
    segments = [
        list(range(0, max(one_third, 1))),
        list(range(max(one_third, 1), max(two_thirds, max(one_third, 1)))),
        list(range(max(two_thirds, max(one_third, 1)), total_rows)),
    ]

    base = max_chunks // 3
    remainder = max_chunks % 3
    segment_quotas = [base + (1 if index < remainder else 0) for index in range(3)]

    chosen: list[int] = []
    for segment_positions, quota in zip(segments, segment_quotas):
        chosen.extend(_evenly_sample_positions(segment_positions, quota))

    seen: set[int] = set()
    ordered_unique: list[int] = []
    for value in sorted(chosen):
        if value in seen:
            continue
        seen.add(value)
        ordered_unique.append(value)

    if len(ordered_unique) < max_chunks:
        for fallback in _evenly_sample_positions(list(range(total_rows)), total_rows):
            if fallback in seen:
                continue
            seen.add(fallback)
            ordered_unique.append(fallback)
            if len(ordered_unique) >= max_chunks:
                break

    return sorted(ordered_unique[:max_chunks])


def _select_distributed_grounding_chunks(
    rows: list[dict[str, Any]],
    *,
    max_chunks: int,
    max_chars: int,
) -> tuple[list[str], list[int]]:
    positions = _compute_distributed_row_positions(len(rows), max_chunks)
    chunks: list[str] = []
    selected_chunk_indexes: list[int] = []
    consumed_chars = 0

    for position in positions:
        if position < 0 or position >= len(rows):
            continue
        row = rows[position]
        entry = _format_grounding_entry(row)
        if not entry:
            continue

        if consumed_chars + len(entry) > max_chars and chunks:
            continue
        if consumed_chars + len(entry) > max_chars and not chunks:
            entry = entry[:max_chars]

        chunks.append(entry)
        selected_chunk_indexes.append(int(row["chunk_index"]))
        consumed_chars += len(entry)
        if len(chunks) >= max_chunks:
            break

    if not chunks and rows:
        fallback_entry = _format_grounding_entry(rows[0])
        if fallback_entry:
            chunks.append(fallback_entry[:max_chars])
            selected_chunk_indexes.append(int(rows[0]["chunk_index"]))

    return chunks, selected_chunk_indexes


def _should_use_model_progression_outline(*, total_rows: int, grounding_chunks: list[str]) -> bool:
    return total_rows >= OUTLINE_MODEL_MIN_GROUNDING_ROWS and len(grounding_chunks) >= 5


def _build_heuristic_progression_outline(summary_text: str, grounding_chunks: list[str]) -> str:
    summary_sentences = _split_into_sentences(summary_text)
    summary_anchor = summary_sentences[0] if summary_sentences else "Core ideas are developed progressively."

    chunk_labels: list[str] = []
    for entry in grounding_chunks:
        first_line = str(entry).splitlines()[0].strip() if entry else ""
        if first_line.startswith("[") and first_line.endswith("]"):
            chunk_labels.append(first_line.strip("[]"))

    if chunk_labels:
        first_label = chunk_labels[0]
        mid_label = chunk_labels[len(chunk_labels) // 2]
        last_label = chunk_labels[-1]
    else:
        first_label = "early section"
        mid_label = "middle section"
        last_label = "later section"

    transitions = [
        f"- Foundations are established in {first_label} and frame the core mechanism.",
        f"- Complexity increases by {mid_label}, where interactions and tradeoffs become explicit.",
        f"- Edge-case behavior and implications appear in {last_label}, clarifying failure boundaries.",
    ]
    transition_block = "\n".join(transitions)
    return (
        f"{summary_anchor} The material progresses from fundamentals to integration and finally to "
        "boundary conditions that pressure-test the core assumptions.\n\n"
        "Core transitions:\n"
        f"{transition_block}"
    )


def _tokenize_for_similarity(text_value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text_value or "").lower())
        if len(token) >= DEDUPLICATION_MIN_TOKEN_LENGTH
    }


def _token_overlap_ratio(left_text: str, right_text: str) -> float:
    left_tokens = _tokenize_for_similarity(left_text)
    right_tokens = _tokenize_for_similarity(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection_size = len(left_tokens.intersection(right_tokens))
    union_size = len(left_tokens.union(right_tokens))
    if union_size == 0:
        return 0.0
    return intersection_size / union_size


def _generate_source_progression_outline(
    *,
    summary_text: str,
    source_type: str,
    grounding_chunks: list[str],
    user_id: str,
    run_id: str,
) -> str:
    chunk_text = "\n\n".join(grounding_chunks) if grounding_chunks else "[No grounding chunks available]"
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_progression_outline:{source_type}",
        user_id=user_id,
        system_prompt=SOURCE_PROGRESSION_OUTLINE_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Summary:\n{summary_text}\n\n"
            "Distributed grounding chunks:\n"
            f"{chunk_text}"
        ),
        response_schema=SOURCE_PROGRESSION_OUTLINE_JSON_SCHEMA,
        required_keys=["progression_outline", "transition_points"],
    )

    progression_outline = str(payload.get("progression_outline") or "").strip()
    transition_points = [
        str(item).strip()
        for item in payload.get("transition_points", [])
        if str(item).strip()
    ]
    if not progression_outline:
        raise ValueError("Model returned empty progression outline text.")

    if transition_points:
        transition_block = "\n".join(f"- {item}" for item in transition_points)
        return f"{progression_outline}\n\nCore transitions:\n{transition_block}"
    return progression_outline


def _split_into_sentences(text_value: str) -> list[str]:
    normalized = " ".join(str(text_value or "").split()).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _deduplicate_section_text(current_text: str, prior_texts: list[str]) -> str:
    current_sentences = _split_into_sentences(current_text)
    if len(current_sentences) < 2:
        return current_text

    prior_sentences: list[str] = []
    for text_value in prior_texts:
        prior_sentences.extend(_split_into_sentences(text_value))

    kept_sentences: list[str] = []
    for sentence in current_sentences:
        sentence_norm = sentence.lower()
        is_duplicate = False
        for prior_sentence in [*prior_sentences, *kept_sentences]:
            prior_norm = prior_sentence.lower()
            if sentence_norm == prior_norm:
                is_duplicate = True
                break
            sequence_similarity = SequenceMatcher(None, sentence_norm, prior_norm).ratio()
            token_overlap = _token_overlap_ratio(sentence_norm, prior_norm)
            if sequence_similarity >= DEDUPLICATION_SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
            if token_overlap >= DEDUPLICATION_TOKEN_OVERLAP_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_sentences.append(sentence)

    if not kept_sentences:
        return current_text

    deduplicated = " ".join(kept_sentences).strip()
    if len(deduplicated) < int(len(current_text) * DEDUPLICATION_MIN_CHAR_RATIO):
        return current_text
    return deduplicated


def _truncate_for_prompt(text_value: str, *, limit: int = 900) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _normalize_continuous_prose(text_value: str) -> str:
    raw_lines = str(text_value or "").splitlines()
    cleaned_lines: list[str] = []

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # Remove disallowed dash styling and markdown residue, including leaked inline hashes.
        line = line.replace("\u2014", ", ").replace("\u2013", ", ")
        line = re.sub(r"\s-\s", ", ", line)

        # Drop markdown heading/list markers but keep the content.
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^#{1,6}(?=\S)", "", line).strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"^>\s+", "", line)
        line = re.sub(r"^\*\*(.+?)\*\*$", r"\1", line)
        line = re.sub(r"#{2,}", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    paragraphs: list[str] = []
    current: list[str] = []
    for line in cleaned_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    prose = "\n\n".join(part for part in paragraphs if part).strip()
    prose = re.sub(r"\n{3,}", "\n\n", prose)
    return prose


def _truncate_to_max_words(text_value: str, *, max_words: int) -> str:
    normalized = _normalize_continuous_prose(text_value)
    if not normalized:
        return ""

    words = normalized.split()
    if len(words) <= max_words:
        return normalized

    sentences = _split_into_sentences(normalized)
    if not sentences:
        return " ".join(words[:max_words]).strip()

    kept_sentences: list[str] = []
    running_words = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        if kept_sentences and running_words + len(sentence_words) > max_words:
            break
        if not kept_sentences and len(sentence_words) > max_words:
            return " ".join(sentence_words[:max_words]).strip()

        kept_sentences.append(sentence)
        running_words += len(sentence_words)

    if not kept_sentences:
        return " ".join(words[:max_words]).strip()

    return " ".join(kept_sentences).strip()


def _section_redundancy_ratio(current_text: str, prior_texts: list[str]) -> float:
    current_sentences = _split_into_sentences(current_text)
    if not current_sentences:
        return 0.0

    prior_sentences: list[str] = []
    for value in prior_texts:
        prior_sentences.extend(_split_into_sentences(value))

    if not prior_sentences:
        return 0.0

    redundant_count = 0
    for sentence in current_sentences:
        sentence_norm = sentence.lower()
        for prior_sentence in prior_sentences:
            prior_norm = prior_sentence.lower()
            sequence_similarity = SequenceMatcher(None, sentence_norm, prior_norm).ratio()
            token_overlap = _token_overlap_ratio(sentence_norm, prior_norm)
            if sequence_similarity >= SECTION_SIMILARITY_THRESHOLD:
                redundant_count += 1
                break
            if token_overlap >= SECTION_TOKEN_OVERLAP_THRESHOLD:
                redundant_count += 1
                break

    return redundant_count / float(len(current_sentences))


def _enforce_section_novelty(current_text: str, prior_texts: list[str]) -> str:
    normalized = _normalize_continuous_prose(current_text)
    deduplicated = _normalize_continuous_prose(_deduplicate_section_text(normalized, prior_texts))
    if _section_redundancy_ratio(deduplicated, prior_texts) <= SECTION_REDUNDANCY_RATIO_THRESHOLD:
        return deduplicated
    return deduplicated


def _content_hash(text_value: str) -> str:
    normalized = _normalize_continuous_prose(text_value)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _redundancy_stats(current_text: str, prior_texts: list[str]) -> tuple[int, int, float]:
    current_sentences = _split_into_sentences(current_text)
    if not current_sentences:
        return 0, 0, 0.0

    prior_sentences: list[str] = []
    for value in prior_texts:
        prior_sentences.extend(_split_into_sentences(value))

    if not prior_sentences:
        return 0, len(current_sentences), 0.0

    overlap_count = 0
    for sentence in current_sentences:
        sentence_norm = sentence.lower()
        for prior_sentence in prior_sentences:
            prior_norm = prior_sentence.lower()
            if sentence_norm == prior_norm:
                overlap_count += 1
                break
            if SequenceMatcher(None, sentence_norm, prior_norm).ratio() >= SECTION_SIMILARITY_THRESHOLD:
                overlap_count += 1
                break
            if _token_overlap_ratio(sentence_norm, prior_norm) >= SECTION_TOKEN_OVERLAP_THRESHOLD:
                overlap_count += 1
                break

    total = len(current_sentences)
    return overlap_count, total, (overlap_count / float(total)) if total else 0.0


def _pairwise_overlap_ratio(left_text: str, right_text: str) -> float:
    left = _normalize_continuous_prose(left_text)
    right = _normalize_continuous_prose(right_text)
    if not left or not right:
        return 0.0
    return max(_section_redundancy_ratio(left, [right]), _section_redundancy_ratio(right, [left]))


def _record_novelty_ledger_entry(
    *,
    user_id: str,
    run_id: str,
    source_id: int | None,
    document_source_ids: list[int],
    section_type: str,
    section_text: str,
    prior_texts: list[str],
    expansion_applied: bool,
) -> None:
    overlap_count, total_count, overlap_ratio = _redundancy_stats(section_text, prior_texts)

    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO generation_novelty_ledger (
                    user_id,
                    run_id,
                    source_id,
                    document_source_ids_json,
                    section_type,
                    content_hash,
                    similarity_to_prior,
                    overlap_sentences_count,
                    total_sentences_count,
                    expansion_applied
                ) VALUES (
                    :user_id,
                    :run_id,
                    :source_id,
                    :document_source_ids_json,
                    :section_type,
                    :content_hash,
                    :similarity_to_prior,
                    :overlap_sentences_count,
                    :total_sentences_count,
                    :expansion_applied
                )
                """
            ),
            {
                "user_id": user_id,
                "run_id": run_id,
                "source_id": source_id,
                "document_source_ids_json": _serialize_ids(document_source_ids),
                "section_type": section_type,
                "content_hash": _content_hash(section_text),
                "similarity_to_prior": overlap_ratio,
                "overlap_sentences_count": overlap_count,
                "total_sentences_count": total_count,
                "expansion_applied": 1 if expansion_applied else 0,
            },
        )


def _record_pairwise_quality_metric(
    *,
    run_id: str,
    user_id: str,
    section_type: str,
    pair_key: str,
    overlap_ratio: float,
) -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO generation_quality_metrics (
                    run_id,
                    user_id,
                    section_type,
                    pair_key,
                    overlap_ratio
                ) VALUES (
                    :run_id,
                    :user_id,
                    :section_type,
                    :pair_key,
                    :overlap_ratio
                )
                """
            ),
            {
                "run_id": run_id,
                "user_id": user_id,
                "section_type": section_type,
                "pair_key": pair_key,
                "overlap_ratio": overlap_ratio,
            },
        )


def _assert_pairwise_uniqueness(
    *,
    run_id: str,
    user_id: str,
    section_group: str,
    section_texts: dict[str, str],
    max_overlap_ratio: float,
) -> None:
    labels = [label for label, text_value in section_texts.items() if _normalize_continuous_prose(text_value)]
    for index, left_label in enumerate(labels):
        for right_label in labels[index + 1:]:
            overlap_ratio = _pairwise_overlap_ratio(
                section_texts[left_label],
                section_texts[right_label],
            )
            pair_key = f"{left_label}::{right_label}"
            _record_pairwise_quality_metric(
                run_id=run_id,
                user_id=user_id,
                section_type=section_group,
                pair_key=pair_key,
                overlap_ratio=overlap_ratio,
            )
            if overlap_ratio > max_overlap_ratio:
                raise ValueError(
                    f"{section_group} overlap gate failed for {pair_key}: "
                    f"{overlap_ratio:.3f} > {max_overlap_ratio:.3f}"
                )


def _expand_nonredundant_text(
    *,
    run_id: str,
    stage_name: str,
    user_id: str,
    context_label: str,
    base_text: str,
    avoid_texts: list[str],
    grounding_chunks: list[str],
    key_terms: list[str] | None = None,
    extra_context: str = "",
) -> str:
    trimmed_grounding = "\n\n".join(grounding_chunks)
    if len(trimmed_grounding) > MAX_EXPANSION_GROUNDING_CHARS:
        trimmed_grounding = trimmed_grounding[:MAX_EXPANSION_GROUNDING_CHARS].rstrip() + "..."

    avoid_block = "\n\n".join(
        _truncate_for_prompt(value, limit=700)
        for value in avoid_texts
        if str(value or "").strip()
    )

    key_terms_text = ", ".join(term for term in (key_terms or []) if term.strip())

    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=stage_name,
        user_id=user_id,
        system_prompt=(
            "You rewrite and deepen grounded learning text. Return only JSON. "
            "Write continuous paragraph prose with no markdown headings or bullet lists. "
            "The output must be non-redundant against the forbidden overlap text while adding new, mechanism-level explanation."
        ),
        user_prompt=(
            f"Context label: {context_label}\n\n"
            f"Base draft to improve:\n{_truncate_for_prompt(base_text, limit=2400)}\n\n"
            f"Forbidden overlap text (do not restate these claims with similar phrasing):\n{avoid_block or '[none]'}\n\n"
            f"Key terms to connect (optional): {key_terms_text or '[none]'}\n\n"
            f"Extra context:\n{extra_context or '[none]'}\n\n"
            f"Grounding chunks:\n{trimmed_grounding or '[none]'}\n\n"
            "Rewrite into a richer layman-friendly but technically faithful explanation. "
            "Prioritize causal chains, assumptions, constraints, tradeoffs, and edge-case behavior. "
            "Avoid repeating base-draft phrasing; add fresh structure and examples from grounded context."
        ),
        response_schema=TEXT_EXPANSION_JSON_SCHEMA,
        required_keys=["expanded_text"],
    )
    return str(payload.get("expanded_text") or "").strip()


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
    progression_outline: str,
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
            f"Progression outline:\n{progression_outline}\n\n"
            "Grounding chunks:\n"
            f"{chunk_text}"
        ),
        response_schema=SOURCE_DEEP_DIVE_JSON_SCHEMA,
        required_keys=["deep_dive_text", "key_terms"],
    )
    deep_dive_text = _truncate_to_max_words(
        _normalize_continuous_prose(str(payload.get("deep_dive_text", "")).strip()),
        max_words=MAX_DEEP_DIVE_WORDS,
    )
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
    progression_outline: str,
    grounding_chunks: list[str],
    user_id: str,
    run_id: str,
) -> list[ReflectionPoint]:
    chunk_text = "\n\n".join(grounding_chunks) if grounding_chunks else "[No grounding chunks available]"
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_reflection:{source_type}",
        user_id=user_id,
        system_prompt=SOCRATIC_REFLECTION_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Summary:\n{summary_text}\n\n"
            f"Deep dive:\n{deep_dive_text}\n\n"
            f"Progression outline:\n{progression_outline}\n\n"
            "Representative grounding chunks:\n"
            f"{chunk_text}\n\n"
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
    progression_outline: str,
    user_id: str,
    run_id: str,
) -> tuple[str, list[str], list[KeyTermExplanation]]:
    chunk_text = "\n\n".join(grounding_chunks) if grounding_chunks else "[No grounding chunks available]"
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_under_surface:{source_type}",
        user_id=user_id,
        system_prompt=UNDER_SURFACE_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Summary:\n{summary_text}\n\n"
            f"Deep dive:\n{deep_dive_text}\n\n"
            f"Progression outline:\n{progression_outline}\n\n"
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

    grounding_rows = _load_grounding_chunk_rows(source_id, user_id)
    grounding_chunks, deep_dive_chunk_indexes = _select_distributed_grounding_chunks(
        grounding_rows,
        max_chunks=MAX_GROUNDING_CHUNKS,
        max_chars=MAX_GROUNDING_CHARS,
    )
    under_surface_chunks, under_surface_chunk_indexes = _select_distributed_grounding_chunks(
        grounding_rows,
        max_chunks=MAX_UNDER_SURFACE_GROUNDING_CHUNKS,
        max_chars=MAX_UNDER_SURFACE_GROUNDING_CHARS,
    )
    reflection_chunks, reflection_chunk_indexes = _select_distributed_grounding_chunks(
        grounding_rows,
        max_chunks=MAX_REFLECTION_GROUNDING_CHUNKS,
        max_chars=MAX_REFLECTION_GROUNDING_CHARS,
    )

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name=f"source_grounding_selection:{source_type}",
        attempt_number=1,
        status="succeeded",
        details=(
            f"total_rows={len(grounding_rows)};"
            f"deep_dive={deep_dive_chunk_indexes};"
            f"under_surface={under_surface_chunk_indexes};"
            f"reflection={reflection_chunk_indexes}"
        ),
    )

    if _should_use_model_progression_outline(
        total_rows=len(grounding_rows),
        grounding_chunks=grounding_chunks,
    ):
        progression_outline = _generate_source_progression_outline(
            summary_text=summary_text,
            source_type=source_type,
            grounding_chunks=grounding_chunks,
            user_id=user_id,
            run_id=run_id,
        )
    else:
        progression_outline = _build_heuristic_progression_outline(
            summary_text=summary_text,
            grounding_chunks=grounding_chunks,
        )
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=f"source_progression_outline:{source_type}",
            attempt_number=1,
            status="skipped",
            details=(
                "heuristic_short_source;"
                f"total_rows={len(grounding_rows)};"
                f"grounding_chunk_count={len(grounding_chunks)}"
            ),
        )
    generated_title = _generate_source_title(summary_text, source_type, user_id, run_id)
    deep_dive_text, key_terms = _generate_source_deep_dive(
        summary_text,
        source_type,
        grounding_chunks,
        progression_outline,
        user_id,
        run_id,
    )
    deep_dive_text = _truncate_to_max_words(
        _enforce_section_novelty(deep_dive_text, [summary_text, progression_outline]),
        max_words=MAX_DEEP_DIVE_WORDS,
    )

    if _section_redundancy_ratio(deep_dive_text, [summary_text, progression_outline]) > SECTION_REDUNDANCY_RATIO_THRESHOLD:
        try:
            refined_deep_dive = _expand_nonredundant_text(
                run_id=run_id,
                stage_name=f"source_deep_dive_refine:{source_type}",
                user_id=user_id,
                context_label=f"{source_type} deep dive novelty refinement",
                base_text=deep_dive_text,
                avoid_texts=[summary_text, progression_outline],
                grounding_chunks=grounding_chunks,
                key_terms=key_terms,
                extra_context=(
                    "Add new mechanism-level detail not already present in summary/progression text. "
                    "Include boundary conditions and one fresh practical implication that was not previously stated."
                ),
            )
            refined_deep_dive = _truncate_to_max_words(
                _enforce_section_novelty(refined_deep_dive, [summary_text, progression_outline]),
                max_words=MAX_DEEP_DIVE_WORDS,
            )
            if refined_deep_dive:
                deep_dive_text = refined_deep_dive
        except Exception as exc:
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=f"source_deep_dive_refine:{source_type}",
                attempt_number=GENERATION_MAX_STAGE_ATTEMPTS,
                status="skipped",
                details=f"fallback_to_first_pass={_trim_error_detail(str(exc))}",
            )

    reflection_points = _generate_reflection_points(
        summary_text,
        deep_dive_text,
        source_type,
        progression_outline,
        reflection_chunks,
        user_id,
        run_id,
    )
    reflection_points = [
        point.model_copy(
            update={
                "explanation": _enforce_section_novelty(
                    point.explanation,
                    [summary_text, deep_dive_text],
                ),
                "under_the_hood": _enforce_section_novelty(
                    point.under_the_hood,
                    [summary_text, deep_dive_text, point.explanation],
                ),
            }
        )
        for point in reflection_points
    ]
    under_surface_explainer, diagnostic_checklist, key_term_explanations = _generate_under_surface_pack(
        summary_text=summary_text,
        deep_dive_text=deep_dive_text,
        key_terms=key_terms,
        source_type=source_type,
        grounding_chunks=under_surface_chunks,
        progression_outline=progression_outline,
        user_id=user_id,
        run_id=run_id,
    )
    under_surface_explainer = _deduplicate_section_text(
        under_surface_explainer,
        [summary_text, deep_dive_text],
    )
    under_surface_explainer = _normalize_continuous_prose(under_surface_explainer)

    try:
        expanded_under_surface = _expand_nonredundant_text(
            run_id=run_id,
            stage_name=f"source_under_surface_expand:{source_type}",
            user_id=user_id,
            context_label=f"{source_type} under-surface explainer",
            base_text=under_surface_explainer,
            avoid_texts=[summary_text, deep_dive_text],
            grounding_chunks=under_surface_chunks,
            key_terms=key_terms,
            extra_context=(
                f"Progression outline:\n{_truncate_for_prompt(progression_outline, limit=1200)}"
            ),
        )
        expanded_under_surface = _deduplicate_section_text(
            _normalize_continuous_prose(expanded_under_surface),
            [summary_text, deep_dive_text, under_surface_explainer],
        )
        if expanded_under_surface:
            under_surface_explainer = expanded_under_surface
    except Exception as exc:
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=f"source_under_surface_expand:{source_type}",
            attempt_number=GENERATION_MAX_STAGE_ATTEMPTS,
            status="skipped",
            details=f"fallback_to_first_pass={_trim_error_detail(str(exc))}",
        )

    under_surface_explainer = _normalize_continuous_prose(under_surface_explainer)

    reflection_text = " ".join(
        " ".join(
            part
            for part in [point.question, point.explanation, point.under_the_hood]
            if part
        )
        for point in reflection_points
    ).strip()

    source_section_texts = {
        "summary": summary_text,
        "deep_dive": deep_dive_text,
        "under_surface": under_surface_explainer,
        "reflection": reflection_text,
    }
    try:
        _assert_pairwise_uniqueness(
            run_id=run_id,
            user_id=user_id,
            section_group=f"source:{source_id}",
            section_texts=source_section_texts,
            max_overlap_ratio=SOURCE_SECTION_PAIR_OVERLAP_THRESHOLD,
        )
    except Exception as exc:
        try:
            refined_deep_dive = _expand_nonredundant_text(
                run_id=run_id,
                stage_name=f"source_role_refine_deep:{source_type}",
                user_id=user_id,
                context_label=f"{source_type} deep dive role-separation refinement",
                base_text=deep_dive_text,
                avoid_texts=[summary_text, under_surface_explainer, reflection_text],
                grounding_chunks=grounding_chunks,
                key_terms=key_terms,
                extra_context=(
                    "Keep this section focused on mechanism-level explanation only. "
                    "Do not restate reflection content or practical checklist language."
                ),
            )
            if refined_deep_dive:
                deep_dive_text = _truncate_to_max_words(
                    _enforce_section_novelty(refined_deep_dive, [summary_text, under_surface_explainer, reflection_text]),
                    max_words=MAX_DEEP_DIVE_WORDS,
                )

            refined_under_surface = _expand_nonredundant_text(
                run_id=run_id,
                stage_name=f"source_role_refine_under:{source_type}",
                user_id=user_id,
                context_label=f"{source_type} under-surface role-separation refinement",
                base_text=under_surface_explainer,
                avoid_texts=[summary_text, deep_dive_text, reflection_text],
                grounding_chunks=under_surface_chunks,
                key_terms=key_terms,
                extra_context=(
                    "Keep this section focused on first-principles and theoretical lens only. "
                    "Do not repeat deep-dive sequence descriptions or reflection prompts."
                ),
            )
            if refined_under_surface:
                under_surface_explainer = _enforce_section_novelty(
                    refined_under_surface,
                    [summary_text, deep_dive_text, reflection_text],
                )

            _assert_pairwise_uniqueness(
                run_id=run_id,
                user_id=user_id,
                section_group=f"source:{source_id}",
                section_texts={
                    "summary": summary_text,
                    "deep_dive": deep_dive_text,
                    "under_surface": under_surface_explainer,
                    "reflection": reflection_text,
                },
                max_overlap_ratio=SOURCE_SECTION_PAIR_OVERLAP_THRESHOLD,
            )
        except Exception as refinement_exc:
            raise ValueError(
                f"Source section uniqueness gate failed for source {source_id}: {refinement_exc}"
            ) from exc

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
        integrated_explanation = _normalize_continuous_prose(
            str(raw_intersection.get("integrated_explanation", "")).strip()
        )
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

    layman_bridge = _normalize_continuous_prose(str(payload.get("layman_bridge", "")).strip())
    synthesis_text = _normalize_continuous_prose(str(payload.get("synthesis_text", "")).strip())
    if len(intersections) < 1:
        raise ValueError("Model returned too few integrated cross-source intersections.")
    if not layman_bridge:
        raise ValueError("Model returned an empty layman bridge.")
    if not synthesis_text:
        raise ValueError("Model returned an empty synthesis text.")

    source_background = [
        video_section.summary_text,
        video_section.deep_dive_text,
        video_section.under_surface_explainer,
    ] + [
        value
        for section in document_sections
        for value in [section.summary_text, section.deep_dive_text, section.under_surface_explainer]
    ]
    shared_grounding_chunks = [
        (
            f"{section.source_type.title()} [{section.generated_title}]\n"
            f"Summary: {_truncate_for_prompt(section.summary_text, limit=320)}\n"
            f"Deep dive: {_truncate_for_prompt(section.deep_dive_text, limit=380)}\n"
            f"Under surface: {_truncate_for_prompt(section.under_surface_explainer, limit=380)}"
        )
        for section in [video_section, *document_sections]
    ]

    refined_intersections: list[InsightIntersection] = []
    prior_intersection_explanations: list[str] = []
    for index, intersection in enumerate(intersections, start=1):
        try:
            expanded_intersection = _expand_nonredundant_text(
                run_id=run_id,
                stage_name=f"combined_intersection_expand:{index}",
                user_id=user_id,
                context_label=f"cross-source intersection: {intersection.intersection_title}",
                base_text=intersection.integrated_explanation,
                avoid_texts=[
                    intersection.why_it_matters,
                    layman_bridge,
                    *prior_intersection_explanations,
                    *source_background,
                ],
                grounding_chunks=shared_grounding_chunks,
                key_terms=[
                    term
                    for sentence in intersection.attributed_sentences
                    for term in sentence.emphasis_terms
                ],
            )
            expanded_intersection = _deduplicate_section_text(
                _normalize_continuous_prose(expanded_intersection),
                [intersection.why_it_matters, *prior_intersection_explanations, *source_background],
            )
            if expanded_intersection:
                refined_intersections.append(
                    intersection.model_copy(update={"integrated_explanation": expanded_intersection})
                )
                prior_intersection_explanations.append(expanded_intersection)
                continue
        except Exception:
            pass

        fallback_intersection = _normalize_continuous_prose(
            _deduplicate_section_text(
                intersection.integrated_explanation,
                [intersection.why_it_matters, *prior_intersection_explanations, *source_background],
            )
        )
        refined_intersections.append(
            intersection.model_copy(update={"integrated_explanation": fallback_intersection})
        )
        prior_intersection_explanations.append(fallback_intersection)

    intersections = refined_intersections
    intersection_evidence_texts = [
        sentence.text
        for entry in intersections
        for sentence in entry.attributed_sentences
        if sentence.text.strip()
    ]

    layman_bridge = _enforce_section_novelty(
        layman_bridge,
        [
            *[entry.why_it_matters for entry in intersections],
            *[entry.integrated_explanation for entry in intersections],
            *intersection_evidence_texts,
            *source_background,
        ],
    )

    try:
        expanded_synthesis = _expand_nonredundant_text(
            run_id=run_id,
            stage_name="combined_synthesis_expand",
            user_id=user_id,
            context_label="cross-source synthesis",
            base_text=synthesis_text,
            avoid_texts=[
                layman_bridge,
                *[entry.why_it_matters for entry in intersections],
                *source_background,
            ],
            grounding_chunks=shared_grounding_chunks,
            key_terms=[
                term
                for entry in intersections
                for sentence in entry.attributed_sentences
                for term in sentence.emphasis_terms
            ],
            extra_context=(
                "Relationship insights:\n"
                + "\n".join(_truncate_for_prompt(item, limit=260) for item in relationship_insights[:8])
            ),
        )
        expanded_synthesis = _normalize_continuous_prose(
            _deduplicate_section_text(
                expanded_synthesis,
                [layman_bridge, *[entry.integrated_explanation for entry in intersections], *source_background],
            )
        )
        if expanded_synthesis:
            synthesis_text = expanded_synthesis
    except Exception:
        synthesis_text = _normalize_continuous_prose(synthesis_text)

    synthesis_text = _enforce_section_novelty(
        synthesis_text,
        [
            layman_bridge,
            *[entry.why_it_matters for entry in intersections],
            *[entry.integrated_explanation for entry in intersections],
            *intersection_evidence_texts,
            *source_background,
        ],
    )

    layman_bridge = _normalize_continuous_prose(layman_bridge)

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
    comparative_analysis = _normalize_continuous_prose(str(result.get("comparative_analysis", "")).strip())
    if not comparative_analysis:
        raise ValueError("Model returned empty comparative analysis text.")

    source_background = [
        video_section.summary_text,
        video_section.deep_dive_text,
        insights_section.synthesis_text,
    ] + [
        value
        for section in document_sections
        for value in [section.summary_text, section.deep_dive_text]
    ]
    grounding_chunks = [
        (
            f"{section.source_type.title()} [{section.generated_title}]\n"
            f"Summary: {_truncate_for_prompt(section.summary_text, limit=320)}\n"
            f"Deep dive: {_truncate_for_prompt(section.deep_dive_text, limit=360)}"
        )
        for section in [video_section, *document_sections]
    ]

    try:
        expanded_analysis = _expand_nonredundant_text(
            run_id=run_id,
            stage_name="comparative_analysis_expand",
            user_id=user_id,
            context_label="cross-source comparative deepening",
            base_text=comparative_analysis,
            avoid_texts=[
                insights_section.layman_bridge,
                insights_section.synthesis_text,
                *[entry.why_it_matters for entry in insights_section.intersections],
                *[entry.integrated_explanation for entry in insights_section.intersections],
                *source_background,
            ],
            grounding_chunks=grounding_chunks,
            key_terms=[
                term
                for entry in insights_section.intersections
                for sentence in entry.attributed_sentences
                for term in sentence.emphasis_terms
            ],
            extra_context=(
                "Relationship insights:\n"
                + "\n".join(_truncate_for_prompt(item, limit=260) for item in relationship_insights[:8])
            ),
        )
        expanded_analysis = _normalize_continuous_prose(
            _deduplicate_section_text(
                expanded_analysis,
                [insights_section.synthesis_text, insights_section.layman_bridge, *source_background],
            )
        )
        if expanded_analysis:
            comparative_analysis = expanded_analysis
    except Exception:
        comparative_analysis = _normalize_continuous_prose(comparative_analysis)

    comparative_analysis = _normalize_continuous_prose(
        _deduplicate_section_text(
            comparative_analysis,
            [
                insights_section.synthesis_text,
                insights_section.layman_bridge,
                *[entry.integrated_explanation for entry in insights_section.intersections],
            ],
        )
    )

    comparative_analysis = _enforce_section_novelty(
        comparative_analysis,
        [
            insights_section.synthesis_text,
            insights_section.layman_bridge,
            *[entry.why_it_matters for entry in insights_section.intersections],
            *[entry.integrated_explanation for entry in insights_section.intersections],
            *[
                sentence.text
                for entry in insights_section.intersections
                for sentence in entry.attributed_sentences
            ],
        ],
    )

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
    scenarios: list[ApplicationScenario] = []
    for item in result.get("application_scenarios", []):
        try:
            scenario = ApplicationScenario.model_validate(item)
        except Exception:
            continue

        normalized_steps: list[str] = []
        for step in scenario.transfer_steps:
            normalized_step = _normalize_continuous_prose(step)
            if normalized_step:
                normalized_steps.append(normalized_step)
        scenarios.append(
            scenario.model_copy(
                update={
                    "scenario_prompt": _normalize_continuous_prose(scenario.scenario_prompt),
                    "transfer_steps": normalized_steps,
                    "common_pitfall": _normalize_continuous_prose(scenario.common_pitfall),
                }
            )
        )
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


def _truncate_sentence(text_value: str, limit: int = 260) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if not normalized:
        return ""

    first_sentence = normalized.split(". ", 1)[0].strip()
    candidate = first_sentence if first_sentence else normalized
    if len(candidate) > limit:
        return candidate[: limit - 3].rstrip() + "..."
    return candidate


def _build_noncomparative_insights_and_quiz(
    source_sections: list[SourceLearningSection],
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    if not source_sections:
        raise ValueError("source_sections cannot be empty.")

    intersections: list[InsightIntersection] = []
    for section in source_sections[:3]:
        summary_sentence = _truncate_sentence(section.summary_text, limit=240)
        deep_sentence = _truncate_sentence(section.deep_dive_text, limit=240)
        emphasis_terms = [term for term in section.key_terms[:4] if term.strip()]

        attributed_sentences: list[AttributedSentence] = []
        for snippet in [summary_sentence, deep_sentence]:
            if not snippet:
                continue
            attributed_sentences.append(
                AttributedSentence(
                    text=snippet,
                    source_id=section.source_id,
                    source_type=section.source_type,
                    emphasis_terms=emphasis_terms,
                )
            )

        if not attributed_sentences:
            attributed_sentences.append(
                AttributedSentence(
                    text=f"Focus on the core mechanism behind '{section.generated_title}'.",
                    source_id=section.source_id,
                    source_type=section.source_type,
                    emphasis_terms=emphasis_terms,
                )
            )

        intersections.append(
            InsightIntersection(
                intersection_title=f"Focus Area: {section.generated_title}",
                why_it_matters=(
                    summary_sentence
                    or f"This source defines the core mechanism for {section.generated_title}."
                ),
                integrated_explanation=(
                    deep_sentence
                    or "Use the deep-dive section and reflection points to pressure-test understanding."
                ),
                attributed_sentences=attributed_sentences,
            )
        )

    parallels = intersections[0].attributed_sentences if intersections else []
    primary = source_sections[0]
    layman_bridge = _truncate_sentence(primary.under_surface_explainer, limit=320) or _truncate_sentence(
        primary.summary_text,
        limit=320,
    )
    source_labels = ", ".join(section.generated_title for section in source_sections[:4])
    synthesis_text = (
        f"This run has {len(source_sections)} source(s): {source_labels}. "
        "Use each source section's deep dive, checklist, and reflection points to reinforce understanding before asking follow-up questions in chat."
    )

    quiz_questions: list[QuizQuestion] = []
    for section in source_sections[:2]:
        key_claim = _truncate_sentence(section.summary_text, limit=180) or section.generated_title
        under_surface = _truncate_sentence(section.under_surface_explainer, limit=220)
        source_evidence = [
            value for value in [
                _truncate_sentence(section.summary_text, limit=180),
                _truncate_sentence(section.deep_dive_text, limit=180),
            ] if value
        ]

        quiz_questions.append(
            QuizQuestion(
                question=(
                    f"For '{section.generated_title}', which statement best matches the grounded explanation in your material?"
                ),
                options=[
                    key_claim,
                    "The source mainly argues from unsupported assumptions.",
                    "The source presents no actionable mechanism.",
                    "The source rejects the central idea entirely.",
                ],
                answer_index=0,
                explanation="The first option restates the grounded claim from your source summary.",
                under_the_hood=under_surface or "Look for the mechanism and boundary conditions in the deep dive.",
                difficulty_level="foundational",
                question_type="synthesis",
                source_evidence=source_evidence or [f"Review source {section.source_id} summary and deep dive."],
            )
        )

    study_advice = (
        "If this run has one source, ask the chatbox to test assumptions and edge cases. "
        "If it has multiple sources, ask for agreements, tensions, and transfer steps across them."
    )

    insights_section = CombinedInsightSection(
        intersections=intersections,
        parallels=parallels,
        layman_bridge=layman_bridge or "Use the source summaries and deep dives as your grounding layer.",
        synthesis_text=synthesis_text,
        comparative_analysis="",
        application_scenarios=[],
        model_name="deterministic-noncomparative",
        schema_version=4,
    )
    quiz_section = CombinedQuizSection(
        questions=quiz_questions,
        study_advice=study_advice,
        model_name="deterministic-noncomparative",
        schema_version=2,
    )
    return insights_section, quiz_section


def _build_source_section_text_map(section: SourceLearningSection) -> dict[str, str]:
    reflection_text = " ".join(
        " ".join(
            part
            for part in [point.question.strip(), point.explanation.strip(), point.under_the_hood.strip()]
            if part
        )
        for point in section.reflection_points
    ).strip()
    key_term_text = " ".join(
        f"{entry.term.strip()}: {entry.explanation.strip()}"
        for entry in section.key_term_explanations
        if entry.term.strip() and entry.explanation.strip()
    ).strip()
    checklist_text = " ".join(item.strip() for item in section.diagnostic_checklist if item.strip()).strip()

    return {
        "summary": section.summary_text,
        "deep_dive": section.deep_dive_text,
        "under_surface": section.under_surface_explainer,
        "reflection": reflection_text,
        "key_terms": key_term_text,
        "diagnostics": checklist_text,
    }


def _build_cross_section_text_map(
    insights: CombinedInsightSection,
    quiz: CombinedQuizSection,
) -> dict[str, str]:
    intersections_text = " ".join(
        " ".join(
            part
            for part in [
                entry.intersection_title.strip(),
                entry.why_it_matters.strip(),
                entry.integrated_explanation.strip(),
                " ".join(sentence.text.strip() for sentence in entry.attributed_sentences if sentence.text.strip()),
            ]
            if part
        )
        for entry in insights.intersections
    ).strip()

    scenarios_text = " ".join(
        " ".join(
            part
            for part in [
                scenario.scenario_title.strip(),
                scenario.scenario_prompt.strip(),
                " ".join(step.strip() for step in scenario.transfer_steps if step.strip()),
                scenario.common_pitfall.strip(),
            ]
            if part
        )
        for scenario in insights.application_scenarios
    ).strip()

    quiz_explanations = " ".join(
        " ".join(
            part
            for part in [
                question.explanation.strip(),
                question.under_the_hood.strip(),
                " ".join(item.strip() for item in question.source_evidence if item.strip()),
            ]
            if part
        )
        for question in quiz.questions
    ).strip()

    return {
        "intersections": intersections_text,
        "bridge": insights.layman_bridge,
        "integrated_analysis": " ".join(
            part
            for part in [insights.synthesis_text, insights.comparative_analysis]
            if part.strip()
        ).strip(),
        "scenarios": scenarios_text,
        "quiz": quiz_explanations,
    }


def generate_tailored_learning(
    video_source_id: int | None,
    document_source_ids: list[int],
    user_id: str,
) -> GenerateTailoredLearningResponse:
    ensure_generation_tables()
    run_id = f"gen-{uuid.uuid4().hex[:12]}"

    normalized_video_id, normalized_document_ids = _normalize_ids(
        video_source_id,
        document_source_ids,
    )
    source_ids: list[int] = []
    if normalized_video_id is not None:
        source_ids.append(normalized_video_id)
    source_ids.extend(normalized_document_ids)

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name="pipeline_start",
        attempt_number=1,
        status="started",
        details=(
            f"video_source_id={normalized_video_id if normalized_video_id is not None else 'none'}; "
            f"document_source_ids={','.join(str(value) for value in normalized_document_ids)}"
        ),
    )

    if normalized_video_id is not None:
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

    if normalized_video_id is not None and normalized_document_ids:
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

    video_section = (
        _build_source_learning_section(normalized_video_id, user_id, run_id)
        if normalized_video_id is not None
        else None
    )
    document_sections = [
        _build_source_learning_section(source_id, user_id, run_id)
        for source_id in normalized_document_ids
    ]

    all_sections = ([video_section] if video_section is not None else []) + document_sections
    if not all_sections:
        raise ValueError("No source learning sections were generated.")

    for section in all_sections:
        section_text_map = _build_source_section_text_map(section)
        source_prior_map: dict[str, list[str]] = {
            "summary": [],
            "deep_dive": [section.summary_text],
            "under_surface": [section.summary_text, section.deep_dive_text],
            "reflection": [section.summary_text, section.deep_dive_text, section.under_surface_explainer],
            "key_terms": [section.summary_text, section.deep_dive_text],
            "diagnostics": [section.summary_text, section.deep_dive_text, section.under_surface_explainer],
        }
        for section_type, text_value in section_text_map.items():
            if not _normalize_continuous_prose(text_value):
                continue
            _record_novelty_ledger_entry(
                user_id=user_id,
                run_id=run_id,
                source_id=section.source_id,
                document_source_ids=[],
                section_type=f"source:{section_type}",
                section_text=text_value,
                prior_texts=source_prior_map.get(section_type, []),
                expansion_applied=False,
            )

    if video_section is not None and normalized_document_ids:
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

        relationship_insights = _load_relationship_insights(source_ids, user_id)

        cache_was_accepted = False
        if cached_combined is not None:
            cached_insights, cached_quiz = cached_combined
            try:
                _assert_pairwise_uniqueness(
                    run_id=run_id,
                    user_id=user_id,
                    section_group="cross-source",
                    section_texts=_build_cross_section_text_map(cached_insights, cached_quiz),
                    max_overlap_ratio=CROSS_SECTION_PAIR_OVERLAP_THRESHOLD,
                )
                insights_section, quiz_section = cached_insights, cached_quiz
                cache_was_accepted = True
            except Exception as exc:
                log_generation_stage_event(
                    run_id=run_id,
                    user_id=user_id,
                    stage_name="combined_learning_cache_gate",
                    attempt_number=1,
                    status="failed",
                    details=_trim_error_detail(str(exc)),
                )

        if not cache_was_accepted:
            last_error: str | None = None
            for combined_attempt in range(1, GENERATION_MAX_STAGE_ATTEMPTS + 1):
                try:
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
                    candidate_insights = insights_section.model_copy(
                        update={
                            "comparative_analysis": comparative_analysis,
                            "application_scenarios": application_scenarios,
                        }
                    )
                    _assert_pairwise_uniqueness(
                        run_id=run_id,
                        user_id=user_id,
                        section_group="cross-source",
                        section_texts=_build_cross_section_text_map(candidate_insights, quiz_section),
                        max_overlap_ratio=CROSS_SECTION_PAIR_OVERLAP_THRESHOLD,
                    )
                    insights_section = candidate_insights
                    _store_combined_learning_sections(
                        user_id=user_id,
                        video_source_id=normalized_video_id,
                        document_source_ids=normalized_document_ids,
                        insights=insights_section,
                        quiz=quiz_section,
                    )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    log_generation_stage_event(
                        run_id=run_id,
                        user_id=user_id,
                        stage_name="cross_section_uniqueness_gate",
                        attempt_number=combined_attempt,
                        status="failed",
                        details=_trim_error_detail(last_error),
                    )
                    if combined_attempt >= GENERATION_MAX_STAGE_ATTEMPTS:
                        raise GenerationStageError(
                            run_id=run_id,
                            stage_name="cross_section_uniqueness_gate",
                            attempt_number=combined_attempt,
                            reason=_trim_error_detail(last_error or "unknown error"),
                        ) from exc

        cross_section_text_map = _build_cross_section_text_map(insights_section, quiz_section)
        cross_prior_map: dict[str, list[str]] = {
            "intersections": [],
            "bridge": [cross_section_text_map.get("intersections", "")],
            "integrated_analysis": [
                cross_section_text_map.get("intersections", ""),
                cross_section_text_map.get("bridge", ""),
            ],
            "scenarios": [
                cross_section_text_map.get("intersections", ""),
                cross_section_text_map.get("bridge", ""),
                cross_section_text_map.get("integrated_analysis", ""),
            ],
            "quiz": [
                cross_section_text_map.get("intersections", ""),
                cross_section_text_map.get("bridge", ""),
                cross_section_text_map.get("integrated_analysis", ""),
                cross_section_text_map.get("scenarios", ""),
            ],
        }
        for section_type, text_value in cross_section_text_map.items():
            if not _normalize_continuous_prose(text_value):
                continue
            _record_novelty_ledger_entry(
                user_id=user_id,
                run_id=run_id,
                source_id=None,
                document_source_ids=normalized_document_ids,
                section_type=f"cross:{section_type}",
                section_text=text_value,
                prior_texts=cross_prior_map.get(section_type, []),
                expansion_applied=False,
            )
    else:
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name="combined_learning_cache",
            attempt_number=1,
            status="skipped",
            details="Non-comparative run: skipping combined comparative generation stages.",
        )
        insights_section, quiz_section = _build_noncomparative_insights_and_quiz(all_sections)
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name="noncomparative_generation",
            attempt_number=1,
            status="succeeded",
            details=f"source_count={len(all_sections)}",
        )

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name="pipeline_end",
        attempt_number=1,
        status="succeeded",
        details=(
            f"video_source_id={normalized_video_id if normalized_video_id is not None else 'none'}; "
            f"document_count={len(normalized_document_ids)}"
        ),
    )

    return GenerateTailoredLearningResponse(
        status_message="Tailored Socratic learning generated successfully.",
        source_ids=source_ids,
        video=video_section,
        documents=document_sections,
        insights=insights_section,
        quiz=quiz_section,
    )
