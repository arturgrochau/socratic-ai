from __future__ import annotations

import json

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.linking import link_source_pair
from app.models import (
    AttributedSentence,
    CombinedInsightSection,
    CombinedQuizSection,
    GenerateTailoredLearningResponse,
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
from prompts.socratic_reflection import (
    SOCRATIC_REFLECTION_JSON_SCHEMA,
    SOCRATIC_REFLECTION_SYSTEM_PROMPT,
)
from prompts.source_deep_dive import (
    SOURCE_DEEP_DIVE_JSON_SCHEMA,
    SOURCE_DEEP_DIVE_SYSTEM_PROMPT,
)
from prompts.source_title import SOURCE_TITLE_JSON_SCHEMA, SOURCE_TITLE_SYSTEM_PROMPT


GENERATION_SCHEMA_VERSION = 2
MAX_GROUNDING_CHUNKS = 10
MAX_GROUNDING_CHARS = 9000


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


def _generate_source_title(summary_text: str, source_type: str, user_id: str) -> str:
    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SOURCE_TITLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source type: {source_type}\n\n"
                    "Generate a concise title from this summary:\n\n"
                    f"{summary_text}"
                ),
            },
        ],
        response_format={"type": "json_schema", "json_schema": SOURCE_TITLE_JSON_SCHEMA},
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="generation",
        model_name=GENERATION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty source title content.")

    payload = json.loads(content)
    title = str(payload.get("title", "")).strip()
    if not title:
        raise ValueError("Model returned an empty source title.")
    return title


def _generate_source_deep_dive(
    summary_text: str,
    source_type: str,
    grounding_chunks: list[str],
    user_id: str,
) -> tuple[str, list[str]]:
    chunk_text = "\n\n".join(grounding_chunks) if grounding_chunks else "[No grounding chunks available]"

    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SOURCE_DEEP_DIVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source type: {source_type}\n\n"
                    f"Source summary:\n{summary_text}\n\n"
                    "Grounding chunks:\n"
                    f"{chunk_text}"
                ),
            },
        ],
        response_format={"type": "json_schema", "json_schema": SOURCE_DEEP_DIVE_JSON_SCHEMA},
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="generation",
        model_name=GENERATION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty deep-dive content.")

    payload = json.loads(content)
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
) -> list[ReflectionPoint]:
    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SOCRATIC_REFLECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source type: {source_type}\n\n"
                    f"Summary:\n{summary_text}\n\n"
                    f"Deep dive:\n{deep_dive_text}\n\n"
                    "Generate Socratic reflection points."
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SOCRATIC_REFLECTION_JSON_SCHEMA,
        },
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="generation",
        model_name=GENERATION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty reflection content.")

    payload = json.loads(content)
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


def _store_source_learning_section(section: SourceLearningSection, user_id: str) -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO source_learning_sections (
                    user_id,
                    source_id,
                    source_type,
                    generated_title,
                    summary_text,
                    reflection_points_json,
                    deep_dive_text,
                    key_terms_json,
                    schema_version,
                    model_name
                ) VALUES (
                    :user_id,
                    :source_id,
                    :source_type,
                    :generated_title,
                    :summary_text,
                    :reflection_points_json,
                    :deep_dive_text,
                    :key_terms_json,
                    :schema_version,
                    :model_name
                )
                ON CONFLICT(user_id, source_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    generated_title = excluded.generated_title,
                    summary_text = excluded.summary_text,
                    reflection_points_json = excluded.reflection_points_json,
                    deep_dive_text = excluded.deep_dive_text,
                    key_terms_json = excluded.key_terms_json,
                    schema_version = excluded.schema_version,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "source_id": section.source_id,
                "source_type": section.source_type,
                "generated_title": section.generated_title,
                "summary_text": section.summary_text,
                "reflection_points_json": json.dumps(
                    [point.model_dump() for point in section.reflection_points],
                    ensure_ascii=True,
                ),
                "deep_dive_text": section.deep_dive_text,
                "key_terms_json": json.dumps(section.key_terms, ensure_ascii=True),
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
                    generated_title,
                    summary_text,
                    reflection_points_json,
                    deep_dive_text,
                    key_terms_json,
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

    reflection_points_raw = json.loads(str(row["reflection_points_json"] or "[]"))
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

    key_terms_raw = json.loads(str(row["key_terms_json"] or "[]"))
    key_terms = [str(term).strip() for term in key_terms_raw if str(term).strip()]

    if not reflection_points or not key_terms:
        return None

    return SourceLearningSection(
        source_id=int(row["source_id"]),
        source_type=str(row["source_type"]),
        generated_title=str(row["generated_title"]),
        summary_text=str(row["summary_text"]),
        deep_dive_text=str(row["deep_dive_text"] or "").strip(),
        key_terms=key_terms,
        reflection_points=reflection_points,
        model_name=str(row["model_name"]),
        schema_version=int(row["schema_version"]),
    )


def _build_source_learning_section(source_id: int, user_id: str) -> SourceLearningSection:
    source_type, _, _ = get_source_metadata(source_id, user_id)

    if CACHE_PROCESSED_SOURCES:
        cached = _load_cached_source_learning_section(source_id, user_id)
        if cached is not None:
            return cached

    summary_text = load_existing_source_summary(source_id, user_id)
    if summary_text is None:
        process_result = process_source(source_id, user_id)
        summary_text = (process_result.source_summary or "").strip()

    if not summary_text:
        raise ValueError(f"Source {source_id} summary is unavailable.")

    grounding_chunks = _load_grounding_chunks(source_id, user_id)
    generated_title = _generate_source_title(summary_text, source_type, user_id)
    deep_dive_text, key_terms = _generate_source_deep_dive(
        summary_text,
        source_type,
        grounding_chunks,
        user_id,
    )
    reflection_points = _generate_reflection_points(
        summary_text,
        deep_dive_text,
        source_type,
        user_id,
    )

    section = SourceLearningSection(
        source_id=source_id,
        source_type=source_type,
        generated_title=generated_title,
        summary_text=summary_text,
        deep_dive_text=deep_dive_text,
        key_terms=key_terms,
        reflection_points=reflection_points,
        model_name=GENERATION_MODEL,
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    _store_source_learning_section(section, user_id)
    return section


def _generate_combined_insights(
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
    relationship_insights: list[str],
    user_id: str,
) -> CombinedInsightSection:
    source_catalog = [
        {
            "source_id": video_section.source_id,
            "source_type": "video",
            "title": video_section.generated_title,
        }
    ] + [
        {
            "source_id": section.source_id,
            "source_type": "document",
            "title": section.generated_title,
        }
        for section in document_sections
    ]

    source_payload = {
        "source_catalog": source_catalog,
        "video": video_section.model_dump(),
        "documents": [section.model_dump() for section in document_sections],
        "relationship_insights": relationship_insights,
    }

    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": COMBINED_INSIGHTS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate cross-source attributed insights from this payload:\n\n"
                    f"{json.dumps(source_payload, ensure_ascii=True)}"
                ),
            },
        ],
        response_format={"type": "json_schema", "json_schema": COMBINED_INSIGHTS_JSON_SCHEMA},
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="generation",
        model_name=GENERATION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty combined insights content.")

    payload = json.loads(content)
    known_sources = {
        video_section.source_id: "video",
        **{section.source_id: "document" for section in document_sections},
    }

    raw_parallels = payload.get("parallels", [])
    parallels: list[AttributedSentence] = []
    for raw_item in raw_parallels:
        source_id = int(raw_item.get("source_id", video_section.source_id))
        if source_id not in known_sources:
            source_id = video_section.source_id

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
            continue

        parallels.append(
            AttributedSentence(
                text=text_value,
                source_id=source_id,
                source_type=source_type,
                emphasis_terms=emphasis_terms,
            )
        )

    layman_bridge = str(payload.get("layman_bridge", "")).strip()
    synthesis_text = str(payload.get("synthesis_text", "")).strip()
    if len(parallels) < 3:
        raise ValueError("Model returned too few cross-source parallels.")
    if not layman_bridge:
        raise ValueError("Model returned an empty layman bridge.")
    if not synthesis_text:
        raise ValueError("Model returned an empty synthesis text.")

    return CombinedInsightSection(
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

    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": COMBINED_QUIZ_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate overlap-focused advanced quiz questions from this payload:\n\n"
                    f"{json.dumps(quiz_payload, ensure_ascii=True)}"
                ),
            },
        ],
        response_format={"type": "json_schema", "json_schema": COMBINED_QUIZ_JSON_SCHEMA},
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="generation",
        model_name=GENERATION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty quiz content.")

    payload = json.loads(content)
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
                    parallels_json,
                    layman_bridge,
                    synthesis_text,
                    quiz_json,
                    schema_version,
                    model_name
                ) VALUES (
                    :user_id,
                    :video_source_id,
                    :document_source_ids_json,
                    :parallels_json,
                    :layman_bridge,
                    :synthesis_text,
                    :quiz_json,
                    :schema_version,
                    :model_name
                )
                ON CONFLICT(user_id, video_source_id, document_source_ids_json) DO UPDATE SET
                    parallels_json = excluded.parallels_json,
                    layman_bridge = excluded.layman_bridge,
                    synthesis_text = excluded.synthesis_text,
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
                "parallels_json": json.dumps(
                    [entry.model_dump() for entry in insights.parallels],
                    ensure_ascii=True,
                ),
                "layman_bridge": insights.layman_bridge,
                "synthesis_text": insights.synthesis_text,
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
                    parallels_json,
                    layman_bridge,
                    synthesis_text,
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

    raw_parallels = json.loads(str(row["parallels_json"] or "[]"))
    parallels: list[AttributedSentence] = []
    for entry in raw_parallels:
        try:
            parallels.append(AttributedSentence.model_validate(entry))
        except Exception:
            continue

    if not parallels:
        return None

    layman_bridge = str(row["layman_bridge"] or "").strip()
    synthesis_text = str(row["synthesis_text"] or "").strip()
    if not layman_bridge or not synthesis_text:
        return None

    quiz_payload = json.loads(str(row["quiz_json"] or "{}"))
    try:
        quiz_section = CombinedQuizSection.model_validate(quiz_payload)
    except Exception:
        return None

    insights_section = CombinedInsightSection(
        parallels=parallels,
        layman_bridge=layman_bridge,
        synthesis_text=synthesis_text,
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

    normalized_video_id, normalized_document_ids = _normalize_ids(
        video_source_id,
        document_source_ids,
    )

    process_source(normalized_video_id, user_id)
    for document_source_id in normalized_document_ids:
        process_source(document_source_id, user_id)

    for document_source_id in normalized_document_ids:
        link_source_pair(normalized_video_id, document_source_id, user_id)

    video_section = _build_source_learning_section(normalized_video_id, user_id)
    document_sections = [
        _build_source_learning_section(source_id, user_id)
        for source_id in normalized_document_ids
    ]

    cached_combined = None
    if CACHE_PROCESSED_SOURCES:
        cached_combined = _load_cached_combined_learning_sections(
            user_id=user_id,
            video_source_id=normalized_video_id,
            document_source_ids=normalized_document_ids,
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
        )
        quiz_section = _generate_combined_quiz(
            video_section=video_section,
            document_sections=document_sections,
            insights_section=insights_section,
            relationship_insights=relationship_insights,
            user_id=user_id,
        )
        _store_combined_learning_sections(
            user_id=user_id,
            video_source_id=normalized_video_id,
            document_source_ids=normalized_document_ids,
            insights=insights_section,
            quiz=quiz_section,
        )

    return GenerateTailoredLearningResponse(
        status_message="Tailored Socratic learning generated successfully.",
        source_ids=[normalized_video_id, *normalized_document_ids],
        video=video_section,
        documents=document_sections,
        insights=insights_section,
        quiz=quiz_section,
    )
