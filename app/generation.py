from __future__ import annotations

import json
import time
import uuid
from typing import Any

from sqlalchemy import text

from app.cost_logging import log_api_usage, log_generation_stage_event
from app.json_reliability import parse_json_object, safe_json_loads
from app.session_logger import get_active_logger, session_log
from app.ledger import (
    KnowledgeUnit,
    append_units,
    ensure_ledger_table,
    load_ledger,
    next_unit_id,
    parse_units,
    render_units,
    store_ledger,
)
from app.models import (
    ApplicationScenario,
    CombinedInsightSection,
    CombinedQuizSection,
    GenerateTailoredLearningResponse,
    InsightIntersection,
    KeyTermExplanation,
    QuizQuestion,
    ReflectionPoint,
    SourceLearningSection,
)
import config
from config import (
    CACHE_PROCESSED_SOURCES,
    db_engine,
    get_aux_client,
    get_aux_model,
    get_llm_client,
)
from prompts import (
    AUDIT_JSON_SCHEMA,
    AUDIT_SYSTEM_PROMPT,
    CONSOLIDATION_JSON_SCHEMA,
    CONSOLIDATION_SYSTEM_PROMPT,
    LEDGER_EXTRACTION_JSON_SCHEMA,
    LEDGER_EXTRACTION_SYSTEM_PROMPT,
    SYNTHESIS_JSON_SCHEMA,
    SYNTHESIS_SYSTEM_PROMPT,
)
from prompts.sections import (
    CROSS_SOURCE_SECTIONS,
    PER_SOURCE_SECTIONS,
    SectionSpec,
)


GENERATION_SCHEMA_VERSION = 14
LEDGER_SCHEMA_VERSION = 1
GENERATION_MAX_STAGE_ATTEMPTS = 3


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
            f"Generation stage '{stage_name}' failed after {attempt_number} attempts "
            f"(run {run_id}): {reason}"
        )


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------


def _run_structured_generation_step(
    *,
    run_id: str,
    stage_name: str,
    user_id: str,
    user_prompt: str,
    system_prompt: str,
    response_schema: dict[str, Any],
    required_keys: list[str] | None = None,
    model_name: str | None = None,
    use_aux: bool = False,
) -> dict[str, Any]:
    last_error_detail = "Retry budget exhausted."
    client = get_aux_client() if use_aux else get_llm_client()
    if model_name is None:
        model_name = get_aux_model() if use_aux else config.generation_model()

    for attempt_number in range(1, GENERATION_MAX_STAGE_ATTEMPTS + 1):
        started = time.perf_counter()
        log_generation_stage_event(
            run_id=run_id, user_id=user_id, stage_name=stage_name,
            attempt_number=attempt_number, status="started",
        )
        try:
            result = client.chat_json(
                model=model_name,
                system=system_prompt,
                user=user_prompt,
                json_schema=response_schema,
                temperature=0.0,
            )
            log_api_usage(
                response=result.raw, user_id=user_id,
                call_stage="generation", model_name=model_name,
            )
            payload = parse_json_object(
                result.content, stage_name=f"{stage_name} attempt {attempt_number}",
            )
            if required_keys:
                missing = [k for k in required_keys if k not in payload]
                if missing:
                    raise ValueError(f"Missing required keys: {', '.join(missing)}")

            duration_ms = int((time.perf_counter() - started) * 1000)
            log_generation_stage_event(
                run_id=run_id, user_id=user_id, stage_name=stage_name,
                attempt_number=attempt_number, status="succeeded",
                duration_ms=duration_ms,
            )
            get_active_logger().record("stage_call", {
                "stage_name": stage_name, "attempt": attempt_number,
                "duration_ms": duration_ms, "model": model_name,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            })
            return payload
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            last_error_detail = str(exc)[:700]
            log_generation_stage_event(
                run_id=run_id, user_id=user_id, stage_name=stage_name,
                attempt_number=attempt_number, status="failed",
                duration_ms=duration_ms, error_message=last_error_detail,
            )

    raise GenerationStageError(
        run_id=run_id, stage_name=stage_name,
        attempt_number=GENERATION_MAX_STAGE_ATTEMPTS, reason=last_error_detail,
    )


# ---------------------------------------------------------------------------
# DB tables
# ---------------------------------------------------------------------------


def ensure_generation_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(text("""
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
        """))
        source_columns = {
            row[1]
            for row in connection.execute(
                text("PRAGMA table_info(source_learning_sections)")
            ).fetchall()
        }
        migrations: dict[str, str] = {
            "deep_dive_text": "TEXT DEFAULT ''",
            "key_terms_json": "TEXT DEFAULT '[]'",
            "schema_version": "INTEGER DEFAULT 1",
            "source_name": "TEXT DEFAULT ''",
            "under_surface_text": "TEXT DEFAULT ''",
            "key_term_explanations_json": "TEXT DEFAULT '[]'",
        }
        for col, col_type in migrations.items():
            if col not in source_columns:
                connection.execute(
                    text(f"ALTER TABLE source_learning_sections ADD COLUMN {col} {col_type}")
                )

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_source_learning_sections_user_source
            ON source_learning_sections (user_id, source_id)
        """))

        connection.execute(text("""
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
        """))
        combined_columns = {
            row[1]
            for row in connection.execute(
                text("PRAGMA table_info(combined_learning_sections)")
            ).fetchall()
        }
        combined_migrations: dict[str, str] = {
            "synthesis_text": "TEXT DEFAULT ''",
            "schema_version": "INTEGER DEFAULT 1",
            "intersections_json": "TEXT DEFAULT '[]'",
            "application_scenarios_json": "TEXT DEFAULT '[]'",
        }
        for col, col_type in combined_migrations.items():
            if col not in combined_columns:
                connection.execute(
                    text(f"ALTER TABLE combined_learning_sections ADD COLUMN {col} {col_type}")
                )

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_combined_learning_sections_user_video
            ON combined_learning_sections (user_id, video_source_id)
        """))

    ensure_ledger_table()


# ---------------------------------------------------------------------------
# Chunk loading helpers
# ---------------------------------------------------------------------------


def _load_source_chunks(source_id: int, user_id: str) -> list[str]:
    with db_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT chunk_text
                FROM source_text_chunks
                WHERE user_id = :user_id AND source_id = :source_id
                ORDER BY chunk_index ASC
            """),
            {"user_id": user_id, "source_id": source_id},
        ).mappings().all()
    return [str(row["chunk_text"]).strip() for row in rows if str(row["chunk_text"] or "").strip()]


def _get_source_metadata(source_id: int, user_id: str) -> tuple[str, str]:
    with db_engine.connect() as connection:
        row = connection.execute(
            text("""
                SELECT source_type, filename
                FROM sources
                WHERE id = :source_id AND user_id = :user_id
            """),
            {"source_id": source_id, "user_id": user_id},
        ).mappings().first()
    if row is None:
        raise ValueError(f"Source {source_id} was not found.")
    return str(row["source_type"]), str(row["filename"])


def _derive_title(filename: str, source_type: str) -> str:
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    name = name.replace("_", " ").replace("-", " ").strip()
    return name.title() if name else f"Untitled {source_type.title()}"


# ---------------------------------------------------------------------------
# Iterative accumulation core
# ---------------------------------------------------------------------------


def _group_chunks_into_windows(chunks: list[str], window_size: int = 0) -> list[str]:
    if not chunks:
        return []
    if window_size <= 0:
        # Larger windows = fewer ledger-extraction calls. The accumulation design
        # still holds (each window sees the full prior ledger), so this trades a
        # little per-call context for a meaningful call-count reduction.
        window_size = 3 if len(chunks) <= 10 else 4
    windows: list[str] = []
    for i in range(0, len(chunks), window_size):
        window_chunks = chunks[i : i + window_size]
        windows.append("\n\n".join(window_chunks))
    return windows


def _extract_source_ledger(
    windows: list[str],
    source_id: int,
    source_name: str,
    *,
    run_id: str,
    user_id: str,
) -> list[KnowledgeUnit]:
    """Iterate windows, emitting deduped typed KnowledgeUnits into a ledger."""
    ledger: list[KnowledgeUnit] = []
    for idx, window in enumerate(windows):
        prior = render_units(ledger) if ledger else "(empty - this is the first material)"
        user_prompt = (
            f"Source: {source_name}\n\n"
            f"=== Ledger built from earlier material (do NOT restate these) ===\n{prior}\n\n"
            f"=== New material (section {idx + 1} of {len(windows)}) ===\n{window}\n\n"
            "Extract only genuinely new knowledge units from the new material."
        )
        payload = _run_structured_generation_step(
            run_id=run_id,
            stage_name=f"ledger:{source_name}:{idx + 1}",
            user_id=user_id,
            system_prompt=LEDGER_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=LEDGER_EXTRACTION_JSON_SCHEMA,
            required_keys=["units"],
            use_aux=True,
        )
        new_units = parse_units(payload.get("units", []), source_id, next_unit_id(ledger))
        append_units(ledger, new_units)
    return ledger


def _consolidate_ledger(
    ledger: list[KnowledgeUnit],
    source_name: str,
    *,
    run_id: str,
    user_id: str,
    extra_violations: list[str] | None = None,
) -> dict[str, Any]:
    rendered = render_units(ledger)
    user_prompt = (
        f"Source: {source_name}\n\n"
        f"=== Knowledge ledger (id | type | source) ===\n{rendered}\n\n"
        "Organize these units into the required JSON structure, following each "
        "section contract. Reference the claims; do not restate them across fields."
    )
    if extra_violations:
        user_prompt += (
            "\n\nA prior draft was rejected for these contract violations. Fix them:\n- "
            + "\n- ".join(extra_violations)
        )
    return _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"consolidation:{source_name}",
        user_id=user_id,
        system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=CONSOLIDATION_JSON_SCHEMA,
        required_keys=["summary", "deep_dive", "key_terms", "under_surface", "reflection_points"],
    )


# ---------------------------------------------------------------------------
# Generic section audit (cheap aux passes, one re-gen on violation)
# ---------------------------------------------------------------------------


def _render_field_for_audit(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return str(value)
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            parts.append(str(item))
        elif "term" in item:
            parts.append(f"{item.get('term')}: {item.get('layman', '')} {item.get('technical', '')}")
        elif "question" in item:
            parts.append(
                f"Q: {item.get('question')} | {item.get('explanation', '')} "
                f"[{item.get('depth_level', '')}]"
            )
        else:
            parts.append(json.dumps(item, ensure_ascii=True))
    return "\n".join(parts)


def _audit_arc(
    specs: tuple[SectionSpec, ...],
    structured: dict[str, Any],
    *,
    run_id: str,
    user_id: str,
) -> list[str]:
    """Audit the whole arc in ONE aux call.

    Sections are rendered in arc order with their contracts so the auditor can
    check each against its own rules and against earlier sections for
    restatement. Sections with no audit rules or empty content are skipped.
    Replaces the former per-section loop (one call per section)."""
    blocks: list[str] = []
    for spec in specs:
        if not spec.audit:
            continue
        field = spec.field.split("+")[0]
        content = _render_field_for_audit(structured.get(field, ""))
        if not content.strip():
            continue
        blocks.append(
            f"### SECTION: {spec.title}\n"
            f"ROLE: {spec.role}\n"
            f"NON-OVERLAP RULE: {spec.forbids}\n"
            f"LENGTH BUDGET: {spec.budget}\n"
            f"CONTENT:\n{content}"
        )

    if not blocks:
        return []

    user_prompt = (
        "Audit the following sections against their contracts. They are listed "
        "in arc order, so each section's 'prior' material is everything above "
        "it. Report only clear violations, each prefixed with its section title.\n\n"
        + "\n\n".join(blocks)
    )
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name="audit:arc",
        user_id=user_id,
        system_prompt=AUDIT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=AUDIT_JSON_SCHEMA,
        required_keys=["ok", "violations"],
        use_aux=True,
    )
    if payload.get("ok"):
        return []
    return [str(v).strip() for v in payload.get("violations", []) if str(v).strip()]


# ---------------------------------------------------------------------------
# Per-source section builder
# ---------------------------------------------------------------------------


def _build_source_learning_section(
    source_id: int,
    user_id: str,
    run_id: str,
) -> SourceLearningSection:
    if CACHE_PROCESSED_SOURCES:
        cached = _load_cached_source_learning_section(source_id, user_id)
        if cached is not None:
            log_generation_stage_event(
                run_id=run_id, user_id=user_id,
                stage_name=f"source_section:{source_id}", attempt_number=1,
                status="cache_hit",
            )
            return cached

    source_type, filename = _get_source_metadata(source_id, user_id)
    source_name = _derive_title(filename, source_type)
    chunks = _load_source_chunks(source_id, user_id)
    if not chunks:
        raise ValueError(f"Source {source_id} has no text chunks.")

    ledger = load_ledger(
        user_id=user_id, source_id=source_id, schema_version=LEDGER_SCHEMA_VERSION,
    )
    if ledger is None:
        windows = _group_chunks_into_windows(chunks)
        ledger = _extract_source_ledger(
            windows, source_id, source_name, run_id=run_id, user_id=user_id,
        )
        store_ledger(
            user_id=user_id, source_id=source_id, units=ledger,
            schema_version=LEDGER_SCHEMA_VERSION,
        )

    structured = _consolidate_ledger(
        ledger, source_name, run_id=run_id, user_id=user_id,
    )
    violations = _audit_arc(
        PER_SOURCE_SECTIONS, structured, run_id=run_id, user_id=user_id,
    )
    if violations:
        structured = _consolidate_ledger(
            ledger, source_name, run_id=run_id, user_id=user_id,
            extra_violations=violations,
        )

    key_terms_raw = structured.get("key_terms", [])
    key_terms: list[str] = []
    key_term_explanations: list[KeyTermExplanation] = []
    for entry in key_terms_raw:
        if isinstance(entry, dict):
            term = str(entry.get("term", "")).strip()
            if term:
                key_terms.append(term)
                key_term_explanations.append(KeyTermExplanation(
                    term=term,
                    layman=str(entry.get("layman", "")).strip(),
                    technical=str(entry.get("technical", "")).strip(),
                ))
        elif isinstance(entry, str) and entry.strip():
            key_terms.append(entry.strip())

    reflection_points: list[ReflectionPoint] = []
    for rp in structured.get("reflection_points", []):
        try:
            reflection_points.append(ReflectionPoint.model_validate(rp))
        except Exception:
            continue

    section = SourceLearningSection(
        source_id=source_id,
        source_type=source_type,
        source_name=source_name,
        generated_title=source_name,
        summary_text=str(structured.get("summary", "")).strip(),
        deep_dive_text=str(structured.get("deep_dive", "")).strip(),
        key_terms=key_terms,
        under_surface_explainer=str(structured.get("under_surface", "")).strip(),
        key_term_explanations=key_term_explanations,
        reflection_points=reflection_points,
        model_name=config.generation_model(),
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    _store_source_learning_section(section, user_id)
    return section


# ---------------------------------------------------------------------------
# Cross-source synthesis
# ---------------------------------------------------------------------------


def _merge_source_ledgers(
    source_sections: list[SourceLearningSection],
    *,
    user_id: str,
) -> list[KnowledgeUnit]:
    """Concatenate per-source ledgers with globally unique ids (no cross-source dedup,
    so genuine convergences between sources remain visible to synthesis)."""
    merged: list[KnowledgeUnit] = []
    counter = 1
    for section in source_sections:
        units = load_ledger(
            user_id=user_id, source_id=section.source_id,
            schema_version=LEDGER_SCHEMA_VERSION,
        ) or []
        for unit in units:
            merged.append(unit.model_copy(update={"id": f"u{counter}"}))
            counter += 1
    return merged


def _ground_intersections(
    raw_intersections: list[Any],
) -> list[InsightIntersection]:
    """Keep only intersections grounded in >=2 distinct source ids (anti-fabrication)."""
    grounded: list[InsightIntersection] = []
    for entry in raw_intersections:
        try:
            intersection = InsightIntersection.model_validate(entry)
        except Exception:
            continue
        source_ids = {s.source_id for s in intersection.attributed_sentences}
        if len(source_ids) >= 2:
            grounded.append(intersection)
    return grounded


def _generate_synthesis(
    source_sections: list[SourceLearningSection],
    *,
    run_id: str,
    user_id: str,
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    merged = _merge_source_ledgers(source_sections, user_id=user_id)
    rendered = render_units(merged)

    def _run(extra_violations: list[str] | None) -> dict[str, Any]:
        user_prompt = (
            "Synthesize ACROSS the sources using only the following knowledge ledger. "
            "Each unit carries its source id. Every intersection must cite claims from at "
            "least two different source ids via attributed_sentences. Do not introduce "
            "topics absent from the ledger.\n\n"
            f"=== Merged knowledge ledger (id | type | source) ===\n{rendered}"
        )
        if extra_violations:
            user_prompt += (
                "\n\nA prior draft was rejected for these contract violations. Fix them:\n- "
                + "\n- ".join(extra_violations)
            )
        return _run_structured_generation_step(
            run_id=run_id,
            stage_name="synthesis",
            user_id=user_id,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=SYNTHESIS_JSON_SCHEMA,
            required_keys=[
                "synthesis_text", "intersections", "questions", "application_scenarios",
            ],
        )

    payload = _run(None)
    violations = _audit_arc(
        CROSS_SOURCE_SECTIONS, payload, run_id=run_id, user_id=user_id,
    )
    if violations:
        payload = _run(violations)

    intersections = _ground_intersections(payload.get("intersections", []))

    questions: list[QuizQuestion] = []
    for entry in payload.get("questions", []):
        try:
            questions.append(QuizQuestion.model_validate(entry))
        except Exception:
            continue

    application_scenarios: list[ApplicationScenario] = []
    for entry in payload.get("application_scenarios", []):
        try:
            application_scenarios.append(ApplicationScenario.model_validate(entry))
        except Exception:
            continue

    insights = CombinedInsightSection(
        synthesis_text=str(payload.get("synthesis_text", "")).strip(),
        intersections=intersections,
        application_scenarios=application_scenarios,
        model_name=config.generation_model(),
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    quiz = CombinedQuizSection(
        questions=questions,
        model_name=config.generation_model(),
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    return insights, quiz


# ---------------------------------------------------------------------------
# Cache / storage
# ---------------------------------------------------------------------------


def _serialize_ids(source_ids: list[int]) -> str:
    return json.dumps(sorted(source_ids), separators=(",", ":"))


def _store_source_learning_section(section: SourceLearningSection, user_id: str) -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO source_learning_sections (
                    user_id, source_id, source_type, source_name,
                    generated_title, summary_text, reflection_points_json,
                    deep_dive_text, key_terms_json, under_surface_text,
                    key_term_explanations_json, schema_version, model_name
                ) VALUES (
                    :user_id, :source_id, :source_type, :source_name,
                    :generated_title, :summary_text, :reflection_points_json,
                    :deep_dive_text, :key_terms_json, :under_surface_text,
                    :key_term_explanations_json, :schema_version, :model_name
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
                    key_term_explanations_json = excluded.key_term_explanations_json,
                    schema_version = excluded.schema_version,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP
            """),
            {
                "user_id": user_id,
                "source_id": section.source_id,
                "source_type": section.source_type,
                "source_name": section.source_name,
                "generated_title": section.generated_title,
                "summary_text": section.summary_text,
                "reflection_points_json": json.dumps(
                    [p.model_dump() for p in section.reflection_points], ensure_ascii=True,
                ),
                "deep_dive_text": section.deep_dive_text,
                "key_terms_json": json.dumps(section.key_terms, ensure_ascii=True),
                "under_surface_text": section.under_surface_explainer,
                "key_term_explanations_json": json.dumps(
                    [e.model_dump() for e in section.key_term_explanations], ensure_ascii=True,
                ),
                "schema_version": section.schema_version,
                "model_name": section.model_name,
            },
        )


def _load_cached_source_learning_section(
    source_id: int, user_id: str,
) -> SourceLearningSection | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text("""
                SELECT source_id, source_type, source_name, generated_title,
                       summary_text, reflection_points_json, deep_dive_text,
                       key_terms_json, under_surface_text,
                       key_term_explanations_json, model_name, schema_version
                FROM source_learning_sections
                WHERE user_id = :user_id AND source_id = :source_id
                LIMIT 1
            """),
            {"user_id": user_id, "source_id": source_id},
        ).mappings().first()

    if row is None:
        return None
    if int(row.get("schema_version") or 1) != GENERATION_SCHEMA_VERSION:
        return None

    reflection_points_raw = safe_json_loads(row.get("reflection_points_json"), default=[])
    reflection_points: list[ReflectionPoint] = []
    for value in reflection_points_raw:
        if isinstance(value, str) and value.strip():
            reflection_points.append(ReflectionPoint(
                question=value.strip(),
                explanation="Reflect on what this means in context.",
                depth_level="foundational",
            ))
        elif isinstance(value, dict):
            try:
                reflection_points.append(ReflectionPoint.model_validate(value))
            except Exception:
                continue

    key_terms_raw = safe_json_loads(row.get("key_terms_json"), default=[])
    key_terms = [str(t).strip() for t in key_terms_raw if str(t).strip()]

    key_term_explanations_raw = safe_json_loads(row.get("key_term_explanations_json"), default=[])
    key_term_explanations: list[KeyTermExplanation] = []
    for entry in key_term_explanations_raw:
        try:
            if isinstance(entry, dict) and "explanation" in entry and "layman" not in entry:
                entry["layman"] = entry.pop("explanation")
                entry["technical"] = ""
            key_term_explanations.append(KeyTermExplanation.model_validate(entry))
        except Exception:
            continue

    summary = str(row.get("summary_text") or "").strip()
    if not summary or not key_terms:
        return None

    return SourceLearningSection(
        source_id=int(row["source_id"]),
        source_type=str(row["source_type"]),
        source_name=str(row["source_name"] or row["generated_title"] or "Source").strip(),
        generated_title=str(row["generated_title"]),
        summary_text=summary,
        deep_dive_text=str(row["deep_dive_text"] or "").strip(),
        key_terms=key_terms,
        under_surface_explainer=str(row["under_surface_text"] or "").strip(),
        key_term_explanations=key_term_explanations,
        reflection_points=reflection_points,
        model_name=str(row["model_name"]),
        schema_version=int(row["schema_version"]),
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
            text("""
                INSERT INTO combined_learning_sections (
                    user_id, video_source_id, document_source_ids_json,
                    intersections_json, parallels_json, layman_bridge,
                    synthesis_text, application_scenarios_json,
                    quiz_json, schema_version, model_name
                ) VALUES (
                    :user_id, :video_source_id, :document_source_ids_json,
                    :intersections_json, :parallels_json, :layman_bridge,
                    :synthesis_text, :application_scenarios_json,
                    :quiz_json, :schema_version, :model_name
                )
                ON CONFLICT(user_id, video_source_id, document_source_ids_json) DO UPDATE SET
                    intersections_json = excluded.intersections_json,
                    parallels_json = excluded.parallels_json,
                    layman_bridge = excluded.layman_bridge,
                    synthesis_text = excluded.synthesis_text,
                    application_scenarios_json = excluded.application_scenarios_json,
                    quiz_json = excluded.quiz_json,
                    schema_version = excluded.schema_version,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP
            """),
            {
                "user_id": user_id,
                "video_source_id": video_source_id,
                "document_source_ids_json": _serialize_ids(document_source_ids),
                "intersections_json": json.dumps(
                    [e.model_dump() for e in insights.intersections], ensure_ascii=True,
                ),
                "parallels_json": "[]",
                "layman_bridge": "",
                "synthesis_text": insights.synthesis_text,
                "application_scenarios_json": json.dumps(
                    [e.model_dump() for e in insights.application_scenarios], ensure_ascii=True,
                ),
                "quiz_json": json.dumps(quiz.model_dump(), ensure_ascii=True),
                "schema_version": GENERATION_SCHEMA_VERSION,
                "model_name": config.generation_model(),
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
            text("""
                SELECT intersections_json, synthesis_text,
                       application_scenarios_json, quiz_json,
                       model_name, schema_version
                FROM combined_learning_sections
                WHERE user_id = :user_id
                  AND video_source_id = :video_source_id
                  AND document_source_ids_json = :document_source_ids_json
                LIMIT 1
            """),
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

    intersections: list[InsightIntersection] = []
    for entry in safe_json_loads(row.get("intersections_json"), default=[]):
        try:
            intersections.append(InsightIntersection.model_validate(entry))
        except Exception:
            continue

    if not intersections:
        return None

    application_scenarios: list[ApplicationScenario] = []
    for entry in safe_json_loads(row.get("application_scenarios_json"), default=[]):
        try:
            application_scenarios.append(ApplicationScenario.model_validate(entry))
        except Exception:
            continue

    quiz_payload = safe_json_loads(row.get("quiz_json"), default={})
    try:
        quiz_section = CombinedQuizSection.model_validate(quiz_payload)
    except Exception:
        return None

    insights_section = CombinedInsightSection(
        synthesis_text=str(row["synthesis_text"] or "").strip(),
        intersections=intersections,
        application_scenarios=application_scenarios,
        model_name=str(row["model_name"]),
        schema_version=int(row["schema_version"]),
    )
    return insights_section, quiz_section


# ---------------------------------------------------------------------------
# ID normalization
# ---------------------------------------------------------------------------


def _normalize_ids(
    video_source_id: int | None,
    document_source_ids: list[int],
) -> tuple[int | None, list[int]]:
    normalized_video = video_source_id if (video_source_id or 0) > 0 else None
    normalized_docs = sorted({sid for sid in document_source_ids if sid > 0})
    if normalized_video is None and not normalized_docs:
        raise ValueError("At least one valid source_id is required.")
    return normalized_video, normalized_docs


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def generate_tailored_learning(
    video_source_id: int | None,
    document_source_ids: list[int],
    user_id: str,
) -> GenerateTailoredLearningResponse:
    ensure_generation_tables()
    run_id = f"gen-{uuid.uuid4().hex[:12]}"

    with session_log(run_id, user_id=user_id):
        normalized_video_id, normalized_document_ids = _normalize_ids(
            video_source_id, document_source_ids,
        )
        source_ids: list[int] = []
        if normalized_video_id is not None:
            source_ids.append(normalized_video_id)
        source_ids.extend(normalized_document_ids)

        log_generation_stage_event(
            run_id=run_id, user_id=user_id,
            stage_name="pipeline_start", attempt_number=1, status="started",
            details=(
                f"video={normalized_video_id or 'none'}; "
                f"docs={','.join(str(v) for v in normalized_document_ids)}"
            ),
        )

        video_section = (
            _build_source_learning_section(normalized_video_id, user_id, run_id)
            if normalized_video_id is not None
            else None
        )
        document_sections = [
            _build_source_learning_section(sid, user_id, run_id)
            for sid in normalized_document_ids
        ]

        all_sections = ([video_section] if video_section else []) + document_sections
        if not all_sections:
            raise ValueError("No source learning sections were generated.")

        total_sources = len(all_sections)
        if total_sources >= 2:
            cached_combined = None
            if CACHE_PROCESSED_SOURCES and normalized_video_id is not None:
                cached_combined = _load_cached_combined_learning_sections(
                    user_id=user_id,
                    video_source_id=normalized_video_id,
                    document_source_ids=normalized_document_ids,
                )

            if cached_combined is not None:
                insights_section, quiz_section = cached_combined
            else:
                insights_section, quiz_section = _generate_synthesis(
                    all_sections, run_id=run_id, user_id=user_id,
                )
                if normalized_video_id is not None:
                    _store_combined_learning_sections(
                        user_id=user_id,
                        video_source_id=normalized_video_id,
                        document_source_ids=normalized_document_ids,
                        insights=insights_section,
                        quiz=quiz_section,
                    )
        else:
            insights_section = CombinedInsightSection(
                synthesis_text="",
                intersections=[],
                application_scenarios=[],
                model_name=config.generation_model(),
            )
            quiz_section = CombinedQuizSection(
                questions=[],
                model_name=config.generation_model(),
            )

        return GenerateTailoredLearningResponse(
            status_message="Tailored Socratic learning generated successfully.",
            source_ids=source_ids,
            video=video_section,
            documents=document_sections,
            insights=insights_section,
            quiz=quiz_section,
        )
