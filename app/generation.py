from __future__ import annotations

import json
import time
import uuid
from typing import Any

from sqlalchemy import text

from app.cost_logging import log_api_usage, log_generation_stage_event
from app.json_reliability import parse_json_object, safe_json_loads
from app.mini_rag import MiniRag
from app.session_logger import get_active_logger, session_log
from app.models import (
    ApplicationScenario,
    CombinedInsightSection,
    CombinedQuizSection,
    GenerateTailoredLearningResponse,
    InsightIntersection,
    KeyTermExplanation,
    LearningSection,
    QuizQuestion,
    ReflectionPoint,
    SourceLearningSection,
)
from config import (
    CACHE_PROCESSED_SOURCES,
    GENERATION_MODEL,
    RETRIEVAL_MODEL,
    db_engine,
    get_llm_client,
)
from prompts import (
    CRITIC_JSON_SCHEMA,
    CRITIC_SYSTEM_PROMPT,
    KEY_CONCEPTS_JSON_SCHEMA,
    KEY_CONCEPTS_SYSTEM_PROMPT,
    OUTLINE_JSON_SCHEMA,
    OUTLINE_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    REFLECTION_JSON_SCHEMA,
    REFLECTION_SYSTEM_PROMPT,
    SECTION_SYSTEM_PROMPT,
    SYNTHESIS_JSON_SCHEMA,
    SYNTHESIS_SYSTEM_PROMPT,
)
from prompts.banned_phrases import BANNED_OPENERS


GENERATION_SCHEMA_VERSION = 13
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
    model_name: str = GENERATION_MODEL,
) -> dict[str, Any]:
    last_error_detail = "Retry budget exhausted."
    client = get_llm_client()

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


def _run_plain_generation_step(
    *,
    run_id: str,
    stage_name: str,
    user_id: str,
    user_prompt: str,
    system_prompt: str,
    model_name: str = GENERATION_MODEL,
) -> str:
    client = get_llm_client()
    started = time.perf_counter()
    log_generation_stage_event(
        run_id=run_id, user_id=user_id, stage_name=stage_name,
        attempt_number=1, status="started",
    )
    result = client.chat_json(
        model=model_name, system=system_prompt, user=user_prompt,
        temperature=0.0,
    )
    log_api_usage(
        response=result.raw, user_id=user_id,
        call_stage="generation", model_name=model_name,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    log_generation_stage_event(
        run_id=run_id, user_id=user_id, stage_name=stage_name,
        attempt_number=1, status="succeeded", duration_ms=duration_ms,
    )
    get_active_logger().record("stage_call", {
        "stage_name": stage_name, "duration_ms": duration_ms,
        "model": model_name,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
    })
    return result.content.strip()


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
            # v2.1: content-driven sections list. JSON-encoded
            # list of {title, body}. Old fields above are no longer written
            # but are kept so cached v2.0 rows can still be read.
            "sections_json": "TEXT DEFAULT '[]'",
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
# Iterative accumulation core (kept for backwards-compat smoke tests and
# in case any caller still references _group_chunks_into_windows).
# v2.1 generation uses outline-first below; this helper is no longer
# called by the new pipeline but is exported for tests.
# ---------------------------------------------------------------------------


def _group_chunks_into_windows(chunks: list[str], window_size: int = 0) -> list[str]:
    if not chunks:
        return []
    if window_size <= 0:
        window_size = 2 if len(chunks) <= 6 else 3
    windows: list[str] = []
    for i in range(0, len(chunks), window_size):
        window_chunks = chunks[i : i + window_size]
        windows.append("\n\n".join(window_chunks))
    return windows


# ---------------------------------------------------------------------------
# v2.1 outline-first generation
# ---------------------------------------------------------------------------


_GENERIC_TITLES_LOWER = {
    "summary", "overview", "introduction", "conclusion", "deep dive",
    "boundary conditions", "boundary conditions & failure modes",
    "hidden assumptions", "under the surface", "reflection",
    "reflection points",
}


def _generate_outline(
    source_name: str,
    chunks: list[str],
    *,
    run_id: str,
    user_id: str,
) -> list[dict[str, Any]]:
    # Cap context for the outline call. The outline only needs enough material
    # to pick titles; full chunks go into individual section drafts.
    sample = "\n\n---\n\n".join(chunks[:12])
    user_prompt = (
        f"Source: {source_name}\n\n"
        f"=== Source material ===\n{sample}\n\n"
        "Design the table of contents. 4-6 content-driven section titles."
    )
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"outline:{source_name}",
        user_id=user_id,
        system_prompt=OUTLINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=OUTLINE_JSON_SCHEMA,
        required_keys=["sections"],
    )
    items = payload.get("sections", []) or []
    # Defensive: drop any entry whose title is a generic scaffold label.
    cleaned: list[dict[str, Any]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        if not title or title.lower() in _GENERIC_TITLES_LOWER:
            continue
        scope = str(entry.get("scope", "")).strip()
        try:
            expected = int(entry.get("expected_paragraphs") or 3)
        except (TypeError, ValueError):
            expected = 3
        cleaned.append({"title": title, "scope": scope, "expected_paragraphs": expected})
    if not cleaned:
        # Last-resort fallback so the pipeline doesn't dead-end. One section
        # over the full source — better than crashing.
        cleaned = [{
            "title": f"How {source_name} works",
            "scope": "Cover the core mechanism and its key implications.",
            "expected_paragraphs": 4,
        }]
    return cleaned


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _render_prior_anchors(
    prior_titles: list[str],
    similar_hits: list[Any],
) -> str:
    if not prior_titles and not similar_hits:
        return "[No prior sections]"
    lines: list[str] = []
    if prior_titles:
        lines.append("Already-covered section titles (do not re-cover their topics):")
        for t in prior_titles:
            lines.append(f"  - {t}")
    if similar_hits:
        lines.append("")
        lines.append("Closest prior paragraphs (do not restate these points):")
        for h in similar_hits:
            snippet = h.text.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:217].rstrip() + "..."
            lines.append(f'  - (from "{h.title}") {snippet}')
    return "\n".join(lines)


def _draft_section(
    *,
    section_idx: int,
    title: str,
    scope: str,
    expected_paragraphs: int,
    prior_titles: list[str],
    similar_hits: list[Any],
    source_name: str,
    source_chunks: list[str],
    run_id: str,
    user_id: str,
) -> str:
    material = "\n\n---\n\n".join(source_chunks)
    if len(material) > 14000:
        material = material[:14000].rstrip() + "..."

    user_prompt = (
        f"Source: {source_name}\n\n"
        f"=== Section assignment ===\n"
        f"Title: {title}\n"
        f"Scope: {scope or '(no explicit scope provided)'}\n"
        f"Target paragraphs: {expected_paragraphs}\n\n"
        f"=== Prior section anchors ===\n{_render_prior_anchors(prior_titles, similar_hits)}\n\n"
        f"=== Source material ===\n{material}\n\n"
        "Write the body of this section now."
    )
    return _run_plain_generation_step(
        run_id=run_id,
        stage_name=f"section_draft:{section_idx}:{title}",
        user_id=user_id,
        system_prompt=SECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


def _critique_section(
    *,
    section_idx: int,
    title: str,
    scope: str,
    body: str,
    prior_titles: list[str],
    similar_hits: list[Any],
    run_id: str,
    user_id: str,
) -> dict[str, Any]:
    user_prompt = (
        f"=== Section ===\n"
        f"Title: {title}\n"
        f"Scope: {scope or '(no explicit scope)'}\n\n"
        f"=== Draft body ===\n{body}\n\n"
        f"=== Prior anchors ===\n{_render_prior_anchors(prior_titles, similar_hits)}\n\n"
        "Produce the critique JSON."
    )
    return _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"section_critic:{section_idx}:{title}",
        user_id=user_id,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=CRITIC_JSON_SCHEMA,
        required_keys=["needs_refinement", "instructions"],
    )


def _refine_section(
    *,
    section_idx: int,
    title: str,
    body: str,
    instructions: str,
    run_id: str,
    user_id: str,
) -> str:
    user_prompt = (
        f"=== Section ===\nTitle: {title}\n\n"
        f"=== Draft body ===\n{body}\n\n"
        f"=== Critique instructions ===\n{instructions}\n\n"
        "Rewrite the section body applying the critique. Plain prose only."
    )
    return _run_plain_generation_step(
        run_id=run_id,
        stage_name=f"section_refine:{section_idx}:{title}",
        user_id=user_id,
        system_prompt=REFINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


def _has_banned_opener(text: str) -> bool:
    """Lightweight local check that survives the critic if it misses one."""
    lowered = text.lstrip().lower()
    for phrase in BANNED_OPENERS:
        if lowered.startswith(phrase.lower()):
            return True
        # Catch mid-paragraph openers too — check every paragraph start.
        for para in _split_paragraphs(text):
            if para.lower().startswith(phrase.lower()):
                return True
    return False


def _generate_key_concepts(
    *,
    sections: list[LearningSection],
    source_name: str,
    run_id: str,
    user_id: str,
) -> list[KeyTermExplanation]:
    body = "\n\n".join(f"## {s.title}\n{s.body}" for s in sections)
    if len(body) > 14000:
        body = body[:14000].rstrip() + "..."
    user_prompt = (
        f"Source: {source_name}\n\n"
        f"=== Completed sections ===\n{body}\n\n"
        "Extract 6-10 key terms with single layman-tone explanations."
    )
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"key_concepts:{source_name}",
        user_id=user_id,
        system_prompt=KEY_CONCEPTS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=KEY_CONCEPTS_JSON_SCHEMA,
        required_keys=["key_concepts"],
    )
    out: list[KeyTermExplanation] = []
    for entry in payload.get("key_concepts", []):
        if not isinstance(entry, dict):
            continue
        term = str(entry.get("term", "")).strip()
        explanation = str(entry.get("explanation", "")).strip()
        if term and explanation:
            out.append(KeyTermExplanation(term=term, explanation=explanation))
    return out


def _generate_reflection_points(
    *,
    sections: list[LearningSection],
    source_name: str,
    run_id: str,
    user_id: str,
) -> list[ReflectionPoint]:
    body = "\n\n".join(f"## {s.title}\n{s.body}" for s in sections)
    if len(body) > 14000:
        body = body[:14000].rstrip() + "..."
    user_prompt = (
        f"Source: {source_name}\n\n"
        f"=== Completed sections ===\n{body}\n\n"
        "Write 4-6 Socratic reflection points across depth levels."
    )
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"reflections:{source_name}",
        user_id=user_id,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=REFLECTION_JSON_SCHEMA,
        required_keys=["reflection_points"],
    )
    out: list[ReflectionPoint] = []
    for entry in payload.get("reflection_points", []):
        try:
            out.append(ReflectionPoint.model_validate(entry))
        except Exception:
            continue
    return out


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

    # 1. Outline
    outline = _generate_outline(source_name, chunks, run_id=run_id, user_id=user_id)

    # 2. Per-section draft + critic + (refine)
    client = get_llm_client()
    mini_rag = MiniRag(client=client, model=RETRIEVAL_MODEL)
    learning_sections: list[LearningSection] = []
    prior_titles: list[str] = []

    for idx, item in enumerate(outline):
        title = item["title"]
        scope = item["scope"]
        expected = item["expected_paragraphs"]

        # Query mini-RAG using the title+scope so we surface prior paragraphs
        # most likely to overlap with this section's intended content.
        similar_hits = mini_rag.find_similar(f"{title}. {scope}", top_k=3)

        draft = _draft_section(
            section_idx=idx,
            title=title,
            scope=scope,
            expected_paragraphs=expected,
            prior_titles=prior_titles,
            similar_hits=similar_hits,
            source_name=source_name,
            source_chunks=chunks,
            run_id=run_id,
            user_id=user_id,
        )

        critique = _critique_section(
            section_idx=idx,
            title=title,
            scope=scope,
            body=draft,
            prior_titles=prior_titles,
            similar_hits=similar_hits,
            run_id=run_id,
            user_id=user_id,
        )

        needs_refine = bool(critique.get("needs_refinement")) or _has_banned_opener(draft)
        final_body = draft
        if needs_refine:
            instructions = str(critique.get("instructions") or "").strip()
            if not instructions:
                instructions = (
                    "Replace any opener that starts with a stock LLM phrase. "
                    "Open with concrete claims grounded in the source."
                )
            final_body = _refine_section(
                section_idx=idx,
                title=title,
                body=draft,
                instructions=instructions,
                run_id=run_id,
                user_id=user_id,
            )

        learning_sections.append(LearningSection(title=title, body=final_body.strip()))
        prior_titles.append(title)
        try:
            mini_rag.add_paragraphs(
                section_idx=idx,
                title=title,
                paragraphs=_split_paragraphs(final_body),
            )
        except Exception:
            # Embedding outage shouldn't abort the run; we just lose
            # anti-redundancy guarantees for subsequent sections.
            pass

    # 3. Key concepts (single layman explanation per term)
    key_term_explanations = _generate_key_concepts(
        sections=learning_sections,
        source_name=source_name,
        run_id=run_id,
        user_id=user_id,
    )
    key_terms = [k.term for k in key_term_explanations]

    # 4. Reflection points
    reflection_points = _generate_reflection_points(
        sections=learning_sections,
        source_name=source_name,
        run_id=run_id,
        user_id=user_id,
    )

    section = SourceLearningSection(
        source_id=source_id,
        source_type=source_type,
        source_name=source_name,
        generated_title=source_name,
        sections=learning_sections,
        summary_text="",  # legacy fields kept empty under v13
        deep_dive_text="",
        under_surface_explainer="",
        key_terms=key_terms,
        key_term_explanations=key_term_explanations,
        reflection_points=reflection_points,
        model_name=GENERATION_MODEL,
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    _store_source_learning_section(section, user_id)
    return section


# ---------------------------------------------------------------------------
# Cross-source synthesis
# ---------------------------------------------------------------------------


def _generate_synthesis(
    source_sections: list[SourceLearningSection],
    *,
    run_id: str,
    user_id: str,
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    source_summaries: list[str] = []
    for section in source_sections:
        block = (
            f"--- Source: {section.source_name} ({section.source_type}) ---\n"
            f"Summary:\n{section.summary_text}\n\n"
            f"Deep Dive:\n{section.deep_dive_text}\n\n"
            f"Under the Surface:\n{section.under_surface_explainer}"
        )
        source_summaries.append(block)

    user_prompt = (
        "Synthesize the following source analyses into a cross-source learning artifact.\n\n"
        + "\n\n".join(source_summaries)
    )

    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name="synthesis",
        user_id=user_id,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=SYNTHESIS_JSON_SCHEMA,
        required_keys=["synthesis_text", "intersections", "questions", "application_scenarios"],
    )

    intersections: list[InsightIntersection] = []
    for entry in payload.get("intersections", []):
        try:
            intersections.append(InsightIntersection.model_validate(entry))
        except Exception:
            continue

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
        model_name=GENERATION_MODEL,
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    quiz = CombinedQuizSection(
        questions=questions,
        model_name=GENERATION_MODEL,
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
                    key_term_explanations_json, sections_json,
                    schema_version, model_name
                ) VALUES (
                    :user_id, :source_id, :source_type, :source_name,
                    :generated_title, :summary_text, :reflection_points_json,
                    :deep_dive_text, :key_terms_json, :under_surface_text,
                    :key_term_explanations_json, :sections_json,
                    :schema_version, :model_name
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
                    sections_json = excluded.sections_json,
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
                "sections_json": json.dumps(
                    [s.model_dump() for s in section.sections], ensure_ascii=True,
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
                       key_term_explanations_json, sections_json,
                       model_name, schema_version
                FROM source_learning_sections
                WHERE user_id = :user_id AND source_id = :source_id
                LIMIT 1
            """),
            {"user_id": user_id, "source_id": source_id},
        ).mappings().first()

    if row is None:
        return None
    # v2.1 only honors v13 cached rows. Older rows regenerate. Acceptable —
    # the schema bump is the explicit invalidation signal.
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
        if not isinstance(entry, dict):
            continue
        term = str(entry.get("term", "")).strip()
        if not term:
            continue
        # v2.0 had `layman` + `technical` — coalesce into the new single
        # `explanation` field so old rows still hydrate.
        explanation = str(entry.get("explanation") or entry.get("layman") or "").strip()
        if not explanation:
            tech = str(entry.get("technical") or "").strip()
            explanation = tech
        if not explanation:
            continue
        try:
            key_term_explanations.append(
                KeyTermExplanation(term=term, explanation=explanation)
            )
        except Exception:
            continue

    sections_raw = safe_json_loads(row.get("sections_json"), default=[])
    learning_sections: list[LearningSection] = []
    for entry in sections_raw:
        if not isinstance(entry, dict):
            continue
        try:
            learning_sections.append(LearningSection.model_validate(entry))
        except Exception:
            continue

    # v2.1 requires sections_json AND at least one key concept.
    if not learning_sections or not key_term_explanations:
        return None

    return SourceLearningSection(
        source_id=int(row["source_id"]),
        source_type=str(row["source_type"]),
        source_name=str(row["source_name"] or row["generated_title"] or "Source").strip(),
        generated_title=str(row["generated_title"]),
        sections=learning_sections,
        summary_text="",
        deep_dive_text="",
        under_surface_explainer="",
        key_terms=key_terms,
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
                model_name=GENERATION_MODEL,
            )
            quiz_section = CombinedQuizSection(
                questions=[],
                model_name=GENERATION_MODEL,
            )

        return GenerateTailoredLearningResponse(
            status_message="Tailored Socratic learning generated successfully.",
            source_ids=source_ids,
            video=video_section,
            documents=document_sections,
            insights=insights_section,
            quiz=quiz_section,
        )
