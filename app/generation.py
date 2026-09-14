from __future__ import annotations

import contextvars
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from sqlalchemy import text

import config
from app.cost_logging import log_api_usage, log_generation_stage_event
from app.json_reliability import parse_json_object, safe_json_loads
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
from app.section_validator import validate_arc
from app.session_logger import get_active_logger, session_log
from config import (
    CACHE_PROCESSED_SOURCES,
    db_engine,
    get_aux_client,
    get_aux_model,
    get_llm_client,
)
from prompts import (
    CONSOLIDATION_JSON_SCHEMA,
    CONSOLIDATION_SYSTEM_PROMPT,
    INSIGHTS_JSON_SCHEMA,
    INSIGHTS_SYSTEM_PROMPT,
    LEDGER_EXTRACTION_JSON_SCHEMA,
    LEDGER_EXTRACTION_SYSTEM_PROMPT,
    QUIZ_JSON_SCHEMA,
    QUIZ_SYSTEM_PROMPT,
    SINGLE_SOURCE_INSIGHTS_SYSTEM_PROMPT,
)
from prompts.sections import (
    CROSS_SOURCE_SECTIONS,
    PER_SOURCE_SECTIONS,
)

GENERATION_SCHEMA_VERSION = 18  # v18: cache rows are keyed by provider+model, not just shape
LEDGER_SCHEMA_VERSION = 2  # v2: parallel context-free window extraction (map-reduce)
GENERATION_MAX_STAGE_ATTEMPTS = 3
# Thread-pool width for fan-out. Actual in-flight LLM calls are capped by the
# per-provider semaphores in config.llm_call_slots (hosted APIs absorb wide
# fan-out; a local Ollama daemon queues excess requests against its timeout).
MAX_PARALLEL_LLM_CALLS = 8

_T = TypeVar("_T")


def _run_parallel(jobs: list[Callable[[], _T]]) -> list[_T]:
    """Run blocking jobs concurrently, preserving order; re-raises the first error.

    Each job is wrapped in a copy of the caller's contextvars so the active
    session logger (a ContextVar) keeps working inside worker threads.
    """
    if not jobs:
        return []
    if len(jobs) == 1:
        return [jobs[0]()]
    wrapped = [
        (lambda job=job, ctx=contextvars.copy_context(): ctx.run(job)) for job in jobs
    ]
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_LLM_CALLS, len(jobs))) as pool:
        return list(pool.map(lambda fn: fn(), wrapped))


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


def _is_non_retryable(exc: Exception) -> bool:
    """Deterministic failures (bad model name, bad key) should surface on the
    first attempt instead of burning the full retry budget re-sending the
    entire prompt."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    return status in (401, 403, 404)


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
    temperature: float = 0.0,
    large_context: bool = False,
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
            with config.llm_call_slots(client.name):  # per-provider concurrency cap
                result = client.chat_json(
                    model=model_name,
                    system=system_prompt,
                    user=user_prompt,
                    json_schema=response_schema,
                    temperature=temperature,
                    large_context=large_context,
                )
            payload = parse_json_object(
                result.content, stage_name=f"{stage_name} attempt {attempt_number}",
            )
            if required_keys:
                missing = [k for k in required_keys if k not in payload]
                if missing:
                    raise ValueError(f"Missing required keys: {', '.join(missing)}")
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            last_error_detail = str(exc)[:700]
            log_generation_stage_event(
                run_id=run_id, user_id=user_id, stage_name=stage_name,
                attempt_number=attempt_number, status="failed",
                duration_ms=duration_ms, error_message=last_error_detail,
            )
            if _is_non_retryable(exc):
                break
            if attempt_number < GENERATION_MAX_STAGE_ATTEMPTS:
                time.sleep(0.5 * attempt_number)
            continue

        # Telemetry sits outside the try: a transient logging hiccup (e.g. a
        # SQLite lock) must never burn a retry attempt for a successful call.
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_api_usage(
            response=result, user_id=user_id,
            call_stage="generation", model_name=model_name,
        )
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

    raise GenerationStageError(
        run_id=run_id, stage_name=stage_name,
        attempt_number=GENERATION_MAX_STAGE_ATTEMPTS, reason=last_error_detail,
    )


# ---------------------------------------------------------------------------
# DB tables
# ---------------------------------------------------------------------------


_ensured_engines: set[int] = set()


def ensure_generation_tables() -> None:
    # Idempotent DDL, but PRAGMA-based migrations on every generate request are
    # pure overhead — run once per engine (startup calls this; requests skip).
    if id(db_engine) in _ensured_engines:
        return
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
            "key_takeaways_json": "TEXT DEFAULT '[]'",
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
    _ensured_engines.add(id(db_engine))


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
    """Map-reduce: extract every window in PARALLEL, then merge with code dedup.

    Map calls are context-free (no prior ledger in the prompt), which removes the
    old quadratic re-send of the accumulated ledger AND turns N serial round-trips
    into one parallel batch. The reduce step is `append_units` — the same
    token-overlap dedup the sequential design relied on — applied in window order
    so unit ids stay deterministic.
    """
    def _extract_window(idx: int, window: str) -> Callable[[], dict[str, Any]]:
        def job() -> dict[str, Any]:
            user_prompt = (
                f"Source: {source_name}\n\n"
                f"=== Material (section {idx + 1} of {len(windows)}) ===\n{window}\n\n"
                "Extract the knowledge units this material teaches."
            )
            return _run_structured_generation_step(
                run_id=run_id,
                stage_name=f"ledger:{source_name}:{idx + 1}",
                user_id=user_id,
                system_prompt=LEDGER_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=LEDGER_EXTRACTION_JSON_SCHEMA,
                required_keys=["units"],
                use_aux=True,
            )
        return job

    payloads = _run_parallel([_extract_window(idx, w) for idx, w in enumerate(windows)])

    ledger: list[KnowledgeUnit] = []
    for payload in payloads:  # window order => deterministic ids and dedup winner
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
        large_context=True,
    )


# ---------------------------------------------------------------------------
# Per-source section builder
# ---------------------------------------------------------------------------
# Section budgets/restatement are validated in pure code (app/section_validator
# .validate_arc) — the former LLM audit call was paying a model to count
# sentences and measure token overlap. One regen on violation, as before.


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
    violations = validate_arc(PER_SOURCE_SECTIONS, structured)
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
        model_name=cache_model_tag(),
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
    valid_source_ids: set[int],
) -> list[InsightIntersection]:
    """Keep only intersections grounded in >=2 *real* distinct source ids.

    Anti-fabrication, in two steps: first drop any attributed sentence citing a
    source id that wasn't actually loaded for this run (a hallucinated id), then
    require the surviving citations to span >=2 distinct sources. This rejects
    both fabricated source ids and single-source 'intersections', and never
    surfaces a fabricated attribution to the user."""
    grounded: list[InsightIntersection] = []
    for entry in raw_intersections:
        try:
            intersection = InsightIntersection.model_validate(entry)
        except Exception:
            continue
        real_sentences = [
            s for s in intersection.attributed_sentences if s.source_id in valid_source_ids
        ]
        source_ids = {s.source_id for s in real_sentences}
        if len(source_ids) >= 2:
            grounded.append(
                intersection.model_copy(update={"attributed_sentences": real_sentences})
            )
    return grounded


def _generate_quiz_payload(
    rendered_ledger: str,
    *,
    run_id: str,
    user_id: str,
    legend: str | None,
) -> dict[str, Any]:
    """Focused quiz call. `legend` present => multi-source comparison questions."""
    if legend is not None:
        scope_line = (
            "The ledger spans MULTIPLE sources; every question must test the "
            "relationship between them.\n\n"
            "=== SOURCE LEGEND ===\n" + legend + "\n\n"
        )
    else:
        scope_line = (
            "The ledger covers a SINGLE source; test its mechanisms, tradeoffs, "
            "and boundaries.\n\n"
        )
    # temperature 0.7: the quiz needs sampling entropy (varied correct-answer
    # positions, distinct distractors); greedy decoding biases the answer index.
    return _run_structured_generation_step(
        run_id=run_id,
        stage_name="quiz",
        user_id=user_id,
        system_prompt=QUIZ_SYSTEM_PROMPT,
        user_prompt=(
            scope_line + f"=== Knowledge ledger (id | type | source) ===\n{rendered_ledger}"
        ),
        response_schema=QUIZ_JSON_SCHEMA,
        required_keys=["questions"],
        temperature=0.7,
        large_context=True,
    )


def _parse_quiz_questions(payload: dict[str, Any], *, max_questions: int = 4) -> list[QuizQuestion]:
    questions: list[QuizQuestion] = []
    for entry in payload.get("questions", []):
        try:
            questions.append(QuizQuestion.model_validate(entry))
        except Exception:
            continue
    return questions[:max_questions]


def _parse_scenarios(payload: dict[str, Any]) -> list[ApplicationScenario]:
    scenarios: list[ApplicationScenario] = []
    for entry in payload.get("application_scenarios", []):
        try:
            scenarios.append(ApplicationScenario.model_validate(entry))
        except Exception:
            continue
    return scenarios


def _generate_synthesis(
    source_sections: list[SourceLearningSection],
    *,
    run_id: str,
    user_id: str,
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    merged = _merge_source_ledgers(source_sections, user_id=user_id)
    rendered = render_units(merged)

    # Source legend: refer to sources by name in prose, never by raw id number.
    legend_lines: list[str] = []
    for section in source_sections:
        prefix = "Video" if section.source_type == "video" else "Document"
        legend_lines.append(f"  source {section.source_id} = {prefix}: {section.generated_title}")
    legend = "\n".join(legend_lines) or "  (no named sources)"

    def _insights_call(extra_violations: list[str] | None = None) -> dict[str, Any]:
        user_prompt = (
            "Synthesize ACROSS the sources using only the following knowledge ledger. "
            "Every intersection must cite claims from at least two different sources via "
            "attributed_sentences. Do not introduce topics absent from the ledger.\n\n"
            "=== SOURCE LEGEND (refer to each source by this name in your prose; never "
            "write 'source <number>' or unit ids) ===\n"
            f"{legend}\n\n"
            f"=== Merged knowledge ledger (id | type | source) ===\n{rendered}"
        )
        if extra_violations:
            user_prompt += (
                "\n\nA prior draft was rejected for these contract violations. Fix them:\n- "
                + "\n- ".join(extra_violations)
            )
        return _run_structured_generation_step(
            run_id=run_id,
            stage_name="synthesis:insights",
            user_id=user_id,
            system_prompt=INSIGHTS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=INSIGHTS_JSON_SCHEMA,
            required_keys=[
                "key_takeaways", "synthesis_text", "intersections", "application_scenarios",
            ],
            large_context=True,
        )

    # Insights and quiz are independent given the same ledger — run them in
    # parallel. The quiz gets the claims without evidence quotes: its prompt
    # forbids citing evidence, so those tokens were pure prompt waste.
    rendered_no_evidence = render_units(merged, include_evidence=False)
    insights_payload, quiz_payload = _run_parallel([
        _insights_call,
        lambda: _generate_quiz_payload(
            rendered_no_evidence, run_id=run_id, user_id=user_id, legend=legend,
        ),
    ])

    violations = validate_arc(CROSS_SOURCE_SECTIONS, insights_payload)
    if violations:
        insights_payload = _insights_call(violations)

    valid_source_ids = {section.source_id for section in source_sections}
    intersections = _ground_intersections(
        insights_payload.get("intersections", []), valid_source_ids
    )
    key_takeaways = [
        str(t).strip() for t in (insights_payload.get("key_takeaways") or []) if str(t).strip()
    ][:3]

    insights = CombinedInsightSection(
        key_takeaways=key_takeaways,
        synthesis_text=str(insights_payload.get("synthesis_text", "")).strip(),
        intersections=intersections,
        application_scenarios=_parse_scenarios(insights_payload),
        model_name=cache_model_tag(),
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    quiz = CombinedQuizSection(
        questions=_parse_quiz_questions(quiz_payload),
        model_name=cache_model_tag(),
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    return insights, quiz


def _generate_single_source_deepening(
    section: SourceLearningSection,
    *,
    run_id: str,
    user_id: str,
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    """Deepen a lone source into takeaways + bigger-picture + apply-it scenarios +
    a quiz, grounded in that one ledger. This is what makes a single document a
    full learning artifact instead of a dead end. No cross-source intersections.

    Insights and quiz are two focused calls run in parallel (same wall-clock as
    one call; better per-artifact quality from a mini model)."""
    units = load_ledger(
        user_id=user_id, source_id=section.source_id, schema_version=LEDGER_SCHEMA_VERSION,
    ) or []
    rendered = render_units(units)

    def _insights_call() -> dict[str, Any]:
        user_prompt = (
            f"Source: {section.generated_title}\n\n"
            "Deepen this single source into key takeaways, a short bigger-picture "
            "reflection, and apply-it scenarios, using only the following knowledge "
            "ledger. Set intersections to an empty array.\n\n"
            f"=== Knowledge ledger (id | type | source) ===\n{rendered}"
        )
        return _run_structured_generation_step(
            run_id=run_id,
            stage_name="deepening:insights",
            user_id=user_id,
            system_prompt=SINGLE_SOURCE_INSIGHTS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=INSIGHTS_JSON_SCHEMA,
            required_keys=["key_takeaways", "synthesis_text", "application_scenarios"],
            large_context=True,
        )

    rendered_no_evidence = render_units(units, include_evidence=False)
    insights_payload, quiz_payload = _run_parallel([
        _insights_call,
        lambda: _generate_quiz_payload(
            rendered_no_evidence, run_id=run_id, user_id=user_id, legend=None,
        ),
    ])

    key_takeaways = [
        str(t).strip() for t in (insights_payload.get("key_takeaways") or []) if str(t).strip()
    ][:3]

    insights = CombinedInsightSection(
        key_takeaways=key_takeaways,
        synthesis_text=str(insights_payload.get("synthesis_text", "")).strip(),
        intersections=[],
        application_scenarios=_parse_scenarios(insights_payload),
        model_name=cache_model_tag(),
        schema_version=GENERATION_SCHEMA_VERSION,
    )
    quiz = CombinedQuizSection(
        questions=_parse_quiz_questions(quiz_payload),
        model_name=cache_model_tag(),
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


def cache_model_tag() -> str:
    """Value stored in the cache rows' model_name column.

    Includes the provider so a pack generated by gpt-4o-mini is never served
    as if it came from the local model (or vice versa) after a mode switch.
    The ledger cache is deliberately NOT keyed this way: it holds extracted
    facts, which are worth reusing across models."""
    return f"{config.get_settings().llm_provider}:{config.generation_model()}"


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
    if str(row.get("model_name") or "") != cache_model_tag():
        return None  # generated by another provider/model: regenerate rather than mislabel

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
                    synthesis_text, application_scenarios_json, key_takeaways_json,
                    quiz_json, schema_version, model_name
                ) VALUES (
                    :user_id, :video_source_id, :document_source_ids_json,
                    :intersections_json, :parallels_json, :layman_bridge,
                    :synthesis_text, :application_scenarios_json, :key_takeaways_json,
                    :quiz_json, :schema_version, :model_name
                )
                ON CONFLICT(user_id, video_source_id, document_source_ids_json) DO UPDATE SET
                    intersections_json = excluded.intersections_json,
                    parallels_json = excluded.parallels_json,
                    layman_bridge = excluded.layman_bridge,
                    synthesis_text = excluded.synthesis_text,
                    application_scenarios_json = excluded.application_scenarios_json,
                    key_takeaways_json = excluded.key_takeaways_json,
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
                # Legacy v1.x columns (NOT NULL in existing DBs): written as inert
                # constants; nothing reads them. Dropping needs a real migration.
                "parallels_json": "[]",
                "layman_bridge": "",
                "synthesis_text": insights.synthesis_text,
                "application_scenarios_json": json.dumps(
                    [e.model_dump() for e in insights.application_scenarios], ensure_ascii=True,
                ),
                "key_takeaways_json": json.dumps(insights.key_takeaways, ensure_ascii=True),
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
                       application_scenarios_json, key_takeaways_json, quiz_json,
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
    if str(row.get("model_name") or "") != cache_model_tag():
        return None  # generated by another provider/model: regenerate rather than mislabel

    intersections: list[InsightIntersection] = []
    for entry in safe_json_loads(row.get("intersections_json"), default=[]):
        try:
            intersections.append(InsightIntersection.model_validate(entry))
        except Exception:
            continue

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

    key_takeaways = [
        str(t).strip()
        for t in safe_json_loads(row.get("key_takeaways_json"), default=[])
        if str(t).strip()
    ]
    synthesis_text = str(row["synthesis_text"] or "").strip()

    # A valid cached artifact has *some* content. For a single source there are no
    # intersections, so accept the row if it has a quiz, takeaways, or synthesis.
    if not (intersections or quiz_section.questions or synthesis_text or key_takeaways):
        return None

    insights_section = CombinedInsightSection(
        key_takeaways=key_takeaways,
        synthesis_text=synthesis_text,
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

        # Sources are independent — build every section concurrently. (Each one
        # upserts rows keyed by (user_id, source_id), so writes don't collide.)
        ordered_ids: list[int] = (
            ([normalized_video_id] if normalized_video_id is not None else [])
            + normalized_document_ids
        )
        built_sections = _run_parallel([
            (lambda sid=sid: _build_source_learning_section(sid, user_id, run_id))
            for sid in ordered_ids
        ])
        by_id = dict(zip(ordered_ids, built_sections, strict=True))

        video_section = by_id.get(normalized_video_id) if normalized_video_id is not None else None
        document_sections = [by_id[sid] for sid in normalized_document_ids]

        all_sections = ([video_section] if video_section else []) + document_sections
        if not all_sections:
            raise ValueError("No source learning sections were generated.")

        # Both single- and multi-source runs produce an enrichment artifact
        # (takeaways + bigger picture + apply-it scenarios + quiz). Caching uses a
        # sentinel video id of 0 when there is no video, so docs-only runs cache too.
        cache_video_key = normalized_video_id if normalized_video_id is not None else 0
        cached_combined = None
        if CACHE_PROCESSED_SOURCES:
            cached_combined = _load_cached_combined_learning_sections(
                user_id=user_id,
                video_source_id=cache_video_key,
                document_source_ids=normalized_document_ids,
            )

        if cached_combined is not None:
            insights_section, quiz_section = cached_combined
        else:
            if len(all_sections) >= 2:
                insights_section, quiz_section = _generate_synthesis(
                    all_sections, run_id=run_id, user_id=user_id,
                )
            else:
                insights_section, quiz_section = _generate_single_source_deepening(
                    all_sections[0], run_id=run_id, user_id=user_id,
                )
            if CACHE_PROCESSED_SOURCES:
                _store_combined_learning_sections(
                    user_id=user_id,
                    video_source_id=cache_video_key,
                    document_source_ids=normalized_document_ids,
                    insights=insights_section,
                    quiz=quiz_section,
                )

        return GenerateTailoredLearningResponse(
            status_message="Tailored Socratic learning generated successfully.",
            source_ids=source_ids,
            video=video_section,
            documents=document_sections,
            insights=insights_section,
            quiz=quiz_section,
        )
