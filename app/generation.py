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
from app.session_logger import get_active_logger, session_log
from app.linking import link_source_pair
from app.models import (
    ApplicationScenario,
    AttributedSentence,
    ClaimLedgerEntry,
    CombinedInsightSection,
    CombinedQuizSection,
    CrossSourceTension,
    GenerateTailoredLearningResponse,
    InsightIntersection,
    KeyTermExplanation,
    QuizQuestion,
    ReflectionPoint,
    SourceLearningSection,
    TransferBridge,
)
from app.processing import get_source_metadata, load_existing_source_summary, process_source
from config import (
    CACHE_PROCESSED_SOURCES,
    CONSOLIDATION_MODEL,
    ENABLE_STRICT_GENERATION_GATES,
    ENABLE_GENERATION_CRITIC_FALLBACK,
    GENERATION_CRITIC_MODEL,
    GENERATION_MODEL,
    GENERATION_PROGRESSIVE_CHUNKING,
    GENERATION_PROGRESSIVE_MIN_CHUNKS,
    GENERATION_PROGRESSIVE_ROUNDS,
    MAX_CHUNK_DIFF_WINDOWS,
    MAX_CROSS_CRITIC_CALLS_PER_RUN,
    MAX_SOURCE_CRITIC_CALLS_PER_RUN,
    db_engine,
    openai_client,
)
from prompts.chunk_diff import (
    CHUNK_DIFF_JSON_SCHEMA,
    CHUNK_DIFF_SYSTEM_PROMPT,
)
from prompts.noncomparative_quiz import (
    NONCOMPARATIVE_QUIZ_JSON_SCHEMA,
    NONCOMPARATIVE_QUIZ_SYSTEM_PROMPT,
)
from prompts.socratic_reflection import (
    SOCRATIC_REFLECTION_JSON_SCHEMA,
    SOCRATIC_REFLECTION_SYSTEM_PROMPT,
)
from prompts.source_content_pack import (
    SOURCE_CONTENT_PACK_JSON_SCHEMA,
    SOURCE_CONTENT_PACK_SYSTEM_PROMPT,
)
from prompts.source_deep_dive import (
    SOURCE_DEEP_DIVE_JSON_SCHEMA,
    SOURCE_DEEP_DIVE_SYSTEM_PROMPT,
)
from prompts.synthesis_consolidator import (
    SYNTHESIS_CONSOLIDATOR_JSON_SCHEMA,
    SYNTHESIS_CONSOLIDATOR_SYSTEM_PROMPT,
)
from prompts.under_surface import UNDER_SURFACE_JSON_SCHEMA, UNDER_SURFACE_SYSTEM_PROMPT


GENERATION_SCHEMA_VERSION = 11
MAX_GROUNDING_CHUNKS = 10
MAX_GROUNDING_CHARS = 9000
MAX_UNDER_SURFACE_GROUNDING_CHUNKS = 6
MAX_UNDER_SURFACE_GROUNDING_CHARS = 7000
MAX_REFLECTION_GROUNDING_CHUNKS = 6
MAX_REFLECTION_GROUNDING_CHARS = 7000
GENERATION_MAX_STAGE_ATTEMPTS = 3
CROSS_SECTION_MAX_ATTEMPTS = 1
CROSS_SECTION_MAX_SECONDS = 240
MAX_STAGE_ERROR_CHARS = 700
DEDUPLICATION_SIMILARITY_THRESHOLD = 0.9
DEDUPLICATION_TOKEN_OVERLAP_THRESHOLD = 0.58
DEDUPLICATION_MIN_CHAR_RATIO = 0.35
DEDUPLICATION_MIN_TOKEN_LENGTH = 4
OUTLINE_MODEL_MIN_GROUNDING_ROWS = 999999
MAX_EXPANSION_GROUNDING_CHARS = 8000
MAX_DEEP_DIVE_WORDS = 900
SECTION_REDUNDANCY_RATIO_THRESHOLD = 0.34
SECTION_SIMILARITY_THRESHOLD = 0.82
SECTION_TOKEN_OVERLAP_THRESHOLD = 0.50
SOURCE_SECTION_PAIR_OVERLAP_THRESHOLD = 0.28
CROSS_SECTION_PAIR_OVERLAP_THRESHOLD = 0.22
SECTION_CLAIM_OVERLAP_THRESHOLD = 0.18
MAX_CLAIMS_PER_SECTION = 24
MIN_CLAIM_CHARS = 28

VARIABLE_FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "constraint": (r"\bconstraint(s)?\b", r"\blimit(s|ation)?\b", r"\bboundary\b"),
    "failure_mode": (r"\bfail(s|ure|ed|ing)?\b", r"\bbreak(s|down)?\b", r"\bcollapse(s|d)?\b"),
    "instability": (r"\binstabilit(y|ies)\b", r"\bvolatile\b", r"\boscillat(e|ion|ing)\b"),
    "assumption": (r"\bassum(es|ption|ed)?\b", r"\bpremise(s)?\b", r"\bimplicit\b"),
    "theoretical_limit": (r"\btheoretical\b", r"\bupper bound\b", r"\blower bound\b"),
    "tradeoff": (r"\btrade[- ]?off(s)?\b", r"\bat the expense of\b", r"\bcost of\b"),
    "latency": (r"\blatenc(y|ies)\b", r"\bdelay(s|ed)?\b"),
    "generalization": (r"\bgeneraliz(e|ation|es|ed)\b", r"\bout[- ]of[- ]distribution\b"),
    "interpretability": (r"\binterpretab(le|ility)\b", r"\bexplainab(le|ility)\b"),
    "stability": (r"\bstabilit(y|ies)\b", r"\brobust(ness)?\b", r"\bdrift\b"),
}

INTERACTION_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "connection": (r"\bconnect(s|ion|ed)?\b", r"\blink(s|ed|age)?\b", r"\boverlap(s|ped)?\b", r"\bbridge\b"),
    "transfer": (r"\btransfer(s|red|ring)?\b", r"\bapply(ing|ied)?\b", r"\badapt(s|ed|ation)?\b", r"\bport(s|ed)?\b"),
    "dependency": (r"\bdepend(s|ency|ent)?\b", r"\brequire(s|ment|d)?\b", r"\benable(s|d|r)?\b", r"\bprerequisite(s)?\b"),
    "constraint": (r"\bconstraint(s)?\b", r"\blimit(s|ation)?\b", r"\bbottleneck(s)?\b", r"\bfragile\b"),
    "tradeoff": (r"\btrade[- ]?off(s)?\b", r"\bat the expense of\b", r"\bdegrade(s|d|ation)?\b"),
}

DEEP_DIVE_OPERATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "constraint": (
        r"\bconstraint(s)?\b",
        r"\blimit(s|ation)?\b",
        r"\bboundary\b",
        r"\bthreshold(s)?\b",
        r"\bbottleneck(s)?\b",
    ),
    "failure_mode": (
        r"\bfail(s|ure|ed|ing)?\b",
        r"\bbreak(s|down)?\b",
        r"\bcollapse(s|d)?\b",
        r"\bdegrad(e|es|ed|ation)\b",
        r"\bunstable\b",
    ),
    "intervention": (
        r"\binterven(e|tion|ing)\b",
        r"\bmitigat(e|es|ed|ion)\b",
        r"\bprevent(s|ed|ing)?\b",
        r"\bcalibrat(e|es|ed|ion)\b",
        r"\badjust(s|ed|ment)?\b",
        r"\btune(s|d|ing)?\b",
        r"\brecover(y|ies)?\b",
    ),
}

EXPLANATORY_RESTATEMENT_PATTERNS: tuple[str, ...] = (
    r"\brefers to\b",
    r"\bmeans that\b",
    r"\bis defined as\b",
    r"\bin other words\b",
    r"\bput simply\b",
    r"\bthis concept\b",
)

DECISION_VALUE_PATTERNS: tuple[str, ...] = (
    r"\bshould\b",
    r"\bmust\b",
    r"\bdecid(e|es|ed|ing)\b",
    r"\bprioritiz(e|es|ed|ing)\b",
    r"\btrade[- ]?off\b",
    r"\bwhen\b",
    r"\bif\b",
    r"\bunless\b",
)

EXTENSION_MARKER_PATTERNS: tuple[str, ...] = (
    r"\bhowever\b",
    r"\bunless\b",
    r"\bexcept\b",
    r"\bunder\b",
    r"\bwhen\b",
    r"\bif\b",
    r"\btrade[- ]?off\b",
    r"\bchalleng(e|es|ed|ing)\b",
    r"\bextend(s|ed|ing)?\b",
    r"\boperationaliz(e|es|ed|ing)\b",
)

CROSS_STAGE_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "mapping": (
        r"\bconnect(s|ion|ed)?\b",
        r"\blink(s|ed|age)?\b",
        r"\bbridge\b",
        r"\balign(s|ment)?\b",
        r"\boverlap(s|ped)?\b",
        r"\bintersection(s)?\b",
        r"\bshared\b",
        r"\bcommon\b",
        r"\bboth\b",
        r"\bacross\b",
        r"\bbetween\b",
        r"\bsame\b",
        r"\bparallel(s)?\b",
        r"\bcorrespond(s|ing)?\b",
        r"\bconverg(e|es|ed|ence)\b",
        r"\breinforc(e|es|ed|ing|ement)\b",
    ),
    "constraint": (r"\bconstraint(s)?\b", r"\bbreakdown\b", r"\bfail(s|ure|ed|ing)?\b", r"\btrade[- ]?off\b"),
    "transfer": (r"\btransfer(s|red|ring)?\b", r"\badapt(s|ed|ation)?\b", r"\bapply(ing|ied)?\b", r"\bmitigat(e|ion|es)?\b"),
    "decision": (
        r"\bdecid(e|es|ed|ing)\b",
        r"\bchoose(s|n)?\b",
        r"\bprioritiz(e|es|ed|ing)\b",
        r"\bimplication(s)?\b",
        r"\bshould\b",
        r"\bwould\b",
        r"\brecommend(ed|ation|ing)?\b",
        r"\bmost likely\b",
    ),
}

TRANSFER_ACTION_PATTERNS: tuple[str, ...] = (
    r"\bapply(ing|ied)?\b",
    r"\badapt(s|ed|ation)?\b",
    r"\bimplement(s|ed|ation)?\b",
    r"\bexecute(s|d|ion)?\b",
    r"\bcalibrat(e|es|ed|ion)\b",
    r"\bmitigat(e|es|ed|ion)\b",
    r"\bmonitor(s|ed|ing)?\b",
    r"\btest(s|ed|ing)?\b",
    r"\bvalidate(s|d|ion)?\b",
)

ANALOGY_MARKER_PATTERNS: tuple[str, ...] = (
    r"\blike\b",
    r"\bas if\b",
    r"\banalog(y|ous|ies)\b",
    r"\bsimilar to\b",
    r"\bmetaphor\b",
    r"\bthink of\b",
    r"\bimagine\b",
)

CONCEPT_STOPWORDS: set[str] = {
    "about",
    "across",
    "after",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "every",
    "first",
    "from",
    "have",
    "into",
    "just",
    "later",
    "many",
    "might",
    "more",
    "most",
    "only",
    "other",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "very",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "would",
}

CROSS_STAGE_REANCHOR_SIMILARITY_THRESHOLD = 0.70
CROSS_STAGE_WITHIN_STAGE_SIMILARITY_THRESHOLD = 0.84
MAX_SENTENCES_PER_STAGE: dict[str, int] = {
    "mapping": 6,
    "constraint": 6,
    "transfer": 7,
    "decision": 7,
}
MAX_PARAGRAPHS_PER_STAGE: dict[str, int] = {
    "mapping": 2,
    "constraint": 2,
    "transfer": 3,
    "decision": 3,
}

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

_CRITIC_BUDGET_BY_RUN: dict[str, dict[str, int]] = {}


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


def _resolve_critic_bucket(stage_name: str) -> str | None:
    lowered = str(stage_name or "").lower()
    if lowered.startswith("source_") or lowered.startswith("source:"):
        return "source"
    if lowered.startswith("combined_") or lowered.startswith("cross") or lowered.startswith("comparative") or lowered.startswith("application_"):
        return "cross"
    return None


def _ensure_critic_budget(run_id: str) -> dict[str, int]:
    if run_id not in _CRITIC_BUDGET_BY_RUN:
        _CRITIC_BUDGET_BY_RUN[run_id] = {
            "source": max(int(MAX_SOURCE_CRITIC_CALLS_PER_RUN), 0),
            "cross": max(int(MAX_CROSS_CRITIC_CALLS_PER_RUN), 0),
        }
    return _CRITIC_BUDGET_BY_RUN[run_id]


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
    model_name: str = GENERATION_MODEL,
) -> dict[str, Any]:
    last_error_detail = "Retry budget exhausted."

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
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if attempt_number > 1 and last_error_detail and last_error_detail != "Retry budget exhausted.":
                messages += [
                    {"role": "assistant", "content": "(previous attempt returned malformed output)"},
                    {
                        "role": "user",
                        "content": (
                            f"Previous attempt failed: {last_error_detail[:400]}. "
                            "Please return valid JSON exactly matching the required schema."
                        ),
                    },
                ]
            completion = openai_client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": response_schema},
            )
            log_api_usage(
                response=completion,
                user_id=user_id,
                call_stage="generation",
                model_name=model_name,
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
            usage = getattr(completion, "usage", None)
            get_active_logger().record(
                "stage_call",
                {
                    "stage_name": stage_name,
                    "attempt": attempt_number,
                    "duration_ms": duration_ms,
                    "model": model_name,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "critic": False,
                },
            )
            return payload
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            error_detail = _trim_error_detail(str(exc))
            last_error_detail = error_detail
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=stage_name,
                attempt_number=attempt_number,
                status="failed",
                duration_ms=duration_ms,
                error_message=error_detail,
            )

    critic_bucket = _resolve_critic_bucket(stage_name)
    critic_budget = _ensure_critic_budget(run_id)
    if (
        ENABLE_GENERATION_CRITIC_FALLBACK
        and critic_bucket is not None
        and critic_budget.get(critic_bucket, 0) > 0
    ):
        critic_attempt_number = GENERATION_MAX_STAGE_ATTEMPTS + 1
        started = time.perf_counter()
        critic_stage_name = f"{stage_name}:critic"
        log_generation_stage_event(
            run_id=run_id,
            user_id=user_id,
            stage_name=critic_stage_name,
            attempt_number=critic_attempt_number,
            status="started",
            details=f"remaining_budget={critic_budget.get(critic_bucket, 0)}",
        )

        try:
            completion = openai_client.chat.completions.create(
                model=GENERATION_CRITIC_MODEL,
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
                model_name=GENERATION_CRITIC_MODEL,
            )

            content = completion.choices[0].message.content
            payload = parse_json_object(content or "", stage_name=f"{critic_stage_name} attempt {critic_attempt_number}")
            if required_keys:
                missing_keys = [key for key in required_keys if key not in payload]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")

            critic_budget[critic_bucket] = max(critic_budget.get(critic_bucket, 0) - 1, 0)
            duration_ms = int((time.perf_counter() - started) * 1000)
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=critic_stage_name,
                attempt_number=critic_attempt_number,
                status="succeeded",
                duration_ms=duration_ms,
                details=(
                    f"fallback_model={GENERATION_CRITIC_MODEL};"
                    f"remaining_budget={critic_budget.get(critic_bucket, 0)}"
                ),
            )
            usage = getattr(completion, "usage", None)
            get_active_logger().record(
                "stage_call",
                {
                    "stage_name": critic_stage_name,
                    "attempt": critic_attempt_number,
                    "duration_ms": duration_ms,
                    "model": GENERATION_CRITIC_MODEL,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "critic": True,
                    "remaining_budget": critic_budget.get(critic_bucket, 0),
                },
            )
            return payload
        except Exception as critic_exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            critic_error = _trim_error_detail(str(critic_exc))
            last_error_detail = f"mini={last_error_detail}; critic={critic_error}"
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=critic_stage_name,
                attempt_number=critic_attempt_number,
                status="failed",
                duration_ms=duration_ms,
                error_message=critic_error,
            )

    raise GenerationStageError(
        run_id=run_id,
        stage_name=stage_name,
        attempt_number=GENERATION_MAX_STAGE_ATTEMPTS,
        reason=last_error_detail,
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
        if "first_principles_synthesis" not in source_columns:
            connection.execute(
                text("ALTER TABLE source_learning_sections ADD COLUMN first_principles_synthesis TEXT DEFAULT ''")
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
                CREATE TABLE IF NOT EXISTS generation_claim_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    source_id INTEGER,
                    document_source_ids_json TEXT DEFAULT '',
                    section_type TEXT NOT NULL,
                    claim_hash TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_claim_ledger_user_run
                ON generation_claim_ledger (user_id, run_id, section_type)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_claim_ledger_claim_hash
                ON generation_claim_ledger (user_id, claim_hash)
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


def _extract_claims(text_value: str, *, limit: int = MAX_CLAIMS_PER_SECTION) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    for sentence in _split_into_sentences(_normalize_continuous_prose(text_value)):
        claim = " ".join(sentence.split()).strip()
        if len(claim) < MIN_CLAIM_CHARS:
            continue
        normalized = claim.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        claims.append(claim)
        if len(claims) >= limit:
            break
    return claims


def _extract_variable_families(text_value: str) -> set[str]:
    normalized = _normalize_continuous_prose(text_value).lower()
    if not normalized:
        return set()

    families: set[str] = set()
    for family, patterns in VARIABLE_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                families.add(family)
                break
    return families


def _section_has_required_variables(
    *,
    section_text: str,
    prior_texts: list[str],
    required_families: list[str],
) -> bool:
    if not _normalize_continuous_prose(section_text):
        return False

    section_families = _extract_variable_families(section_text)
    prior_families: set[str] = set()
    for value in prior_texts:
        prior_families.update(_extract_variable_families(value))

    new_families = section_families - prior_families
    if not new_families:
        return False
    if not required_families:
        return True
    return bool(new_families.intersection(set(required_families)))


def _format_intent_guidance(intent_entry: dict[str, Any]) -> str:
    role = str(intent_entry.get("role") or "").strip()
    required_variables = [str(item).strip() for item in intent_entry.get("required_variables", []) if str(item).strip()]
    forbidden_claims = [
        _truncate_for_prompt(str(item), limit=180)
        for item in intent_entry.get("forbidden_claims", [])
        if str(item).strip()
    ]

    lines = [
        f"Role contract: {role or 'not specified'}",
        "Required variable families: " + (", ".join(required_variables) if required_variables else "[none]"),
        "Forbidden prior claims:\n" + ("\n".join(f"- {item}" for item in forbidden_claims[:8]) if forbidden_claims else "- [none]"),
    ]
    return "\n".join(lines)


def _build_source_intent_map(
    *,
    summary_text: str,
    progression_outline: str,
) -> dict[str, dict[str, Any]]:
    baseline_claims = _extract_claims(f"{summary_text}\n\n{progression_outline}", limit=12)

    return {
        "deep_dive": {
            "role": "Constraints, boundary conditions, failure modes, and instability behavior only.",
            "required_variables": ["constraint", "failure_mode", "instability", "tradeoff"],
            "forbidden_claims": baseline_claims,
        },
        "under_surface": {
            "role": "Hidden assumptions, latent premises, and theoretical limits only.",
            "required_variables": ["assumption", "theoretical_limit", "stability", "generalization"],
            "forbidden_claims": baseline_claims,
        },
        "reflection": {
            "role": "Transfer diagnostics, decision boundaries, and counterexample-oriented reasoning only.",
            "required_variables": ["tradeoff", "failure_mode", "assumption", "constraint"],
            "forbidden_claims": baseline_claims,
        },
    }


def _normalize_evidence_key(text_value: str) -> str:
    normalized = _normalize_continuous_prose(text_value).lower()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _build_cross_intent_map(
    *,
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
) -> dict[str, dict[str, Any]]:
    source_claims = _extract_claims(
        "\n\n".join(
            [video_section.summary_text, video_section.deep_dive_text]
            + [section.summary_text for section in document_sections]
            + [section.deep_dive_text for section in document_sections]
        ),
        limit=24,
    )
    return {
        "connection": {
            "role": "Define cross-source relationships once with grounded evidence.",
            "required_types": ["connection"],
            "forbidden_claims": [],
        },
        "dependency": {
            "role": "Explain enabling and requiring relations without re-summarizing sources.",
            "required_types": ["dependency", "transfer"],
            "forbidden_claims": source_claims,
        },
        "tradeoff": {
            "role": "Isolate conflicts, degradations, and decision boundaries.",
            "required_types": ["tradeoff", "constraint"],
            "forbidden_claims": source_claims,
        },
        "friction": {
            "role": "Show breakdown points in applied transfer and mitigation steps.",
            "required_types": ["transfer", "constraint", "tradeoff"],
            "forbidden_claims": source_claims,
        },
        "quiz": {
            "role": "Assess misconceptions and edge conditions without re-teaching prior sections.",
            "required_types": ["tradeoff", "constraint", "dependency"],
            "forbidden_claims": source_claims,
        },
    }


def _classify_interaction_types(text_value: str) -> set[str]:
    normalized = _normalize_continuous_prose(text_value).lower()
    if not normalized:
        return set()

    found_types: set[str] = set()
    for type_name, patterns in INTERACTION_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                found_types.add(type_name)
                break
    return found_types


def _source_restatement_ratio(text_value: str, source_texts: list[str]) -> float:
    normalized = _normalize_continuous_prose(text_value)
    if not normalized:
        return 0.0
    ratios = [
        _token_overlap_ratio(normalized, source_text)
        for source_text in source_texts
        if _normalize_continuous_prose(source_text)
    ]
    return max(ratios) if ratios else 0.0


def _contains_pattern(text_value: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_continuous_prose(text_value).lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in patterns)


def _sentence_similarity_to_prior(sentence: str, prior_texts: list[str]) -> float:
    sentence_norm = _normalize_continuous_prose(sentence).lower()
    if not sentence_norm:
        return 0.0

    best = 0.0
    for prior_text in prior_texts:
        for prior_sentence in _split_into_sentences(_normalize_continuous_prose(prior_text)):
            prior_norm = prior_sentence.lower()
            if not prior_norm:
                continue
            best = max(best, SequenceMatcher(None, sentence_norm, prior_norm).ratio())
            best = max(best, _token_overlap_ratio(sentence_norm, prior_norm))
    return best


def _classify_deep_dive_operation(sentence: str) -> str | None:
    normalized = _normalize_continuous_prose(sentence).lower()
    if not normalized:
        return None

    if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["failure_mode"]):
        return "failure_mode"
    if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["intervention"]):
        return "intervention"
    if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["constraint"]):
        return "constraint"
    return None


def _sentence_adds_failure_or_decision_value(sentence: str) -> bool:
    normalized = _normalize_continuous_prose(sentence).lower()
    if not normalized:
        return False
    if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["failure_mode"]):
        return True
    if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["intervention"]):
        return True
    return _contains_pattern(normalized, DECISION_VALUE_PATTERNS)


def _is_extension_sentence(sentence: str) -> bool:
    return _contains_pattern(sentence, EXTENSION_MARKER_PATTERNS)


def _filter_deep_dive_sentences(deep_dive_text: str, prior_texts: list[str]) -> str:
    kept: list[str] = []
    seen_signatures: set[str] = set()

    for sentence in _split_into_sentences(_normalize_continuous_prose(deep_dive_text)):
        normalized_sentence = " ".join(sentence.split()).strip()
        if not normalized_sentence:
            continue
        signature = normalized_sentence.lower()
        if signature in seen_signatures:
            continue

        if _contains_pattern(normalized_sentence, EXPLANATORY_RESTATEMENT_PATTERNS):
            continue

        operation_type = _classify_deep_dive_operation(normalized_sentence)
        if operation_type not in {"constraint", "failure_mode", "intervention"}:
            continue

        if not _sentence_adds_failure_or_decision_value(normalized_sentence):
            continue

        similarity_to_prior = _sentence_similarity_to_prior(normalized_sentence, [*prior_texts, *kept])
        if similarity_to_prior >= 0.66 and not _is_extension_sentence(normalized_sentence):
            continue

        seen_signatures.add(signature)
        kept.append(normalized_sentence)

    return " ".join(kept).strip()


def _filter_under_surface_sentences(under_surface_text: str, prior_texts: list[str]) -> str:
    kept: list[str] = []
    seen_signatures: set[str] = set()
    required_under_surface_families = {"assumption", "theoretical_limit", "stability", "generalization"}

    for sentence in _split_into_sentences(_normalize_continuous_prose(under_surface_text)):
        normalized_sentence = " ".join(sentence.split()).strip()
        if not normalized_sentence:
            continue
        signature = normalized_sentence.lower()
        if signature in seen_signatures:
            continue

        if _contains_pattern(normalized_sentence, EXPLANATORY_RESTATEMENT_PATTERNS):
            continue

        families = _extract_variable_families(normalized_sentence)
        if not families.intersection(required_under_surface_families):
            continue

        similarity_to_prior = _sentence_similarity_to_prior(normalized_sentence, [*prior_texts, *kept])
        if similarity_to_prior >= 0.68 and not _is_extension_sentence(normalized_sentence):
            continue

        seen_signatures.add(signature)
        kept.append(normalized_sentence)

    return " ".join(kept).strip()


def _count_cross_stage_pattern_hits(text_value: str, patterns: tuple[str, ...]) -> int:
    normalized = _normalize_continuous_prose(text_value).lower()
    if not normalized:
        return 0
    return sum(1 for pattern in patterns if re.search(pattern, normalized))


def _ensure_mapping_signal(text_value: str) -> str:
    normalized = _normalize_continuous_prose(text_value)
    if not normalized:
        return normalized

    mapping_patterns = CROSS_STAGE_ROLE_PATTERNS["mapping"]
    if _count_cross_stage_pattern_hits(normalized, mapping_patterns) > 0:
        return normalized

    # Deterministic fallback: keep original mapping content but prepend one
    # explicit mapping anchor so stage-role validation doesn't fail on style.
    mapping_anchor = "The sources connect through a shared mechanism across both materials."
    return f"{mapping_anchor} {normalized}".strip()


def _ensure_constraint_signal(text_value: str) -> str:
    normalized = _normalize_continuous_prose(text_value)
    if not normalized:
        return normalized

    constraint_patterns = CROSS_STAGE_ROLE_PATTERNS["constraint"]
    if _count_cross_stage_pattern_hits(normalized, constraint_patterns) > 0:
        return normalized

    # Deterministic fallback: preserve content and prepend one explicit
    # constraint anchor so stage-role validation does not fail on style-only drift.
    constraint_anchor = "A key constraint appears when conditions tighten and failure risk increases."
    return f"{constraint_anchor} {normalized}".strip()


def _infer_cross_stage_role(text_value: str) -> str:
    scores = {
        role: _count_cross_stage_pattern_hits(text_value, patterns)
        for role, patterns in CROSS_STAGE_ROLE_PATTERNS.items()
    }
    best_role = max(scores, key=scores.get)
    return best_role if scores[best_role] > 0 else "unknown"


def _cross_stage_role_scores(text_value: str) -> dict[str, int]:
    return {
        role: _count_cross_stage_pattern_hits(text_value, patterns)
        for role, patterns in CROSS_STAGE_ROLE_PATTERNS.items()
    }


def _extract_concept_keys(text_value: str) -> set[str]:
    normalized = _normalize_continuous_prose(text_value).lower()
    if not normalized:
        return set()
    tokens = re.findall(r"[a-z0-9]{4,}", normalized)
    return {token for token in tokens if token not in CONCEPT_STOPWORDS}


def _has_analogy_marker(text_value: str) -> bool:
    return _contains_pattern(text_value, ANALOGY_MARKER_PATTERNS)


def _sentence_role_signal_score(stage_name: str, sentence: str) -> int:
    normalized = _normalize_continuous_prose(sentence)
    if not normalized:
        return 0

    score = _count_cross_stage_pattern_hits(normalized, CROSS_STAGE_ROLE_PATTERNS.get(stage_name, tuple()))
    if stage_name == "constraint":
        if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["failure_mode"]):
            score += 1
        if _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["constraint"]):
            score += 1
    elif stage_name == "transfer":
        if _contains_pattern(normalized, TRANSFER_ACTION_PATTERNS):
            score += 1
    elif stage_name == "decision":
        if _contains_pattern(normalized, DECISION_VALUE_PATTERNS):
            score += 1
    return score


def _sentence_adds_stage_value(stage_name: str, sentence: str) -> bool:
    normalized = _normalize_continuous_prose(sentence)
    if not normalized:
        return False

    if stage_name == "mapping":
        return _contains_pattern(normalized, CROSS_STAGE_ROLE_PATTERNS["mapping"])
    if stage_name == "constraint":
        return (
            _contains_pattern(normalized, CROSS_STAGE_ROLE_PATTERNS["constraint"])
            or _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["failure_mode"])
            or _contains_pattern(normalized, DEEP_DIVE_OPERATION_PATTERNS["constraint"])
        )
    if stage_name == "transfer":
        return (
            _contains_pattern(normalized, CROSS_STAGE_ROLE_PATTERNS["transfer"])
            or _contains_pattern(normalized, TRANSFER_ACTION_PATTERNS)
        )
    if stage_name == "decision":
        return (
            _contains_pattern(normalized, CROSS_STAGE_ROLE_PATTERNS["decision"])
            or _contains_pattern(normalized, DECISION_VALUE_PATTERNS)
            or _contains_pattern(normalized, CROSS_STAGE_ROLE_PATTERNS["constraint"])
        )
    return False


def _is_sentence_role_compatible(stage_name: str, sentence: str) -> bool:
    role_scores = _cross_stage_role_scores(sentence)
    expected = _sentence_role_signal_score(stage_name, sentence)
    other_best = max((score for role, score in role_scores.items() if role != stage_name), default=0)

    if expected <= 0 and not _sentence_adds_stage_value(stage_name, sentence):
        return False
    if other_best > expected + 1:
        return False
    return True


def _sentence_signature(sentence: str) -> str:
    concepts = sorted(_extract_concept_keys(sentence))
    if concepts:
        return "|".join(concepts[:2])
    return "generic"


def _split_into_paragraphs(text_value: str) -> list[str]:
    normalized = _normalize_continuous_prose(text_value)
    if not normalized:
        return []
    return [paragraph.strip() for paragraph in re.split(r"\n{2,}", normalized) if paragraph.strip()]


def _compose_stage_paragraphs(stage_name: str, sentences: list[str]) -> str:
    if not sentences:
        return ""

    max_sentences = MAX_SENTENCES_PER_STAGE.get(stage_name, 6)
    max_paragraphs = MAX_PARAGRAPHS_PER_STAGE.get(stage_name, 3)
    trimmed_sentences = sentences[:max_sentences]

    paragraphs: list[str] = []
    current_sentences: list[str] = []
    current_signature: str | None = None

    for sentence in trimmed_sentences:
        signature = _sentence_signature(sentence)
        if not current_sentences:
            current_sentences = [sentence]
            current_signature = signature
            continue

        if signature != current_signature or len(current_sentences) >= 2:
            paragraphs.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_signature = signature
        else:
            current_sentences.append(sentence)

    if current_sentences:
        paragraphs.append(" ".join(current_sentences).strip())

    deduplicated_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if deduplicated_paragraphs and _pairwise_overlap_ratio(paragraph, deduplicated_paragraphs[-1]) >= 0.60:
            continue
        deduplicated_paragraphs.append(paragraph)

    return "\n\n".join(deduplicated_paragraphs[:max_paragraphs]).strip()


def _record_cross_stage_sentence_concepts(
    *,
    user_id: str,
    run_id: str,
    stage_name: str,
    document_source_ids: list[int],
    sentence_concepts: list[tuple[str, set[str]]],
) -> None:
    if not sentence_concepts:
        return

    with db_engine.begin() as connection:
        for sentence, concept_keys in sentence_concepts:
            normalized_sentence = _normalize_continuous_prose(sentence)
            if not normalized_sentence:
                continue
            concept_blob = ",".join(sorted(concept_keys)[:12])
            claim_text = f"concepts={concept_blob or 'none'} | sentence={_truncate_for_prompt(normalized_sentence, limit=520)}"
            connection.execute(
                text(
                    """
                    INSERT INTO generation_claim_ledger (
                        user_id,
                        run_id,
                        source_id,
                        document_source_ids_json,
                        section_type,
                        claim_hash,
                        claim_text
                    ) VALUES (
                        :user_id,
                        :run_id,
                        :source_id,
                        :document_source_ids_json,
                        :section_type,
                        :claim_hash,
                        :claim_text
                    )
                    """
                ),
                {
                    "user_id": user_id,
                    "run_id": run_id,
                    "source_id": None,
                    "document_source_ids_json": _serialize_ids(document_source_ids),
                    "section_type": f"cross_concept:{stage_name}",
                    "claim_hash": hashlib.sha256(normalized_sentence.lower().encode("utf-8", errors="ignore")).hexdigest(),
                    "claim_text": claim_text,
                },
            )


def _tighten_stage_text(
    *,
    stage_name: str,
    text_value: str,
    prior_stage_texts: list[str],
    introduced_concepts: set[str] | None = None,
    analogy_state: dict[str, int] | None = None,
) -> tuple[str, list[tuple[str, set[str]]]]:
    normalized = _normalize_continuous_prose(text_value)
    if not normalized:
        return "", []

    baseline_concepts = set(introduced_concepts or set())
    if not baseline_concepts:
        for prior_text in prior_stage_texts:
            baseline_concepts.update(_extract_concept_keys(prior_text))

    tracker = analogy_state if analogy_state is not None else {"used": 0}
    kept_sentences: list[str] = []
    sentence_concepts: list[tuple[str, set[str]]] = []
    local_introduced = set(baseline_concepts)

    for sentence in _split_into_sentences(normalized):
        normalized_sentence = " ".join(sentence.split()).strip()
        if not normalized_sentence:
            continue

        if not _is_sentence_role_compatible(stage_name, normalized_sentence):
            continue

        concept_keys = _extract_concept_keys(normalized_sentence)
        has_new_concept = bool(concept_keys - local_introduced)
        adds_value = _sentence_adds_stage_value(stage_name, normalized_sentence)

        similarity_to_prior = _sentence_similarity_to_prior(normalized_sentence, [*prior_stage_texts, *kept_sentences])
        if similarity_to_prior >= CROSS_STAGE_REANCHOR_SIMILARITY_THRESHOLD and not adds_value:
            continue
        if stage_name != "mapping" and not has_new_concept and not adds_value and not _is_extension_sentence(normalized_sentence):
            continue

        if _has_analogy_marker(normalized_sentence):
            if tracker.get("used", 0) >= 1:
                if not (has_new_concept and _is_extension_sentence(normalized_sentence)):
                    continue
            else:
                tracker["used"] = tracker.get("used", 0) + 1

        if _sentence_similarity_to_prior(normalized_sentence, kept_sentences) >= CROSS_STAGE_WITHIN_STAGE_SIMILARITY_THRESHOLD:
            continue

        kept_sentences.append(normalized_sentence)
        if concept_keys:
            local_introduced.update(concept_keys)
        sentence_concepts.append((normalized_sentence, concept_keys))

    if not kept_sentences:
        fallback_sentences = _split_into_sentences(normalized)
        if stage_name == "mapping":
            fallback_sentences = _split_into_sentences(_ensure_mapping_signal(normalized))
        elif stage_name == "constraint":
            fallback_sentences = _split_into_sentences(_ensure_constraint_signal(normalized))

        if not fallback_sentences:
            return "", []

        kept_sentences = fallback_sentences[: min(2, len(fallback_sentences))]
        sentence_concepts = [(sentence, _extract_concept_keys(sentence)) for sentence in kept_sentences]

    tightened = _compose_stage_paragraphs(stage_name, kept_sentences)
    if stage_name == "mapping":
        tightened = _ensure_mapping_signal(tightened)
    elif stage_name == "constraint":
        tightened = _ensure_constraint_signal(tightened)

    return tightened, sentence_concepts


def _tighten_progressive_cross_stage_texts(
    *,
    stage_texts: dict[str, str],
    user_id: str | None = None,
    run_id: str | None = None,
    document_source_ids: list[int] | None = None,
) -> dict[str, str]:
    stage_order = ["mapping", "constraint", "transfer", "decision"]
    introduced_concepts: set[str] = set()
    analogy_state = {"used": 0}
    prior_texts: list[str] = []
    tightened: dict[str, str] = {}
    doc_ids = document_source_ids or []

    for stage_name in stage_order:
        stage_text = stage_texts.get(stage_name, "")
        tightened_text, sentence_concepts = _tighten_stage_text(
            stage_name=stage_name,
            text_value=stage_text,
            prior_stage_texts=prior_texts,
            introduced_concepts=introduced_concepts,
            analogy_state=analogy_state,
        )
        tightened[stage_name] = tightened_text
        prior_texts.append(tightened_text)
        for _, concept_keys in sentence_concepts:
            introduced_concepts.update(concept_keys)

        if user_id and run_id:
            _record_cross_stage_sentence_concepts(
                user_id=user_id,
                run_id=run_id,
                stage_name=stage_name,
                document_source_ids=doc_ids,
                sentence_concepts=sentence_concepts,
            )

    return tightened


def _mapping_claim_overlap_ratio(left_claims: list[str], right_claims: list[str]) -> float:
    left = {claim.lower() for claim in left_claims if claim.strip()}
    right = {claim.lower() for claim in right_claims if claim.strip()}
    if not left or not right:
        return 0.0
    shared = left.intersection(right)
    return len(shared) / float(min(len(left), len(right)))


def _build_progressive_cross_stage_texts(
    insights: CombinedInsightSection,
    quiz: CombinedQuizSection,
    *,
    user_id: str | None = None,
    run_id: str | None = None,
    document_source_ids: list[int] | None = None,
) -> dict[str, str]:
    cross_texts = _build_cross_section_text_map(insights, quiz)
    mapping_text = " ".join(
        part
        for part in [cross_texts.get("connection", ""), cross_texts.get("bridge", "")]
        if _normalize_continuous_prose(part)
    ).strip()
    constraint_text = " ".join(
        part
        for part in [cross_texts.get("dependency", ""), cross_texts.get("tradeoff", "")]
        if _normalize_continuous_prose(part)
    ).strip()
    transfer_text = cross_texts.get("friction", "")
    decision_text = " ".join(
        part
        for part in [cross_texts.get("quiz", ""), quiz.study_advice]
        if _normalize_continuous_prose(part)
    ).strip()

    stage_texts = {
        "mapping": mapping_text,
        "constraint": constraint_text,
        "transfer": transfer_text,
        "decision": decision_text,
    }

    return _tighten_progressive_cross_stage_texts(
        stage_texts=stage_texts,
        user_id=user_id,
        run_id=run_id,
        document_source_ids=document_source_ids,
    )


def _assert_progressive_cross_structure(
    *,
    stage_texts: dict[str, str],
    source_texts: list[str],
    mapping_lock_claims: list[str] | None,
) -> list[str]:
    required_stage_roles = {
        "mapping": "mapping",
        "constraint": "constraint",
        "transfer": "transfer",
        "decision": "decision",
    }

    for stage_name, expected_role in required_stage_roles.items():
        stage_text = _normalize_continuous_prose(stage_texts.get(stage_name, ""))
        if stage_name == "mapping" and stage_text:
            stage_text = _ensure_mapping_signal(stage_text)
            stage_texts[stage_name] = stage_text
        if stage_name == "constraint" and stage_text:
            stage_text = _ensure_constraint_signal(stage_text)
            stage_texts[stage_name] = stage_text
        if stage_name in {"mapping", "constraint"} and not stage_text:
            raise ValueError(f"cross-source progression failed: missing required stage '{stage_name}'")
        if not stage_text:
            continue

        role_scores = _cross_stage_role_scores(stage_text)
        expected_score = role_scores.get(expected_role, 0)
        if expected_score <= 0:
            raise ValueError(
                f"cross-source progression failed: stage '{stage_name}' missing required '{expected_role}' signal"
            )

        inferred_role = _infer_cross_stage_role(stage_text)
        if stage_name == "decision":
            transfer_score = role_scores.get("transfer", 0)
            # Quiz explanations can mention transfer actions, but they must still contain
            # a strong decision/implication signal to pass this stage.
            if inferred_role == "transfer" and expected_score + 1 < transfer_score:
                raise ValueError(
                    f"cross-source progression failed: stage '{stage_name}' resolved to '{inferred_role}'"
                )
            if inferred_role not in {"decision", "transfer"}:
                raise ValueError(
                    f"cross-source progression failed: stage '{stage_name}' resolved to '{inferred_role}'"
                )
            continue

        if stage_name == "constraint" and inferred_role in {"decision", "mapping"}:
            # Constraint sections may include upstream mapping or recommendation language.
            # If explicit constraint signal is present, keep this stage valid.
            continue

        if stage_name == "mapping" and inferred_role != "mapping":
            # Mapping text can include downstream phrasing. Keep it valid when
            # explicit mapping signal is present.
            continue

        if inferred_role != expected_role:
            raise ValueError(
                f"cross-source progression failed: stage '{stage_name}' resolved to '{inferred_role}'"
            )

    mapping_text = _ensure_mapping_signal(_normalize_continuous_prose(stage_texts.get("mapping", "")))
    stage_texts["mapping"] = mapping_text
    if not mapping_text:
        raise ValueError("cross-source progression failed: mapping stage empty")

    mapping_claims = _extract_claims(mapping_text, limit=14)
    if not mapping_claims:
        raise ValueError("cross-source progression failed: mapping stage missing substantive claims")

    if mapping_lock_claims is not None:
        lock_ratio = _mapping_claim_overlap_ratio(mapping_claims, mapping_lock_claims)
        if lock_ratio < 0.55:
            raise ValueError(
                f"cross-source progression failed: mapping drift across retries ({lock_ratio:.3f} < 0.550)"
            )

    source_restatement_thresholds = {
        "constraint": 0.45,
        "transfer": 0.40,
        "decision": 0.35,
    }
    for stage_name, threshold in source_restatement_thresholds.items():
        stage_text = _normalize_continuous_prose(stage_texts.get(stage_name, ""))
        if not stage_text:
            continue
        restatement_ratio = _source_restatement_ratio(stage_text, source_texts)
        if restatement_ratio > threshold:
            raise ValueError(
                f"cross-source progression failed: stage '{stage_name}' restates sources ({restatement_ratio:.3f} > {threshold:.3f})"
            )

        mapping_overlap = _section_redundancy_ratio(stage_text, [mapping_text])
        if mapping_overlap > 0.38:
            raise ValueError(
                f"cross-source progression failed: stage '{stage_name}' remaps connection content ({mapping_overlap:.3f} > 0.380)"
            )

    return mapping_claims


def _assert_interaction_stage_progression(
    *,
    stage_name: str,
    stage_text: str,
    used_types: set[str],
    required_new_types: set[str],
    source_texts: list[str],
    max_source_restatement_ratio: float,
) -> set[str]:
    normalized = _normalize_continuous_prose(stage_text)
    if not normalized:
        return set()

    stage_types = _classify_interaction_types(normalized)
    if not stage_types:
        raise ValueError(f"{stage_name} failed interaction typing: no interaction markers detected")

    new_types = stage_types - used_types
    if required_new_types and not new_types.intersection(required_new_types):
        raise ValueError(
            f"{stage_name} failed interaction typing: no new required interaction type in {sorted(stage_types)}"
        )

    restatement = _source_restatement_ratio(normalized, source_texts)
    if restatement > max_source_restatement_ratio:
        raise ValueError(
            f"{stage_name} restates source material too strongly: {restatement:.3f} > {max_source_restatement_ratio:.3f}"
        )

    return stage_types


def _record_claim_ledger_entries(
    *,
    user_id: str,
    run_id: str,
    source_id: int | None,
    document_source_ids: list[int],
    section_type: str,
    section_text: str,
) -> None:
    claims = _extract_claims(section_text)
    if not claims:
        return

    with db_engine.begin() as connection:
        for claim in claims:
            connection.execute(
                text(
                    """
                    INSERT INTO generation_claim_ledger (
                        user_id,
                        run_id,
                        source_id,
                        document_source_ids_json,
                        section_type,
                        claim_hash,
                        claim_text
                    ) VALUES (
                        :user_id,
                        :run_id,
                        :source_id,
                        :document_source_ids_json,
                        :section_type,
                        :claim_hash,
                        :claim_text
                    )
                    """
                ),
                {
                    "user_id": user_id,
                    "run_id": run_id,
                    "source_id": source_id,
                    "document_source_ids_json": _serialize_ids(document_source_ids),
                    "section_type": section_type,
                    "claim_hash": hashlib.sha256(claim.lower().encode("utf-8", errors="ignore")).hexdigest(),
                    "claim_text": claim,
                },
            )


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


def _pairwise_claim_overlap_ratio(left_text: str, right_text: str) -> float:
    left_claims = {claim.lower() for claim in _extract_claims(left_text)}
    right_claims = {claim.lower() for claim in _extract_claims(right_text)}
    if not left_claims or not right_claims:
        return 0.0
    shared = left_claims.intersection(right_claims)
    baseline = float(min(len(left_claims), len(right_claims)))
    if baseline <= 0.0:
        return 0.0
    return len(shared) / baseline


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

    _record_claim_ledger_entries(
        user_id=user_id,
        run_id=run_id,
        source_id=source_id,
        document_source_ids=document_source_ids,
        section_type=section_type,
        section_text=section_text,
    )

    get_active_logger().record(
        "novelty",
        {
            "section_type": section_type,
            "source_id": source_id,
            "overlap_ratio": round(overlap_ratio, 4),
            "overlap_sentences": overlap_count,
            "total_sentences": total_count,
            "expansion_applied": expansion_applied,
            "violates_threshold": overlap_ratio > SECTION_REDUNDANCY_RATIO_THRESHOLD,
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
    max_claim_overlap_ratio: float = SECTION_CLAIM_OVERLAP_THRESHOLD,
    enable_claim_gate: bool = False,
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

            if enable_claim_gate:
                claim_overlap_ratio = _pairwise_claim_overlap_ratio(
                    section_texts[left_label],
                    section_texts[right_label],
                )
                _record_pairwise_quality_metric(
                    run_id=run_id,
                    user_id=user_id,
                    section_type=section_group,
                    pair_key=f"{pair_key}::claims",
                    overlap_ratio=claim_overlap_ratio,
                )
                if claim_overlap_ratio > max_claim_overlap_ratio:
                    raise ValueError(
                        f"{section_group} claim overlap gate failed for {pair_key}: "
                        f"{claim_overlap_ratio:.3f} > {max_claim_overlap_ratio:.3f}"
                    )


def _find_worst_overlap_pair(
    *,
    section_texts: dict[str, str],
    max_overlap_ratio: float,
    max_claim_overlap_ratio: float,
    enable_claim_gate: bool,
) -> tuple[str, str, float, float] | None:
    labels = [label for label, text_value in section_texts.items() if _normalize_continuous_prose(text_value)]
    worst: tuple[str, str, float, float] | None = None
    worst_score = 0.0

    for index, left_label in enumerate(labels):
        for right_label in labels[index + 1:]:
            overlap_ratio = _pairwise_overlap_ratio(section_texts[left_label], section_texts[right_label])
            claim_overlap_ratio = (
                _pairwise_claim_overlap_ratio(section_texts[left_label], section_texts[right_label])
                if enable_claim_gate
                else 0.0
            )
            overlap_fail = overlap_ratio > max_overlap_ratio
            claim_fail = enable_claim_gate and claim_overlap_ratio > max_claim_overlap_ratio
            if not overlap_fail and not claim_fail:
                continue

            score = 0.0
            if max_overlap_ratio > 0.0:
                score = max(score, overlap_ratio / max_overlap_ratio)
            if enable_claim_gate and max_claim_overlap_ratio > 0.0:
                score = max(score, claim_overlap_ratio / max_claim_overlap_ratio)

            if score >= worst_score:
                worst_score = score
                worst = (left_label, right_label, overlap_ratio, claim_overlap_ratio)

    return worst


def _reflection_points_to_text(reflection_points: list[ReflectionPoint]) -> str:
    return " ".join(
        " ".join(part for part in [point.question, point.explanation] if part)
        for point in reflection_points
    ).strip()


def _apply_source_hard_cuts(
    *,
    summary_text: str,
    deep_dive_text: str,
    under_surface_explainer: str,
    first_principles_synthesis: str,
    reflection_points: list[ReflectionPoint],
    diagnostic_checklist: list[str],
    key_term_explanations: list[KeyTermExplanation],
) -> tuple[str, str, str, list[ReflectionPoint], list[str], list[KeyTermExplanation], list[str]]:
    cut_priority = {
        "reflection": 0,
        "under_surface": 1,
        "deep_dive": 2,
    }

    current_deep_dive = deep_dive_text
    current_under_surface = under_surface_explainer
    current_reflection_points = [point.model_copy(deep=True) for point in reflection_points]
    current_diagnostic_checklist = [item for item in diagnostic_checklist]
    current_key_term_explanations = [entry.model_copy(deep=True) for entry in key_term_explanations]
    applied_cuts: list[str] = []

    for _ in range(4):
        section_texts = {
            "summary": summary_text,
            "deep_dive": current_deep_dive,
            "under_surface": current_under_surface,
            "reflection": _reflection_points_to_text(current_reflection_points),
        }
        worst = _find_worst_overlap_pair(
            section_texts=section_texts,
            max_overlap_ratio=SOURCE_SECTION_PAIR_OVERLAP_THRESHOLD,
            max_claim_overlap_ratio=SECTION_CLAIM_OVERLAP_THRESHOLD,
            enable_claim_gate=True,
        )
        if worst is None:
            break

        left_label, right_label, _, _ = worst
        candidates = [
            label
            for label in [left_label, right_label]
            if label in cut_priority and _normalize_continuous_prose(section_texts.get(label, ""))
        ]
        if not candidates:
            break

        cut_label = sorted(candidates, key=lambda label: cut_priority[label])[0]
        applied_cuts.append(cut_label)

        if cut_label == "reflection":
            current_reflection_points = []
        elif cut_label == "under_surface":
            current_under_surface = ""
            current_diagnostic_checklist = []
            current_key_term_explanations = []
        elif cut_label == "deep_dive":
            current_deep_dive = ""

    return (
        current_deep_dive,
        current_under_surface,
        first_principles_synthesis,
        current_reflection_points,
        current_diagnostic_checklist,
        current_key_term_explanations,
        applied_cuts,
    )


def _apply_cross_hard_cuts(
    *,
    insights: CombinedInsightSection,
    quiz: CombinedQuizSection,
) -> tuple[CombinedInsightSection, CombinedQuizSection, list[str]]:
    cut_priority = {
        "quiz": 0,
        "friction": 1,
        "bridge": 2,
        "tradeoff": 3,
        "dependency": 4,
    }

    current_insights = insights.model_copy(deep=True)
    current_quiz = quiz.model_copy(deep=True)
    applied_cuts: list[str] = []

    for _ in range(6):
        section_texts = _build_cross_section_text_map(current_insights, current_quiz)
        worst = _find_worst_overlap_pair(
            section_texts=section_texts,
            max_overlap_ratio=CROSS_SECTION_PAIR_OVERLAP_THRESHOLD,
            max_claim_overlap_ratio=SECTION_CLAIM_OVERLAP_THRESHOLD,
            enable_claim_gate=True,
        )
        if worst is None:
            break

        left_label, right_label, _, _ = worst
        candidates = [
            label
            for label in [left_label, right_label]
            if label in cut_priority and _normalize_continuous_prose(section_texts.get(label, ""))
        ]
        if not candidates:
            break

        cut_label = sorted(candidates, key=lambda label: cut_priority[label])[0]
        applied_cuts.append(cut_label)

        if cut_label == "quiz":
            current_quiz = current_quiz.model_copy(update={"questions": [], "study_advice": ""})
        elif cut_label == "friction":
            current_insights = current_insights.model_copy(update={"application_scenarios": []})
        elif cut_label == "bridge":
            current_insights = current_insights.model_copy(update={"layman_bridge": ""})
        elif cut_label == "tradeoff":
            current_insights = current_insights.model_copy(update={"comparative_analysis": ""})
        elif cut_label == "dependency":
            current_insights = current_insights.model_copy(update={"synthesis_text": ""})

    return current_insights, current_quiz, applied_cuts


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


def _derive_title_from_filename(source_name: str, source_type: str) -> str:
    from pathlib import Path as _Path
    stem = _Path(source_name).stem if source_name else f"{source_type}_source"
    title = stem.replace("_", " ").replace("-", " ").strip()
    title = " ".join(w.capitalize() for w in title.split())
    return title[:60] or f"{source_type.capitalize()} Source"


def _generate_source_content_pack(
    *,
    summary_text: str,
    source_type: str,
    claim_ledger: list[ClaimLedgerEntry],
    progression_outline: str,
    grounding_chunks: list[str],
    user_id: str,
    run_id: str,
) -> tuple[str, list[str], str, str, list[ReflectionPoint]]:
    """Single call replacing deep_dive + under_surface + reflection."""
    claim_text = _format_claim_ledger(claim_ledger)
    chunk_text = "\n\n".join(grounding_chunks) if grounding_chunks else "[No grounding chunks available]"
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_content_pack:{source_type}",
        user_id=user_id,
        system_prompt=SOURCE_CONTENT_PACK_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Source summary (do not restate this — build on it):\n{summary_text}\n\n"
            f"Progression outline:\n{progression_outline}\n\n"
            "CLAIM LEDGER (novel facts not in summary — use as primary material for deep_dive):\n"
            f"{claim_text}\n\n"
            f"Grounding chunks:\n{chunk_text}"
        ),
        response_schema=SOURCE_CONTENT_PACK_JSON_SCHEMA,
        required_keys=["deep_dive_text", "key_terms", "first_principles_synthesis", "under_surface_explainer", "reflection_points"],
    )

    deep_dive_text = _truncate_to_max_words(
        _normalize_continuous_prose(str(payload.get("deep_dive_text", "")).strip()),
        max_words=MAX_DEEP_DIVE_WORDS,
    )
    key_terms = [
        str(t).strip() for t in payload.get("key_terms", []) if str(t).strip()
    ]
    first_principles_synthesis = str(payload.get("first_principles_synthesis", "")).strip()
    under_surface_explainer = _normalize_continuous_prose(
        str(payload.get("under_surface_explainer", "")).strip()
    )

    raw_points = payload.get("reflection_points", [])
    reflection_points: list[ReflectionPoint] = []
    for raw_point in raw_points:
        try:
            reflection_points.append(ReflectionPoint.model_validate(raw_point))
        except Exception:
            point_text = str(raw_point).strip()
            if point_text:
                reflection_points.append(
                    ReflectionPoint(
                        question=point_text,
                        explanation="Reflect on how this idea works in concrete situations.",
                        depth_level="foundational",
                    )
                )

    if not deep_dive_text:
        raise ValueError("Model returned an empty deep-dive section.")
    if len(key_terms) < 5:
        raise ValueError("Model returned too few key terms.")
    if len(reflection_points) < 4:
        raise ValueError("Model returned too few reflection points.")

    return deep_dive_text, key_terms, first_principles_synthesis, under_surface_explainer, reflection_points


def _build_chunk_windows(
    grounding_chunks: list[str],
    *,
    window_size: int = 4,
    max_windows: int | None = None,
) -> list[str]:
    """Split grounding chunks into overlapping windows for claim extraction."""
    if not grounding_chunks:
        return []

    effective_max = max_windows or MAX_CHUNK_DIFF_WINDOWS
    windows: list[str] = []
    for start in range(0, len(grounding_chunks), max(window_size - 1, 1)):
        window = grounding_chunks[start : start + window_size]
        if window:
            windows.append("\n\n".join(window))
        if len(windows) >= effective_max:
            break

    return windows


def _extract_chunk_diff_claims(
    *,
    summary_text: str,
    grounding_chunks: list[str],
    source_type: str,
    user_id: str,
    run_id: str,
) -> list[ClaimLedgerEntry]:
    """Stage 3: Extract novel claims from chunk windows that are NOT in the summary."""
    # Tune applied from a diagnostic run: on small sources the chunk-diff
    # prompt spends ~500+ tokens to extract 0 new claims (the model has too
    # little material to differentiate). Empirically the call earns its keep
    # only when there are 3+ chunks to actually compare across.
    if len(grounding_chunks) <= 2:
        return []

    windows = _build_chunk_windows(grounding_chunks)
    if not windows:
        return []

    all_claims: list[ClaimLedgerEntry] = []
    seen_claims: set[str] = set()

    for window_index, window_text in enumerate(windows):
        try:
            payload = _run_structured_generation_step(
                run_id=run_id,
                stage_name=f"chunk_diff:{source_type}:window_{window_index}",
                user_id=user_id,
                system_prompt=CHUNK_DIFF_SYSTEM_PROMPT,
                user_prompt=(
                    f"Source summary:\n{summary_text}\n\n"
                    f"Chunk window {window_index + 1} of {len(windows)}:\n{window_text}"
                ),
                response_schema=CHUNK_DIFF_JSON_SCHEMA,
                required_keys=["claims"],
            )
            for raw_claim in payload.get("claims", []):
                try:
                    entry = ClaimLedgerEntry.model_validate(raw_claim)
                    claim_normalized = entry.claim.lower().strip()
                    if claim_normalized in seen_claims:
                        continue
                    # Check token overlap with existing claims for dedup
                    is_duplicate = False
                    for existing in all_claims:
                        if _token_overlap_ratio(entry.claim, existing.claim) > 0.50:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        seen_claims.add(claim_normalized)
                        all_claims.append(entry)
                except Exception:
                    continue
        except Exception as exc:
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=f"chunk_diff:{source_type}:window_{window_index}",
                attempt_number=GENERATION_MAX_STAGE_ATTEMPTS,
                status="skipped",
                details=f"window_error={_trim_error_detail(str(exc))}",
            )

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name=f"chunk_diff_complete:{source_type}",
        attempt_number=1,
        status="succeeded",
        details=f"total_claims={len(all_claims)};windows_processed={len(windows)}",
    )
    return all_claims


def _filter_used_claims(
    full_claims: list[ClaimLedgerEntry],
    used_text: str,
) -> list[ClaimLedgerEntry]:
    """Remove claims whose content appears in used_text (token overlap > 0.5)."""
    return [
        claim for claim in full_claims
        if _token_overlap_ratio(claim.claim, used_text) < 0.50
    ]


def _format_claim_ledger(claims: list[ClaimLedgerEntry]) -> str:
    """Format claim ledger entries into a prompt-ready text block."""
    if not claims:
        return "[No novel claims extracted — analyze omissions and gaps instead]"
    lines = []
    for index, claim in enumerate(claims, start=1):
        lines.append(
            f"{index}. [{claim.claim_type}] {claim.claim} "
            f"(grounding: \"{claim.grounding_quote}\")"
        )
    return "\n".join(lines)


def _generate_source_deep_dive(
    summary_text: str,
    source_type: str,
    claim_ledger: list[ClaimLedgerEntry],
    progression_outline: str,
    intent_guidance: str,
    user_id: str,
    run_id: str,
) -> tuple[str, list[str]]:
    claim_text = _format_claim_ledger(claim_ledger)
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_deep_dive:{source_type}",
        user_id=user_id,
        system_prompt=SOURCE_DEEP_DIVE_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Source summary (context only, do not repeat):\n{summary_text}\n\n"
            f"Progression outline:\n{progression_outline}\n\n"
            f"Controller intent guidance:\n{intent_guidance or '[none]'}\n\n"
            "CLAIM LEDGER (novel facts NOT in the summary — use these as primary material):\n"
            f"{claim_text}"
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
    claim_ledger: list[ClaimLedgerEntry],
    intent_guidance: str,
    user_id: str,
    run_id: str,
) -> list[ReflectionPoint]:
    claim_text = _format_claim_ledger(claim_ledger)
    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name=f"source_reflection:{source_type}",
        user_id=user_id,
        system_prompt=SOCRATIC_REFLECTION_SYSTEM_PROMPT,
        user_prompt=(
            f"Source type: {source_type}\n\n"
            f"Summary:\n{summary_text}\n\n"
            f"Boundary conditions & failure modes:\n{deep_dive_text}\n\n"
            f"Progression outline:\n{progression_outline}\n\n"
            f"Controller intent guidance:\n{intent_guidance or '[none]'}\n\n"
            f"Claim ledger (novel facts):\n{claim_text}\n\n"
            "Generate Socratic reflection points with reasoning traps."
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
    intent_guidance: str,
    user_id: str,
    run_id: str,
) -> tuple[str, str, list[str], list[KeyTermExplanation]]:
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
            f"Controller intent guidance:\n{intent_guidance or '[none]'}\n\n"
            f"Grounding chunks:\n{chunk_text}"
        ),
        response_schema=UNDER_SURFACE_JSON_SCHEMA,
        required_keys=["under_surface_explainer", "first_principles_synthesis", "diagnostic_checklist", "key_term_explanations"],
    )
    under_surface_explainer = str(payload.get("under_surface_explainer") or "").strip()
    first_principles_synthesis = str(payload.get("first_principles_synthesis") or "").strip()
    diagnostic_checklist = [
        str(item).strip()
        for item in payload.get("diagnostic_checklist", [])
        if str(item).strip()
    ]
    key_term_explanations: list[KeyTermExplanation] = []
    for entry in payload.get("key_term_explanations", []):
        try:
            if isinstance(entry, dict) and "explanation" in entry and "layman" not in entry:
                entry["layman"] = entry.pop("explanation")
                entry["technical"] = ""
            key_term_explanations.append(KeyTermExplanation.model_validate(entry))
        except Exception:
            continue

    if not under_surface_explainer:
        raise ValueError("Model returned empty under-surface explainer text.")
    if len(diagnostic_checklist) < 3:
        raise ValueError("Model returned too few diagnostic checklist points.")

    # Fallback: if first_principles_synthesis is empty, concatenate technical fields.
    if not first_principles_synthesis and key_term_explanations:
        first_principles_synthesis = " ".join(
            entry.technical.strip() for entry in key_term_explanations if entry.technical.strip()
        )

    return under_surface_explainer, first_principles_synthesis, diagnostic_checklist, key_term_explanations


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
                    first_principles_synthesis,
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
                    :first_principles_synthesis,
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
                    first_principles_synthesis = excluded.first_principles_synthesis,
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
                "first_principles_synthesis": section.first_principles_synthesis,
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
                    first_principles_synthesis,
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
            if isinstance(entry, dict) and "explanation" in entry and "layman" not in entry:
                entry["layman"] = entry.pop("explanation")
                entry["technical"] = ""
            key_term_explanations.append(KeyTermExplanation.model_validate(entry))
        except Exception:
            continue

    under_surface_explainer = str(row.get("under_surface_text") or "").strip()

    if not str(row.get("summary_text") or "").strip() or not key_terms:
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
        first_principles_synthesis=str(row.get("first_principles_synthesis") or "").strip(),
        diagnostic_checklist=diagnostic_checklist,
        key_term_explanations=key_term_explanations,
        reflection_points=reflection_points,
        model_name=str(row["model_name"]),
        schema_version=int(row["schema_version"]),
    )


def _select_chronological_slice(
    rows: list[dict[str, Any]],
    *,
    end_index: int,
    max_chars: int,
) -> list[str]:
    """Return formatted grounding entries from rows[0:end_index] bounded by max_chars."""
    chunks: list[str] = []
    consumed = 0
    for row in rows[:max(end_index, 0)]:
        entry = _format_grounding_entry(row)
        if not entry:
            continue
        if consumed + len(entry) > max_chars and chunks:
            break
        if consumed + len(entry) > max_chars and not chunks:
            entry = entry[:max_chars]
        chunks.append(entry)
        consumed += len(entry)
    return chunks


def _progressive_content_pack(
    *,
    summary_text: str,
    source_type: str,
    claim_ledger: list[ClaimLedgerEntry],
    base_outline: str,
    grounding_rows: list[dict[str, Any]],
    user_id: str,
    run_id: str,
) -> tuple[str, list[str], str, str, list[ReflectionPoint]]:
    """Multi-round build: round k gets chronological [0 : k*N/total] chunks
    plus the prior round's pack as carried-over context. Last round's output
    is the official content pack.

    Experimental — gated by GENERATION_PROGRESSIVE_CHUNKING. Logs per-round
    telemetry via the active session logger so the diagnostic can compare it
    against the single-call baseline.
    """
    rounds = max(2, int(GENERATION_PROGRESSIVE_ROUNDS))
    total_rows = len(grounding_rows)
    logger = get_active_logger()

    deep_dive_text = ""
    key_terms: list[str] = []
    first_principles_synthesis = ""
    under_surface_explainer = ""
    reflection_points: list[ReflectionPoint] = []

    for round_index in range(1, rounds + 1):
        end_index = max(1, (round_index * total_rows) // rounds)
        slice_chunks = _select_chronological_slice(
            grounding_rows,
            end_index=end_index,
            max_chars=MAX_GROUNDING_CHARS,
        )

        # Thread prior round's output into the outline so the model treats it
        # as "what you have so far; extend, do not restart".
        if deep_dive_text:
            carried = (
                f"\n\nCarried-over deep dive from prior round (do not restate; extend with new claims only):\n"
                f"{deep_dive_text[:1800]}"
            )
            if under_surface_explainer:
                carried += (
                    f"\n\nCarried-over under-surface from prior round:\n"
                    f"{under_surface_explainer[:1200]}"
                )
            outline = base_outline + carried
        else:
            outline = base_outline

        (
            deep_dive_text,
            key_terms,
            first_principles_synthesis,
            under_surface_explainer,
            reflection_points,
        ) = _generate_source_content_pack(
            summary_text=summary_text,
            source_type=source_type,
            claim_ledger=claim_ledger,
            progression_outline=outline,
            grounding_chunks=slice_chunks,
            user_id=user_id,
            run_id=f"{run_id}:p{round_index}",
        )

        logger.record(
            "progressive_round",
            {
                "round": round_index,
                "total_rounds": rounds,
                "chunks_used": len(slice_chunks),
                "deep_dive_chars": len(deep_dive_text or ""),
                "under_surface_chars": len(under_surface_explainer or ""),
                "reflections": len(reflection_points or []),
            },
        )

    return (
        deep_dive_text,
        key_terms,
        first_principles_synthesis,
        under_surface_explainer,
        reflection_points,
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

    # Strip markdown headers that may exist in cached summaries from the old prompt format.
    summary_text = _normalize_continuous_prose(summary_text)

    grounding_rows = _load_grounding_chunk_rows(source_id, user_id)
    grounding_chunks, grounding_chunk_indexes = _select_distributed_grounding_chunks(
        grounding_rows,
        max_chunks=MAX_GROUNDING_CHUNKS,
        max_chars=MAX_GROUNDING_CHARS,
    )

    log_generation_stage_event(
        run_id=run_id,
        user_id=user_id,
        stage_name=f"source_grounding_selection:{source_type}",
        attempt_number=1,
        status="succeeded",
        details=f"total_rows={len(grounding_rows)};selected={grounding_chunk_indexes}",
    )

    # Always use heuristic outline — cheaper and sufficient for deep dive context
    progression_outline = _build_heuristic_progression_outline(
        summary_text=summary_text,
        grounding_chunks=grounding_chunks,
    )

    generated_title = _derive_title_from_filename(str(source_name), source_type)

    # Extract novel claims (1 window only)
    claim_ledger = _extract_chunk_diff_claims(
        summary_text=summary_text,
        grounding_chunks=grounding_chunks,
        source_type=source_type,
        user_id=user_id,
        run_id=run_id,
    )

    # Single combined call by default. With GENERATION_PROGRESSIVE_CHUNKING=true
    # we run N rounds with growing chronological slices, threading the previous
    # round's output through progression_outline. Costs ~N× tokens; promote only
    # if the AB diagnostic shows clear quality gain.
    if (
        GENERATION_PROGRESSIVE_CHUNKING
        and len(grounding_rows) >= GENERATION_PROGRESSIVE_MIN_CHUNKS
    ):
        (
            deep_dive_text,
            key_terms,
            first_principles_synthesis,
            under_surface_explainer,
            reflection_points,
        ) = _progressive_content_pack(
            summary_text=summary_text,
            source_type=source_type,
            claim_ledger=claim_ledger,
            base_outline=progression_outline,
            grounding_rows=grounding_rows,
            user_id=user_id,
            run_id=run_id,
        )
    else:
        deep_dive_text, key_terms, first_principles_synthesis, under_surface_explainer, reflection_points = (
            _generate_source_content_pack(
                summary_text=summary_text,
                source_type=source_type,
                claim_ledger=claim_ledger,
                progression_outline=progression_outline,
                grounding_chunks=grounding_chunks,
                user_id=user_id,
                run_id=run_id,
            )
        )

    # Deterministic dedup (no LLM calls)
    deep_dive_text = _truncate_to_max_words(
        _enforce_section_novelty(deep_dive_text, [summary_text, progression_outline]),
        max_words=MAX_DEEP_DIVE_WORDS,
    )
    under_surface_explainer = _deduplicate_section_text(
        under_surface_explainer,
        [summary_text, deep_dive_text],
    )
    reflection_points = [
        point.model_copy(
            update={
                "explanation": _enforce_section_novelty(
                    point.explanation,
                    [summary_text, deep_dive_text, under_surface_explainer],
                ),
            }
        )
        for point in reflection_points
    ]

    low_mechanism_density = False
    reflection_text = _reflection_points_to_text(reflection_points)

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
            enable_claim_gate=True,
        )
    except Exception:
        # Hard cuts as last resort — no extra LLM calls
        (
            deep_dive_text,
            under_surface_explainer,
            first_principles_synthesis,
            reflection_points,
            _diagnostic_checklist,
            _key_term_explanations,
            applied_cuts,
        ) = _apply_source_hard_cuts(
            summary_text=summary_text,
            deep_dive_text=deep_dive_text,
            under_surface_explainer=under_surface_explainer,
            first_principles_synthesis=first_principles_synthesis,
            reflection_points=reflection_points,
            diagnostic_checklist=[],
            key_term_explanations=[],
        )
        for cut_label in applied_cuts:
            log_generation_stage_event(
                run_id=run_id,
                user_id=user_id,
                stage_name=f"source_hard_cut:{source_id}",
                attempt_number=1,
                status="succeeded",
                details=f"cut_section={cut_label}",
            )
            get_active_logger().record(
                "hard_cut",
                {"source_id": source_id, "cut_section": cut_label},
            )
        low_mechanism_density = True
        get_active_logger().record(
            "low_mechanism_density",
            {"source_id": source_id, "applied_cuts": applied_cuts},
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
        first_principles_synthesis=first_principles_synthesis,
        diagnostic_checklist=[],
        key_term_explanations=[],
        reflection_points=reflection_points,
        low_mechanism_density=low_mechanism_density,
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


def _generate_synthesis_consolidation(
    video_section: SourceLearningSection,
    document_sections: list[SourceLearningSection],
    user_id: str,
    run_id: str,
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    source_catalog = [
        {
            "source_id": video_section.source_id,
            "source_type": "video",
            "title": video_section.generated_title,
            "summary": video_section.summary_text,
            "deep_dive": video_section.deep_dive_text,
        }
    ] + [
        {
            "source_id": section.source_id,
            "source_type": "document",
            "title": section.generated_title,
            "summary": section.summary_text,
            "deep_dive": section.deep_dive_text,
        }
        for section in document_sections
    ]

    payload = _run_structured_generation_step(
        run_id=run_id,
        stage_name="synthesis_consolidation",
        user_id=user_id,
        system_prompt=SYNTHESIS_CONSOLIDATOR_SYSTEM_PROMPT,
        user_prompt=(
            f"Here are the summaries and deep dives for {len(source_catalog)} sources:\n\n"
            + "\n\n".join(
                f"Source: {s['title']}\n"
                f"Type: {s['source_type']}\n"
                f"Summary:\n{s['summary']}\n\n"
                f"Deep Dive:\n{s['deep_dive']}"
                for s in source_catalog
            )
        ),
        response_schema=SYNTHESIS_CONSOLIDATOR_JSON_SCHEMA,
        required_keys=["synthesis_text", "intersections", "application_scenarios", "questions"],
        model_name=CONSOLIDATION_MODEL,
    )

    synthesis_text = payload.get("synthesis_text", "").strip()
    raw_questions = payload.get("questions", [])
    raw_intersections = payload.get("intersections", []) or []
    raw_application_scenarios = payload.get("application_scenarios", []) or []

    questions = []
    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        questions.append(
            QuizQuestion(
                question=str(q.get("question", "")).strip(),
                options=[str(opt).strip() for opt in q.get("options", [])[:4]],
                answer_index=int(q.get("answer_index", 0)),
                explanation=str(q.get("explanation", "")).strip(),
                source_evidence=[],
            )
        )

    # Build intersections from the model output. Each intersection gets one
    # attributed_sentence per source so the downstream UI has provenance.
    all_sections = [video_section] + list(document_sections)
    default_emphasis = []
    for s in all_sections[:2]:
        default_emphasis.extend(s.key_terms[:3])
    intersections: list[InsightIntersection] = []
    for entry in raw_intersections:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("intersection_title", "")).strip()
        why = str(entry.get("why_it_matters", "")).strip()
        integrated = str(entry.get("integrated_explanation", "")).strip()
        if not (title and why and integrated):
            continue
        attributed: list[AttributedSentence] = []
        for section in all_sections[:3]:
            snippet = _truncate_sentence(
                _normalize_continuous_prose(section.deep_dive_text),
                limit=200,
            )
            if snippet:
                attributed.append(
                    AttributedSentence(
                        text=snippet,
                        source_id=section.source_id,
                        source_type=section.source_type,
                        emphasis_terms=[t for t in section.key_terms[:3] if t.strip()],
                    )
                )
        intersections.append(
            InsightIntersection(
                intersection_title=title,
                why_it_matters=why,
                integrated_explanation=integrated,
                attributed_sentences=attributed,
            )
        )

    # Fallback to the prior post-hoc fabrication if the model returned nothing.
    if not intersections:
        for section in all_sections[:3]:
            summary_sentence = _truncate_sentence(_normalize_continuous_prose(section.summary_text), limit=240)
            deep_sentence = _truncate_sentence(_normalize_continuous_prose(section.deep_dive_text), limit=240)
            intersections.append(
                InsightIntersection(
                    intersection_title=f"Focus Area: {section.generated_title}",
                    why_it_matters=summary_sentence or f"This source defines the core mechanism for {section.generated_title}.",
                    integrated_explanation=deep_sentence or "Use the boundary analysis and reflection points to pressure-test understanding.",
                    attributed_sentences=[
                        AttributedSentence(
                            text=summary_sentence or "Core mechanism focus area.",
                            source_id=section.source_id,
                            source_type=section.source_type,
                            emphasis_terms=[t for t in section.key_terms[:4] if t.strip()],
                        )
                    ],
                )
            )

    application_scenarios: list[ApplicationScenario] = []
    for entry in raw_application_scenarios:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("scenario_title", "")).strip()
        prompt = str(entry.get("scenario_prompt", "")).strip()
        steps = [str(step).strip() for step in (entry.get("transfer_steps") or []) if str(step).strip()]
        pitfall = str(entry.get("common_pitfall", "")).strip()
        if not (title and prompt and steps and pitfall):
            continue
        application_scenarios.append(
            ApplicationScenario(
                scenario_title=title,
                scenario_prompt=prompt,
                transfer_steps=steps[:4],
                common_pitfall=pitfall,
            )
        )

    insights = CombinedInsightSection(
        synthesis_text=synthesis_text,
        intersections=intersections,
        parallels=[],
        layman_bridge="",
        comparative_analysis="",
        application_scenarios=application_scenarios,
        model_name=CONSOLIDATION_MODEL,
    )

    quiz = CombinedQuizSection(
        questions=questions,
        study_advice="",
        model_name=CONSOLIDATION_MODEL,
    )

    return insights, quiz


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

    comparative_analysis = str(row.get("comparative_analysis_text") or "").strip()
    raw_application_scenarios = safe_json_loads(row.get("application_scenarios_json"), default=[])
    application_scenarios: list[ApplicationScenario] = []
    for entry in raw_application_scenarios:
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

    # Strip markdown heading markers that may have leaked from summary generation.
    normalized = re.sub(r"^#{1,6}\s*", "", normalized).strip()
    normalized = re.sub(r"\s*#{1,6}\s*", " ", normalized).strip()

    first_sentence = normalized.split(". ", 1)[0].strip()
    candidate = first_sentence if first_sentence else normalized
    if len(candidate) > limit:
        return candidate[: limit - 3].rstrip() + "..."
    return candidate


def _build_noncomparative_insights_and_quiz(
    source_sections: list[SourceLearningSection],
    *,
    user_id: str = "",
    run_id: str = "",
) -> tuple[CombinedInsightSection, CombinedQuizSection]:
    if not source_sections:
        raise ValueError("source_sections cannot be empty.")

    # Build clean intersections without markdown header leaks.
    intersections: list[InsightIntersection] = []
    for section in source_sections[:3]:
        summary_clean = _normalize_continuous_prose(section.summary_text)
        deep_clean = _normalize_continuous_prose(section.deep_dive_text)
        summary_sentence = _truncate_sentence(summary_clean, limit=240)
        deep_sentence = _truncate_sentence(deep_clean, limit=240)
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
                    or "Use the boundary analysis and reflection points to pressure-test understanding."
                ),
                attributed_sentences=attributed_sentences,
            )
        )

    parallels = intersections[0].attributed_sentences if intersections else []
    primary = source_sections[0]
    layman_clean = _normalize_continuous_prose(primary.under_surface_explainer)
    summary_clean = _normalize_continuous_prose(primary.summary_text)
    layman_bridge = _truncate_sentence(layman_clean, limit=320) or _truncate_sentence(
        summary_clean,
        limit=320,
    )

    # Build source context for the LLM quiz call.
    quiz_context_parts: list[str] = []
    for section in source_sections[:3]:
        source_summary = _normalize_continuous_prose(section.summary_text)
        source_deep = _normalize_continuous_prose(section.deep_dive_text)
        source_under = _normalize_continuous_prose(section.under_surface_explainer)
        terms = ", ".join(section.key_terms[:8])
        quiz_context_parts.append(
            f"Source: {section.generated_title} (ID: {section.source_id}, type: {section.source_type})\n"
            f"Summary: {_truncate_for_prompt(source_summary, limit=600)}\n"
            f"Boundary analysis: {_truncate_for_prompt(source_deep, limit=400)}\n"
            f"Hidden assumptions: {_truncate_for_prompt(source_under, limit=300)}\n"
            f"Key terms: {terms}"
        )
    quiz_context = "\n\n---\n\n".join(quiz_context_parts)

    # LLM-generated quiz (replaces template stubs).
    synthesis_text = ""
    study_advice = ""
    quiz_questions: list[QuizQuestion] = []

    try:
        payload = _run_structured_generation_step(
            run_id=run_id or "noncomp",
            stage_name="noncomparative_quiz",
            user_id=user_id,
            system_prompt=NONCOMPARATIVE_QUIZ_SYSTEM_PROMPT,
            user_prompt=(
                f"Generate a challenging Socratic quiz from the following {len(source_sections)} source(s):\n\n"
                f"{quiz_context}"
            ),
            response_schema=NONCOMPARATIVE_QUIZ_JSON_SCHEMA,
            required_keys=["questions", "synthesis_text", "study_advice"],
        )
        synthesis_text = _normalize_continuous_prose(str(payload.get("synthesis_text", "")).strip())
        study_advice = _normalize_continuous_prose(str(payload.get("study_advice", "")).strip())

        for raw_question in payload.get("questions", []):
            try:
                quiz_questions.append(QuizQuestion.model_validate(raw_question))
            except Exception:
                continue

        if quiz_questions:
            log_generation_stage_event(
                run_id=run_id or "noncomp",
                user_id=user_id,
                stage_name="noncomparative_quiz_llm",
                attempt_number=1,
                status="succeeded",
                details=f"questions={len(quiz_questions)}",
            )
    except Exception as exc:
        log_generation_stage_event(
            run_id=run_id or "noncomp",
            user_id=user_id,
            stage_name="noncomparative_quiz_llm",
            attempt_number=1,
            status="failed",
            details=f"fallback_to_template={_trim_error_detail(str(exc))}",
        )

    # Fallback: if LLM quiz fails, produce minimal stub (but still cleaner than before).
    if not quiz_questions:
        for section in source_sections[:2]:
            clean_summary = _normalize_continuous_prose(section.summary_text)
            key_claim = _truncate_sentence(clean_summary, limit=180) or section.generated_title
            quiz_questions.append(
                QuizQuestion(
                    question=f"Which mechanism does '{section.generated_title}' identify as its primary contribution?",
                    options=[
                        key_claim,
                        "The source argues there is no reliable mechanism in this domain.",
                        "The source focuses exclusively on historical precedent without proposing new mechanisms.",
                        "The source dismisses all existing approaches without alternative.",
                    ],
                    answer_index=0,
                    explanation=f"The correct answer captures the core mechanism from '{section.generated_title}'.",
                    source_evidence=[key_claim, f"Review the full summary of source {section.source_id}."],
                )
            )

    if not synthesis_text:
        source_labels = ", ".join(section.generated_title for section in source_sections[:4])
        synthesis_text = (
            f"This session covers {len(source_sections)} source(s): {source_labels}. "
            "Use each source's boundary analysis and reflection points to deepen understanding before exploring cross-source connections in chat."
        )

    if not study_advice:
        study_advice = (
            "Start by reviewing each source's key terms and boundary conditions. "
            "Then test yourself with the quiz below, paying attention to the reasoning traps in each distractor."
        )

    insights_section = CombinedInsightSection(
        intersections=intersections,
        parallels=parallels,
        layman_bridge=layman_bridge or "Use the source summaries and boundary analyses as your grounding layer.",
        synthesis_text=synthesis_text,
        comparative_analysis="",
        application_scenarios=[],
        model_name=GENERATION_MODEL,
        schema_version=4,
    )
    quiz_section = CombinedQuizSection(
        questions=quiz_questions,
        study_advice=study_advice,
        model_name=GENERATION_MODEL,
        schema_version=2,
    )
    return insights_section, quiz_section


def _build_source_section_text_map(section: SourceLearningSection) -> dict[str, str]:
    reflection_text = " ".join(
        " ".join(
            part
            for part in [point.question.strip(), point.explanation.strip()]
            if part
        )
        for point in section.reflection_points
    ).strip()
    key_term_text = " ".join(
        f"{entry.term.strip()}: {entry.layman.strip()} {entry.technical.strip()}"
        for entry in section.key_term_explanations
        if entry.term.strip() and (entry.layman.strip() or entry.technical.strip())
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
                " ".join(item.strip() for item in question.source_evidence if item.strip()),
            ]
            if part
        )
        for question in quiz.questions
    ).strip()

    return {
        "connection": intersections_text,
        "bridge": insights.layman_bridge,
        "dependency": insights.synthesis_text,
        "tradeoff": insights.comparative_analysis,
        "friction": scenarios_text,
        "quiz": quiz_explanations,
    }


def generate_tailored_learning(
    video_source_id: int | None,
    document_source_ids: list[int],
    user_id: str,
) -> GenerateTailoredLearningResponse:
    ensure_generation_tables()
    run_id = f"gen-{uuid.uuid4().hex[:12]}"

    with session_log(run_id, user_id=user_id) as session:
        return _generate_tailored_learning_inner(
            video_source_id=video_source_id,
            document_source_ids=document_source_ids,
            user_id=user_id,
            run_id=run_id,
            session=session,
        )


def _generate_tailored_learning_inner(
    *,
    video_source_id: int | None,
    document_source_ids: list[int],
    user_id: str,
    run_id: str,
    session: Any,
) -> GenerateTailoredLearningResponse:
    normalized_video_id, normalized_document_ids = _normalize_ids(
        video_source_id,
        document_source_ids,
    )
    source_ids: list[int] = []
    if normalized_video_id is not None:
        source_ids.append(normalized_video_id)
    source_ids.extend(normalized_document_ids)

    session.record(
        "pipeline_start",
        {
            "video_source_id": normalized_video_id,
            "document_source_ids": normalized_document_ids,
            "n_sources": len(source_ids),
        },
    )

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

        if cached_combined is not None:
            insights_section, quiz_section = cached_combined
        else:
            try:
                insights_section, quiz_section = _generate_synthesis_consolidation(
                    video_section=video_section,
                    document_sections=document_sections,
                    user_id=user_id,
                    run_id=run_id,
                )
                _store_combined_learning_sections(
                    user_id=user_id,
                    video_source_id=normalized_video_id,
                    document_source_ids=normalized_document_ids,
                    insights=insights_section,
                    quiz=quiz_section,
                )
            except Exception as exc:
                log_generation_stage_event(
                    run_id=run_id,
                    user_id=user_id,
                    stage_name="combined_learning_generation",
                    attempt_number=1,
                    status="failed",
                    details=_trim_error_detail(str(exc)),
                )
                raise

    else:
        insights_section = CombinedInsightSection(
            synthesis_text="",
            intersections=[],
            parallels=[],
            layman_bridge="",
            comparative_analysis="",
            application_scenarios=[],
            model_name=CONSOLIDATION_MODEL,
        )
        quiz_section = CombinedQuizSection(
            questions=[],
            study_advice="",
            model_name=CONSOLIDATION_MODEL,
        )

    return GenerateTailoredLearningResponse(
        status_message="Tailored Socratic learning generated successfully.",
        source_ids=source_ids,
        video=video_section,
        documents=document_sections,
        insights=insights_section,
        quiz=quiz_section,
    )
