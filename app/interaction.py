from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

import config
from app.cost_logging import log_api_usage
from app.json_reliability import safe_json_loads
from app.llm_client import ChatResult, StreamState
from app.models import (
    AskModelOutput,
    AskResponse,
    InteractionSessionState,
    InteractionTurnRecord,
)
from app.retrieval import build_context_text, retrieve_context
from config import db_engine, get_llm_client
from prompts import (
    CHAT_JSON_SCHEMA,
    build_chat_system_prompt,
    chat_context_scale,
)

logger = logging.getLogger(__name__)

MAX_STORED_TURNS = 20
MAX_PROMPT_TURNS = 4
MAX_GENERATED_CONTEXT_CHARS = 2200


_ensured_engines: set[int] = set()


def ensure_interaction_tables() -> None:
    if id(db_engine) in _ensured_engines:
        return
    with db_engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS interaction_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        session_columns = {
            row[1]
            for row in connection.execute(
                text("PRAGMA table_info(interaction_sessions)")
            ).fetchall()
        }
        if "user_id" not in session_columns:
            connection.execute(
                text("ALTER TABLE interaction_sessions ADD COLUMN user_id TEXT DEFAULT 'legacy'")
            )
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_interaction_sessions_user_id
            ON interaction_sessions (user_id)
        """))
    _ensured_engines.add(id(db_engine))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_source_ids(source_ids: list[int]) -> list[int]:
    normalized = sorted({sid for sid in source_ids if sid > 0})
    if not normalized:
        raise ValueError("At least one valid source_id is required.")
    return normalized


def _build_in_clause(values: list[int], prefix: str) -> tuple[str, dict[str, int]]:
    params: dict[str, int] = {}
    keys: list[str] = []
    for idx, value in enumerate(values):
        key = f"{prefix}_{idx}"
        params[key] = value
        keys.append(f":{key}")
    return ", ".join(keys), params


def _truncate_text(text_value: str, max_chars: int) -> str:
    """Shorten to <= max_chars, preferring a sentence boundary.

    A blind character cut drops the back half of an enumeration ("A) ... and
    B) ..." -> "A) ...") and feeds the model a clause fragment. Instead, greedily
    keep whole sentences up to the budget; only hard-cut when the first sentence
    alone already exceeds it."""
    normalized = " ".join(text_value.split())
    if len(normalized) <= max_chars:
        return normalized

    budget = max_chars - 3  # leave room for the ellipsis
    kept = ""
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        candidate = f"{kept} {sentence}".strip() if kept else sentence
        if len(candidate) > budget:
            break
        kept = candidate

    if not kept:  # first sentence overruns the budget on its own
        kept = normalized[:budget].rstrip()
    return kept.rstrip() + "..."


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def _load_interaction_session(session_id: str, user_id: str) -> InteractionSessionState | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text("""
                SELECT state_json
                FROM interaction_sessions
                WHERE session_id = :session_id AND user_id = :user_id
                LIMIT 1
            """),
            {"session_id": session_id, "user_id": user_id},
        ).mappings().first()

    if row is None:
        return None

    state_raw = safe_json_loads(row.get("state_json"), default={})
    try:
        return InteractionSessionState.model_validate(state_raw)
    except Exception:
        return None


def _save_interaction_session(state: InteractionSessionState, user_id: str) -> None:
    state_json = state.model_dump_json()
    source_ids_json = json.dumps(sorted(state.source_ids), separators=(",", ":"))

    with db_engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO interaction_sessions (session_id, user_id, source_ids_json, state_json)
                VALUES (:session_id, :user_id, :source_ids_json, :state_json)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
            """),
            {
                "session_id": state.session_id,
                "user_id": user_id,
                "source_ids_json": source_ids_json,
                "state_json": state_json,
            },
        )


# ---------------------------------------------------------------------------
# Generated analysis context loader
# ---------------------------------------------------------------------------


def _load_generated_learning_context(
    source_ids: list[int], user_id: str, max_chars: int = MAX_GENERATED_CONTEXT_CHARS,
) -> str:
    if not source_ids:
        return ""

    in_clause, params = _build_in_clause(source_ids, "source_id")
    params["user_id"] = user_id

    with db_engine.connect() as connection:
        source_rows = connection.execute(
            text(f"""
                SELECT source_id, source_type, source_name, generated_title,
                       summary_text, deep_dive_text, key_terms_json
                FROM source_learning_sections
                WHERE user_id = :user_id AND source_id IN ({in_clause})
                ORDER BY source_type ASC, source_id ASC
            """),
            params,
        ).mappings().all()

    source_blocks: list[str] = []
    for row in source_rows:
        key_terms = safe_json_loads(row.get("key_terms_json"), default=[])
        key_terms_text = ", ".join(
            str(t).strip() for t in key_terms[:6] if str(t).strip()
        )
        source_name = str(row.get("source_name") or row.get("generated_title") or "source").strip()
        source_type = str(row.get("source_type") or "source").strip()
        source_blocks.append(
            f"{source_type.title()} [{source_name}]\n"
            f"Summary: {_truncate_text(str(row.get('summary_text') or ''), 280)}\n"
            f"Deep dive: {_truncate_text(str(row.get('deep_dive_text') or ''), 320)}\n"
            f"Key terms: {key_terms_text or 'n/a'}"
        )

    # Only the synthesis generated for THIS session's exact source set may be
    # injected — the newest row for the user can belong to a different pack,
    # which used to leak unrelated cross-source context into every prompt.
    session_set = set(source_ids)
    combined_row = None
    with db_engine.connect() as connection:
        combined_candidates = connection.execute(
            text("""
                SELECT video_source_id, document_source_ids_json,
                       synthesis_text, intersections_json
                FROM combined_learning_sections
                WHERE user_id = :user_id
                ORDER BY updated_at DESC, id DESC
                LIMIT 25
            """),
            {"user_id": user_id},
        ).mappings().all()

    for candidate in combined_candidates:
        row_ids = {
            int(v) for v in safe_json_loads(candidate.get("document_source_ids_json"), default=[])
            if isinstance(v, (int, float)) and int(v) > 0
        }
        video_id = int(candidate.get("video_source_id") or 0)
        if video_id > 0:
            row_ids.add(video_id)
        if row_ids == session_set:
            combined_row = candidate
            break

    combined_block = ""
    if combined_row is not None:
        synthesis = _truncate_text(str(combined_row.get("synthesis_text") or ""), 420)
        if synthesis:
            intersections = safe_json_loads(combined_row.get("intersections_json"), default=[])
            titles = []
            for entry in intersections[:4]:
                title = str(entry.get("title", "")).strip() if isinstance(entry, dict) else ""
                if title:
                    titles.append(title)
            combined_block = (
                "Cross-source synthesis:\n"
                f"Synthesis: {synthesis}\n"
                f"Intersections: {', '.join(titles) if titles else 'n/a'}"
            )

    full_context = "\n\n".join(p for p in ["\n\n".join(source_blocks), combined_block] if p).strip()
    if not full_context:
        return ""
    return _truncate_text(full_context, max_chars)


# ---------------------------------------------------------------------------
# Depth tier (pure heuristic — replaced an LLM classifier call per chat turn)
# ---------------------------------------------------------------------------
# The tier only scales how much context is assembled and the answer-length
# budget; a keyword/length heuristic routes it as well as a model did, at zero
# cost and zero latency. Misroutes degrade gracefully (slightly more or less
# context), never break correctness.

_DEEPEN_MARKERS = (
    "deeper", "more detail", "elaborate", "expand", "go on", "tell me more",
    "more depth", "keep going", "dig into",
)
_LOOKUP_PREFIXES = ("what is", "what's", "define", "who ", "when ", "where ", "which ")
_ANALYZE_MARKERS = (
    "compare", "contrast", "versus", " vs ", "trade-off", "tradeoff", "why ",
    "synthesize", "relationship", "difference", "implication", "evaluate",
    "critique", "analyze", "analyse",
)


def _heuristic_depth(query: str, recent_turns: list[InteractionTurnRecord]) -> str:
    """Map a query to a length tier so the answer scope matches the question scope."""
    q = " ".join(query.lower().split())
    if recent_turns and any(marker in q for marker in _DEEPEN_MARKERS):
        return "deepen"
    if any(marker in q for marker in _ANALYZE_MARKERS) or len(q) > 200:
        return "analyze"
    if len(q) < 60 and any(q.startswith(p) for p in _LOOKUP_PREFIXES):
        return "lookup"
    return "explain"


# ---------------------------------------------------------------------------
# Chat completion
# ---------------------------------------------------------------------------


def _chat_user_prompt(
    query: str,
    context_text: str,
    generated_context: str,
    recent_turns: list[InteractionTurnRecord],
) -> str:
    recent_turns_json = [turn.model_dump() for turn in recent_turns]
    return (
        f"User query:\n{query}\n\n"
        f"Generated learning context:\n{generated_context or '[None available]'}\n\n"
        f"Retrieved grounded context:\n{context_text}\n\n"
        f"Recent conversation turns:\n{json.dumps(recent_turns_json, ensure_ascii=True)}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@dataclass
class TurnPlan:
    """Everything a chat turn needs before the model is called: validated
    inputs, the loaded session, the depth tier and the assembled prompt."""

    state: InteractionSessionState
    query: str
    tier: str
    system_prompt: str
    user_prompt: str
    user_id: str


def prepare_turn(
    query: str,
    source_ids: list[int],
    session_id: str,
    *,
    user_id: str,
    top_k: int = 8,
    plain_output: bool = False,
) -> TurnPlan:
    """Validate, load the session, retrieve context and build the prompt.

    Shared by the JSON (`handle_user_query`) and streaming (`stream_user_query`)
    paths so both stay identical up to the model call."""
    ensure_interaction_tables()

    normalized_source_ids = _normalize_source_ids(source_ids)
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query cannot be empty.")

    session_id = session_id.strip()
    if not session_id:
        raise ValueError("session_id cannot be empty.")

    state = _load_interaction_session(session_id, user_id)
    if state is None:
        state = InteractionSessionState(
            session_id=session_id,
            source_ids=normalized_source_ids,
        )
    elif sorted(state.source_ids) != normalized_source_ids:
        raise ValueError("session_id source_ids do not match the existing session context.")

    recent_turns = state.turns[-MAX_PROMPT_TURNS:]

    tier = _heuristic_depth(normalized_query, recent_turns)
    scale = chat_context_scale(tier)
    generated_context = _load_generated_learning_context(
        normalized_source_ids, user_id,
        max_chars=max(400, int(MAX_GENERATED_CONTEXT_CHARS * scale)),
    )
    scaled_top_k = max(2, int(round(min(top_k, 8) * scale)))

    retrieved_context = None
    try:
        retrieved_context = retrieve_context(
            query=normalized_query,
            source_ids=normalized_source_ids,
            user_id=user_id,
            top_k=scaled_top_k,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to no-retrieval, never 500 the turn
        if not isinstance(exc, ValueError):
            logger.warning("Retrieval failed for session %s: %s", session_id, exc)

    if retrieved_context is not None:
        context_text = build_context_text(retrieved_context)
    else:
        context_text = "[No retrieval results]"

    return TurnPlan(
        state=state,
        query=normalized_query,
        tier=tier,
        system_prompt=build_chat_system_prompt(tier, plain=plain_output),
        user_prompt=_chat_user_prompt(normalized_query, context_text, generated_context, recent_turns),
        user_id=user_id,
    )


_EMPTY_ANSWER_FALLBACK = (
    "Your uploaded materials do not appear to cover this directly. "
    "Try rephrasing the question against something specific from the lecture or document."
)


def finish_turn(plan: TurnPlan, answer: str) -> AskResponse:
    """Record the answer on the session and persist it."""
    answer = (answer or "").strip() or _EMPTY_ANSWER_FALLBACK
    state = plan.state
    state.turns.append(InteractionTurnRecord(query=plan.query, answer=answer))
    if len(state.turns) > MAX_STORED_TURNS:
        state.turns = state.turns[-MAX_STORED_TURNS:]
    _save_interaction_session(state, plan.user_id)
    return AskResponse(session_id=state.session_id, answer=answer, model_name=config.chat_model())


def handle_user_query(
    query: str,
    source_ids: list[int],
    session_id: str,
    *,
    user_id: str,
    top_k: int = 8,
) -> AskResponse:
    plan = prepare_turn(query, source_ids, session_id, user_id=user_id, top_k=top_k)
    client = get_llm_client()
    result = client.chat_json(
        model=config.chat_model(),
        system=plan.system_prompt,
        user=plan.user_prompt,
        json_schema=CHAT_JSON_SCHEMA,
        temperature=0.3,
    )
    # Pass the ChatResult (which always carries usage), not the raw provider
    # payload: Ollama's raw dict has no `usage` key, which nulled telemetry.
    log_api_usage(response=result, user_id=user_id, call_stage="interaction", model_name=config.chat_model())
    if not result.content:
        raise RuntimeError("Chat model returned an empty response.")
    model_output = AskModelOutput.model_validate_json(result.content)
    return finish_turn(plan, model_output.answer)


def stream_user_query(
    query: str,
    source_ids: list[int],
    session_id: str,
    *,
    user_id: str,
    top_k: int = 8,
) -> Iterator[dict[str, Any]]:
    """Same turn as handle_user_query, yielded as events for a streaming route:
    {"delta": str}... then {"done": True, "session_id", "answer", "model_name"}."""
    plan = prepare_turn(query, source_ids, session_id, user_id=user_id, top_k=top_k, plain_output=True)
    client = get_llm_client()
    state = StreamState()
    parts: list[str] = []
    for delta in client.chat_stream(
        model=config.chat_model(),
        system=plan.system_prompt,
        user=plan.user_prompt,
        temperature=0.3,
        state=state,
    ):
        parts.append(delta)
        yield {"delta": delta}
    answer = "".join(parts)
    log_api_usage(
        response=ChatResult(content=answer, raw=None, usage=state.usage),
        user_id=user_id,
        call_stage="interaction",
        model_name=config.chat_model(),
    )
    response = finish_turn(plan, answer)
    yield {
        "done": True,
        "session_id": response.session_id,
        "answer": response.answer,
        "model_name": response.model_name,
    }
