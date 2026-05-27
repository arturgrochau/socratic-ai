from __future__ import annotations

import json

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.json_reliability import safe_json_loads
from app.models import (
    AskModelOutput,
    AskResponse,
    InteractionSessionState,
    InteractionTurnRecord,
)
from app.retrieval import build_context_text, retrieve_context
from config import CHAT_MODEL, db_engine, get_llm_client
from prompts import CHAT_JSON_SCHEMA, CHAT_SYSTEM_PROMPT


MAX_STORED_TURNS = 20
MAX_PROMPT_TURNS = 4
MAX_GENERATED_CONTEXT_CHARS = 2200


def ensure_interaction_tables() -> None:
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
    normalized = " ".join(text_value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


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


def _load_generated_learning_context(source_ids: list[int], user_id: str) -> str:
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

    combined_row = None
    with db_engine.connect() as connection:
        combined_candidates = connection.execute(
            text("""
                SELECT synthesis_text, intersections_json
                FROM combined_learning_sections
                WHERE user_id = :user_id
                ORDER BY updated_at DESC, id DESC
                LIMIT 5
            """),
            {"user_id": user_id},
        ).mappings().all()

    for candidate in combined_candidates:
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
    return _truncate_text(full_context, MAX_GENERATED_CONTEXT_CHARS)


# ---------------------------------------------------------------------------
# Chat completion
# ---------------------------------------------------------------------------


def _run_chat_completion(
    query: str,
    context_text: str,
    generated_context: str,
    recent_turns: list[InteractionTurnRecord],
    user_id: str,
) -> AskModelOutput:
    recent_turns_json = [turn.model_dump() for turn in recent_turns]
    client = get_llm_client()

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Generated learning context:\n{generated_context or '[None available]'}\n\n"
        f"Retrieved grounded context:\n{context_text}\n\n"
        f"Recent conversation turns:\n{json.dumps(recent_turns_json, ensure_ascii=True)}"
    )

    result = client.chat_json(
        model=CHAT_MODEL,
        system=CHAT_SYSTEM_PROMPT,
        user=user_prompt,
        json_schema=CHAT_JSON_SCHEMA,
        temperature=0.3,
    )
    log_api_usage(
        response=result.raw,
        user_id=user_id,
        call_stage="interaction",
        model_name=CHAT_MODEL,
    )

    content = result.content
    if not content:
        raise RuntimeError("Chat model returned an empty response.")

    return AskModelOutput.model_validate_json(content)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def handle_user_query(
    query: str,
    source_ids: list[int],
    session_id: str,
    *,
    user_id: str,
    top_k: int = 8,
) -> AskResponse:
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
    generated_context = _load_generated_learning_context(normalized_source_ids, user_id)

    retrieved_context = None
    try:
        retrieved_context = retrieve_context(
            query=normalized_query,
            source_ids=normalized_source_ids,
            user_id=user_id,
            top_k=min(top_k, 8),
        )
    except ValueError:
        pass

    if retrieved_context is not None:
        context_text = build_context_text(retrieved_context)
    else:
        context_text = "[No retrieval results]"

    model_output = _run_chat_completion(
        query=normalized_query,
        context_text=context_text,
        generated_context=generated_context,
        recent_turns=recent_turns,
        user_id=user_id,
    )

    answer = (model_output.answer or "").strip()
    if not answer:
        answer = (
            "Your uploaded materials do not appear to cover this directly. "
            "Try rephrasing the question against something specific from the lecture or document."
        )

    state.turns.append(InteractionTurnRecord(query=normalized_query, answer=answer))
    if len(state.turns) > MAX_STORED_TURNS:
        state.turns = state.turns[-MAX_STORED_TURNS:]

    _save_interaction_session(state, user_id)

    return AskResponse(
        session_id=state.session_id,
        answer=answer,
        model_name=CHAT_MODEL,
    )
