from __future__ import annotations

import json
import re

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.models import (
    AskModelOutput,
    AskResponse,
    InteractionSessionState,
    InteractionTurnRecord,
    RetrievedContext,
)
from app.retrieval import build_context_text, retrieve_context
from config import INTERACTION_MODEL, db_engine, openai_client
from prompts.interaction import (
    UNIFIED_INTERACTION_JSON_SCHEMA,
    UNIFIED_INTERACTION_SYSTEM_PROMPT,
)


MAX_STORED_TURNS = 20
MAX_PROMPT_TURNS = 4
INSUFFICIENT_CONTEXT_ANSWER = "I don't have enough information in the provided materials to answer that."
INSUFFICIENT_CONTEXT_FOLLOW_UP = (
    "Which part of your uploaded lecture or notes should we inspect next?"
)
STRONG_BRIDGE_RELATIONS = {"reinforces", "partial_overlap"}
MIN_STRONG_BRIDGE_CONFIDENCE = 0.72
BROAD_QUERY_PATTERNS = [
    "what is this about",
    "what is this video about",
    "what is this document about",
    "summarize",
    "summary",
    "overview",
    "big picture",
    "high level",
    "how do these sources relate",
    "how are these sources related",
]


def ensure_interaction_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS interaction_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        session_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(interaction_sessions)")).fetchall()
        }
        if "user_id" not in session_columns:
            connection.execute(
                text("ALTER TABLE interaction_sessions ADD COLUMN user_id TEXT DEFAULT 'legacy'")
            )
            connection.execute(
                text(
                    """
                    UPDATE interaction_sessions
                    SET user_id = 'legacy'
                    WHERE user_id IS NULL OR user_id = ''
                    """
                )
            )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_interaction_sessions_user_id
                ON interaction_sessions (user_id)
                """
            )
        )


def _normalize_source_ids(source_ids: list[int]) -> list[int]:
    normalized_ids = sorted({source_id for source_id in source_ids if source_id > 0})
    if not normalized_ids:
        raise ValueError("At least one valid source_id is required.")
    return normalized_ids


def _build_in_clause(values: list[int], prefix: str) -> tuple[str, dict[str, int]]:
    params: dict[str, int] = {}
    keys: list[str] = []

    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        keys.append(f":{key}")

    return ", ".join(keys), params


def _is_broad_query(query: str) -> bool:
    lowered_query = query.lower().strip()
    if not lowered_query:
        return False

    if any(pattern in lowered_query for pattern in BROAD_QUERY_PATTERNS):
        return True

    # "What is X about" / "Give me an overview" style questions are broad by intent.
    if re.search(r"\b(what|give)\b.*\b(about|overview|summary)\b", lowered_query):
        return True

    return False


def _scoped_session_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def _load_interaction_session(session_id: str, user_id: str) -> InteractionSessionState | None:
    scoped_session_id = _scoped_session_key(user_id, session_id)
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT state_json
                FROM interaction_sessions
                WHERE session_id = :session_id
                  AND user_id = :user_id
                """
            ),
            {"session_id": scoped_session_id, "user_id": user_id},
        ).mappings().first()

    if row is None:
        return None

    return InteractionSessionState.model_validate_json(str(row["state_json"]))


def _save_interaction_session(state: InteractionSessionState, user_id: str) -> None:
    scoped_session_id = _scoped_session_key(user_id, state.session_id)
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO interaction_sessions (
                    session_id,
                    user_id,
                    source_ids_json,
                    state_json
                ) VALUES (
                    :session_id,
                    :user_id,
                    :source_ids_json,
                    :state_json
                )
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    source_ids_json = excluded.source_ids_json,
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "session_id": scoped_session_id,
                "user_id": user_id,
                "source_ids_json": json.dumps(state.source_ids, ensure_ascii=True),
                "state_json": state.model_dump_json(),
            },
        )


def _build_learning_bridge(retrieved_context: RetrievedContext) -> str | None:
    term_labels: dict[str, str] = {}
    for hit in retrieved_context.hits:
        if hit.field_type == "term" and hit.concept_id and hit.text.strip():
            term_labels[hit.concept_id] = hit.text.strip()

    bridge_candidates = [
        edge
        for edge in retrieved_context.relationships
        if edge.relation_type in STRONG_BRIDGE_RELATIONS
        and edge.confidence >= MIN_STRONG_BRIDGE_CONFIDENCE
    ]
    if not bridge_candidates:
        return None

    best_edge = sorted(bridge_candidates, key=lambda edge: edge.confidence, reverse=True)[0]
    source_label = term_labels.get(best_edge.source_concept_id, best_edge.source_concept_id)
    target_label = term_labels.get(best_edge.target_concept_id, best_edge.target_concept_id)

    if best_edge.relation_type == "reinforces":
        relation_phrase = "reinforce each other"
    else:
        relation_phrase = "partially overlap and can be compared side by side"

    return (
        f"Learning bridge: {source_label} and {target_label} {relation_phrase}; "
        "connecting them helps transfer understanding across your materials."
    )


def _load_source_summaries(source_ids: list[int], user_id: str) -> list[dict[str, str | int]]:
    if not source_ids:
        return []

    in_clause, params = _build_in_clause(source_ids, "source_id")
    query = text(
        f"""
        SELECT source_id, source_type, summary_text
        FROM source_summaries
        WHERE user_id = :user_id
          AND source_id IN ({in_clause})
        ORDER BY source_type ASC, source_id ASC
        """
    )
    params["user_id"] = user_id

    with db_engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return [
        {
            "source_id": int(row["source_id"]),
            "source_type": str(row["source_type"]),
            "summary_text": str(row["summary_text"]),
        }
        for row in rows
        if str(row["summary_text"] or "").strip()
    ]


def _load_relationship_insights(source_ids: list[int], user_id: str) -> list[str]:
    if len(source_ids) < 2:
        return []

    in_clause, params = _build_in_clause(source_ids, "source_id")
    query = text(
        f"""
        SELECT
            source_concept_id,
            target_concept_id,
            relation_type,
            confidence,
            explanation
        FROM concept_relationship_edges
        WHERE user_id = :user_id
          AND source_source_id IN ({in_clause})
          AND target_source_id IN ({in_clause})
        ORDER BY confidence DESC, updated_at DESC, id DESC
        LIMIT 12
        """
    )
    params["user_id"] = user_id

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


def _run_unified_completion(
    query: str,
    retrieved_context: RetrievedContext,
    recent_turns: list[InteractionTurnRecord],
    user_id: str,
) -> AskModelOutput:
    context_text = build_context_text(retrieved_context)
    recent_turns_json = [turn.model_dump() for turn in recent_turns]

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Retrieved grounded context:\n{context_text}\n\n"
        f"Recent conversation turns JSON:\n{json.dumps(recent_turns_json, ensure_ascii=True)}"
    )

    completion = openai_client.chat.completions.create(
        model=INTERACTION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": UNIFIED_INTERACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": UNIFIED_INTERACTION_JSON_SCHEMA,
        },
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="interaction",
        model_name=INTERACTION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Interaction model returned an empty response.")

    return AskModelOutput.model_validate_json(content)


def _run_summary_completion(
    query: str,
    source_summaries: list[dict[str, str | int]],
    relationship_insights: list[str],
    recent_turns: list[InteractionTurnRecord],
    user_id: str,
) -> AskModelOutput:
    summary_lines = [
        (
            f"- Source {entry['source_id']} ({entry['source_type']}): "
            f"{str(entry['summary_text'])}"
        )
        for entry in source_summaries
    ]
    summary_text = "\n".join(summary_lines) if summary_lines else "- None"
    relationship_text = "\n".join(relationship_insights) if relationship_insights else "- None"
    recent_turns_json = [turn.model_dump() for turn in recent_turns]

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"High-level source summaries:\n{summary_text}\n\n"
        f"Cross-source relationship insights:\n{relationship_text}\n\n"
        f"Recent conversation turns JSON:\n{json.dumps(recent_turns_json, ensure_ascii=True)}"
    )

    completion = openai_client.chat.completions.create(
        model=INTERACTION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": UNIFIED_INTERACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": UNIFIED_INTERACTION_JSON_SCHEMA,
        },
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="interaction",
        model_name=INTERACTION_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Interaction model returned an empty summary response.")

    return AskModelOutput.model_validate_json(content)


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
    else:
        if sorted(state.source_ids) != normalized_source_ids:
            raise ValueError("session_id source_ids do not match the existing session context.")

    recent_turns = state.turns[-MAX_PROMPT_TURNS:]
    concept_ids: list[str] = []
    retrieved_context: RetrievedContext | None = None
    source_summaries = _load_source_summaries(normalized_source_ids, user_id)
    is_broad_query = _is_broad_query(normalized_query)

    if is_broad_query and source_summaries:
        relationship_insights = _load_relationship_insights(normalized_source_ids, user_id)
        model_output = _run_summary_completion(
            query=normalized_query,
            source_summaries=source_summaries,
            relationship_insights=relationship_insights,
            recent_turns=recent_turns,
            user_id=user_id,
        )
        answer = (model_output.answer or "").strip()
        follow_up_question = (model_output.follow_up_question or "").strip()
    else:
        try:
            retrieved_context = retrieve_context(
                query=normalized_query,
                source_ids=normalized_source_ids,
                user_id=user_id,
                top_k=top_k,
            )
        except ValueError:
            if source_summaries:
                relationship_insights = _load_relationship_insights(normalized_source_ids, user_id)
                model_output = _run_summary_completion(
                    query=normalized_query,
                    source_summaries=source_summaries,
                    relationship_insights=relationship_insights,
                    recent_turns=recent_turns,
                    user_id=user_id,
                )
                answer = (model_output.answer or "").strip()
                follow_up_question = (model_output.follow_up_question or "").strip()
            else:
                answer = INSUFFICIENT_CONTEXT_ANSWER
                follow_up_question = INSUFFICIENT_CONTEXT_FOLLOW_UP
        else:
            concept_ids = retrieved_context.concept_ids
            model_output = _run_unified_completion(
                query=normalized_query,
                retrieved_context=retrieved_context,
                recent_turns=recent_turns,
                user_id=user_id,
            )
            answer = (model_output.answer or "").strip()
            follow_up_question = (model_output.follow_up_question or "").strip()

    if not answer:
        answer = INSUFFICIENT_CONTEXT_ANSWER
    if not follow_up_question:
        follow_up_question = None

    if retrieved_context is not None and answer != INSUFFICIENT_CONTEXT_ANSWER:
        learning_bridge = _build_learning_bridge(retrieved_context)
        if learning_bridge and learning_bridge not in answer:
            answer = f"{answer}\n\n{learning_bridge}".strip()

    state.turns.append(
        InteractionTurnRecord(
            query=normalized_query,
            answer=answer,
            follow_up_question=follow_up_question,
        )
    )
    if len(state.turns) > MAX_STORED_TURNS:
        state.turns = state.turns[-MAX_STORED_TURNS:]

    _save_interaction_session(state, user_id)

    return AskResponse(
        session_id=state.session_id,
        answer=answer,
        follow_up_question=follow_up_question,
        concept_ids=concept_ids,
        model_name=INTERACTION_MODEL,
    )
