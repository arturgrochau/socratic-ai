from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.json_reliability import safe_json_loads
from app.session_logger import SessionLogger
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
MAX_GENERATED_CONTEXT_CHARS = 2200
# Kept for back-compat in _is_insufficient_answer detection. Live answers no
# longer return this string verbatim; the relaxed system prompt produces a
# Socratic redirect instead.
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
DEEPENING_QUERY_PATTERNS = [
    "elaborate further",
    "go deeper",
    "expand on this",
    "explain more",
    "more detail",
    "deepen this",
    "tell me more",
]
REDUNDANCY_SIMILARITY_THRESHOLD = 0.88
REDUNDANCY_TOKEN_OVERLAP_THRESHOLD = 0.62
REDUNDANCY_MIN_CHAR_RATIO = 0.45
DEEPENING_REDUNDANCY_SIMILARITY_THRESHOLD = 0.74
DEEPENING_REDUNDANCY_TOKEN_OVERLAP_THRESHOLD = 0.50
DEEPENING_REDUNDANCY_MIN_CHAR_RATIO = 0.25


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


def _truncate_text(text_value: str, max_chars: int) -> str:
    normalized = " ".join(text_value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _split_sentences(text_value: str) -> list[str]:
    normalized = " ".join(str(text_value or "").split()).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[A-Za-z0-9']+", left.lower()) if len(token) >= 4}
    right_tokens = {token for token in re.findall(r"[A-Za-z0-9']+", right.lower()) if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(min(len(left_tokens), len(right_tokens)))


def _sanitize_continuous_text(text_value: str) -> str:
    lines = str(text_value or "").splitlines()
    cleaned_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        line = line.replace("\u2014", ", ").replace("\u2013", ", ")
        line = re.sub(r"\s-\s", ", ", line)
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^#{1,6}(?=\S)", "", line).strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"^>\s+", "", line)
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

    return "\n\n".join(part for part in paragraphs if part).strip()


def _strip_redundant_sentences(
    current_text: str,
    reference_text: str,
    *,
    similarity_threshold: float = REDUNDANCY_SIMILARITY_THRESHOLD,
    token_overlap_threshold: float = REDUNDANCY_TOKEN_OVERLAP_THRESHOLD,
    min_char_ratio: float = REDUNDANCY_MIN_CHAR_RATIO,
    allow_shorter_output: bool = False,
) -> str:
    current_sentences = _split_sentences(current_text)
    if len(current_sentences) < 2:
        return current_text

    reference_sentences = _split_sentences(reference_text)
    if not reference_sentences:
        return current_text

    kept_sentences: list[str] = []
    for sentence in current_sentences:
        sentence_norm = sentence.lower()
        duplicate = False
        for reference in [*reference_sentences, *kept_sentences]:
            reference_norm = reference.lower()
            if sentence_norm == reference_norm:
                duplicate = True
                break
            if SequenceMatcher(None, sentence_norm, reference_norm).ratio() >= similarity_threshold:
                duplicate = True
                break
            if _token_overlap_ratio(sentence_norm, reference_norm) >= token_overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept_sentences.append(sentence)

    if not kept_sentences:
        return current_text

    deduped = " ".join(kept_sentences).strip()
    if len(deduped) < int(len(current_text) * min_char_ratio):
        if allow_shorter_output:
            return deduped
        return current_text
    return deduped


def _has_meaningful_text(value: str) -> bool:
    return bool(value and value.strip() and value.strip() != "[None available]")


def _is_insufficient_answer(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized == INSUFFICIENT_CONTEXT_ANSWER.lower():
        return True
    return "don't have enough information" in normalized and "provided materials" in normalized


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


def _should_include_generated_context(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return False
    if _is_broad_query(lowered):
        return True
    return any(
        token in lowered
        for token in ["intersection", "cross-source", "synthesis", "compare", "connection", "relationship"]
    )


def _is_deepening_query(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return False
    return any(pattern in lowered for pattern in DEEPENING_QUERY_PATTERNS)


def _extract_answer_claims(answer_text: str, *, max_claims: int = 3) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(answer_text):
        normalized = " ".join(sentence.split()).strip()
        if len(normalized) < 36:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        claims.append(normalized)
        if len(claims) >= max_claims:
            break
    return claims


def _wants_long_form(query: str) -> bool:
    lowered = query.lower().strip()
    if _is_deepening_query(lowered):
        return True
    return any(token in lowered for token in ["how", "why", "mechanism", "under the hood", "first principles"]) and len(lowered) > 24


def _expand_followup_query(query: str, recent_turns: list[InteractionTurnRecord]) -> str:
    if not _is_deepening_query(query) or not recent_turns:
        return query

    last_turn = recent_turns[-1]
    prior_claims = _extract_answer_claims(last_turn.answer, max_claims=3)
    claims_block = "\n".join(f"- {claim}" for claim in prior_claims) or "- [no clear prior claims]"
    return (
        f"{query}. Focus on the previous discussion topic. "
        "Do not restate the previous assistant answer with paraphrasing. "
        "Add one or two new dimensions chosen from hidden assumption, constraint, failure mode, or tradeoff. "
        "Do not repeat the listed prior claims unless needed for one short context sentence. "
        f"Prior claims to avoid repeating:\n{claims_block}\n"
        f"Previous user question: {last_turn.query}. "
        f"Previous assistant answer (truncated): {_truncate_text(last_turn.answer, 820)}"
    ).strip()


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

    if source_label == best_edge.source_concept_id or target_label == best_edge.target_concept_id:
        return None

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
            f"- {row['relation_type']} (confidence={float(row['confidence']):.2f}) | "
            f"{str(row['explanation'])}"
        )
        for row in rows
    ]


def _load_generated_learning_context(source_ids: list[int], user_id: str) -> str:
    if not source_ids:
        return ""

    in_clause, params = _build_in_clause(source_ids, "source_id")
    params["user_id"] = user_id

    source_rows = []
    with db_engine.connect() as connection:
        source_rows = connection.execute(
            text(
                f"""
                SELECT source_id, source_type, source_name, generated_title, summary_text, deep_dive_text, key_terms_json
                FROM source_learning_sections
                WHERE user_id = :user_id
                  AND source_id IN ({in_clause})
                ORDER BY source_type ASC, source_id ASC
                """
            ),
            params,
        ).mappings().all()

    source_blocks: list[str] = []
    for row in source_rows:
        key_terms = safe_json_loads(row.get("key_terms_json"), default=[])
        key_terms_text = ", ".join(
            str(term).strip() for term in key_terms[:6] if str(term).strip()
        )
        source_name = str(row.get("source_name") or row.get("generated_title") or "source").strip()
        source_type = str(row.get("source_type") or "source").strip()
        source_blocks.append(
            (
                f"{source_type.title()} [{source_name}]\n"
                f"Title: {str(row.get('generated_title') or '').strip()}\n"
                f"Summary: {_truncate_text(str(row.get('summary_text') or ''), 280)}\n"
                f"Deep dive: {_truncate_text(str(row.get('deep_dive_text') or ''), 320)}\n"
                f"Key terms: {key_terms_text or 'n/a'}"
            )
        )

    generated_sections = "\n\n".join(source_blocks)

    combined_row = None
    with db_engine.connect() as connection:
        combined_candidates = connection.execute(
            text(
                """
                SELECT
                    video_source_id,
                    document_source_ids_json,
                    intersections_json,
                    layman_bridge,
                    synthesis_text,
                    comparative_analysis_text,
                    application_scenarios_json,
                    quiz_json
                FROM combined_learning_sections
                WHERE user_id = :user_id
                ORDER BY updated_at DESC, id DESC
                LIMIT 20
                """
            ),
            {"user_id": user_id},
        ).mappings().all()

    source_id_set = set(source_ids)
    best_score = -1
    for candidate in combined_candidates:
        try:
            candidate_doc_ids = {
                int(value)
                for value in safe_json_loads(candidate.get("document_source_ids_json"), default=[])
            }
        except Exception:
            candidate_doc_ids = set()

        candidate_video_id = int(candidate.get("video_source_id") or 0)
        score = 0
        if candidate_video_id in source_id_set:
            score += 2
        score += len(candidate_doc_ids.intersection(source_id_set))
        if score > best_score:
            best_score = score
            combined_row = candidate

    combined_block = ""
    if combined_row is not None and best_score > 0:
        intersections = safe_json_loads(combined_row.get("intersections_json"), default=[])
        intersection_titles = []
        for entry in intersections[:4]:
            title = str(entry.get("intersection_title", "")).strip() if isinstance(entry, dict) else ""
            if title:
                intersection_titles.append(title)

        quiz_payload = safe_json_loads(combined_row.get("quiz_json"), default={})
        study_advice = _truncate_text(str(quiz_payload.get("study_advice") or ""), 320)
        comparative_analysis = _truncate_text(
            str(combined_row.get("comparative_analysis_text") or ""),
            420,
        )

        scenario_payload = safe_json_loads(combined_row.get("application_scenarios_json"), default=[])
        scenario_titles: list[str] = []
        for entry in scenario_payload[:3]:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("scenario_title") or "").strip()
            if title:
                scenario_titles.append(title)

        combined_block = (
            "Combined synthesis context\n"
            f"Intersections: {', '.join(intersection_titles) if intersection_titles else 'n/a'}\n"
            f"Layman bridge: {_truncate_text(str(combined_row.get('layman_bridge') or ''), 320)}\n"
            f"Synthesis: {_truncate_text(str(combined_row.get('synthesis_text') or ''), 420)}\n"
            f"Comparative analysis: {comparative_analysis or 'n/a'}\n"
            f"Application scenarios: {', '.join(scenario_titles) if scenario_titles else 'n/a'}\n"
            f"Study advice: {study_advice or 'n/a'}"
        )

    full_context = "\n\n".join(part for part in [generated_sections, combined_block] if part).strip()
    if not full_context:
        return ""
    return _truncate_text(full_context, MAX_GENERATED_CONTEXT_CHARS)


def _build_best_effort_answer(
    *,
    query: str,
    source_summaries: list[dict[str, str | int]],
    relationship_insights: list[str],
    generated_learning_context: str,
    long_form: bool,
) -> str:
    summary_fragments = [
        _truncate_text(str(entry.get("summary_text") or ""), 220)
        for entry in source_summaries[:2]
        if _has_meaningful_text(str(entry.get("summary_text") or ""))
    ]

    first_principles_bits: list[str] = []
    if summary_fragments:
        first_principles_bits.append(" ".join(summary_fragments))
    if _has_meaningful_text(generated_learning_context):
        first_principles_bits.append(
            _truncate_text(generated_learning_context, 420)
        )

    bridge_note = ""
    if relationship_insights:
        bridge_note = _truncate_text(relationship_insights[0].lstrip("- "), 220)
        if "unrelated" in bridge_note.lower():
            bridge_note = ""

    merged_signal = _truncate_text(" ".join(first_principles_bits), 900 if long_form else 520)
    direct_response = (
        f"For your question about '{query}', "
        f"{merged_signal if merged_signal else 'the available material has limited direct signal, but there are still useful grounded cues.'}"
    )

    reflective_tail = "Notice where the mechanism stays the same across contexts and where constraints force a different interpretation."
    if bridge_note:
        reflective_tail = (
            f"A key cross-source bridge is: {bridge_note}. "
            "Track how this bridge holds up as examples become more specific and assumptions shift."
        )

    if long_form:
        return _sanitize_continuous_text(
            (
            f"{direct_response}\n\n"
            "To go deeper, trace the causal chain step by step, identify which assumptions are fixed, "
            "and check what breaks when those assumptions are relaxed.\n\n"
            f"{reflective_tail}"
            ).strip()
        )

    return _sanitize_continuous_text(f"{direct_response}\n\n{reflective_tail}".strip())


def _run_unified_completion(
    query: str,
    retrieved_context: RetrievedContext,
    recent_turns: list[InteractionTurnRecord],
    generated_learning_context: str,
    long_form: bool,
    user_id: str,
) -> AskModelOutput:
    context_text = build_context_text(retrieved_context)
    recent_turns_json = [turn.model_dump() for turn in recent_turns]

    user_prompt = (
        "Priority instruction: answer the user query directly in the opening sentence, then give only essential grounded support.\n"
        + (
            "Provide a deeper and longer answer (approximately 3-5 substantial paragraphs) with mechanism-level explanation in plain language.\n\n"
            if long_form
            else "\n"
        )
        +
        f"User query:\n{query}\n\n"
        f"Generated learning context:\n{generated_learning_context or '[None available]'}\n\n"
        f"Retrieved grounded context:\n{context_text}\n\n"
        f"Recent conversation turns JSON:\n{json.dumps(recent_turns_json, ensure_ascii=True)}"
    )

    completion = openai_client.chat.completions.create(
        model=INTERACTION_MODEL,
        temperature=0.3,
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
    generated_learning_context: str,
    long_form: bool,
    user_id: str,
) -> AskModelOutput:
    summary_lines = [
        (
            f"- {str(entry['source_type']).title()} material: "
            f"{str(entry['summary_text'])}"
        )
        for entry in source_summaries
    ]
    summary_text = "\n".join(summary_lines) if summary_lines else "- None"
    relationship_text = "\n".join(relationship_insights) if relationship_insights else "- None"
    recent_turns_json = [turn.model_dump() for turn in recent_turns]

    user_prompt = (
        "Priority instruction: answer the user query directly in the opening sentence, then give only essential grounded support.\n"
        + (
            "Provide a deeper and longer answer (approximately 3-5 substantial paragraphs) with mechanism-level explanation in plain language.\n\n"
            if long_form
            else "\n"
        )
        +
        f"User query:\n{query}\n\n"
        f"Generated learning context:\n{generated_learning_context or '[None available]'}\n\n"
        f"High-level source summaries:\n{summary_text}\n\n"
        f"Cross-source relationship insights:\n{relationship_text}\n\n"
        f"Recent conversation turns JSON:\n{json.dumps(recent_turns_json, ensure_ascii=True)}"
    )

    completion = openai_client.chat.completions.create(
        model=INTERACTION_MODEL,
        temperature=0.3,
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
    expanded_query = _expand_followup_query(normalized_query, recent_turns)
    long_form = _wants_long_form(normalized_query)
    concept_ids: list[str] = []
    retrieved_context: RetrievedContext | None = None
    source_summaries = _load_source_summaries(normalized_source_ids, user_id)
    generated_learning_context = ""
    if _should_include_generated_context(expanded_query):
        generated_learning_context = _load_generated_learning_context(normalized_source_ids, user_id)
    relationship_insights = _load_relationship_insights(normalized_source_ids, user_id)
    is_broad_query = _is_broad_query(expanded_query)
    effective_top_k = max(4, min(top_k, 8 if long_form else 6))

    has_context_signal = bool(source_summaries) or _has_meaningful_text(generated_learning_context)

    # Route once. Broad queries skip retrieval entirely; specific queries try
    # retrieval, and only fall back to summary when retrieval has no usable
    # signal at all. No retry-on-insufficient-answer (it doubled the call cost
    # for ambiguous queries with no quality gain).
    use_summary_path = is_broad_query and has_context_signal
    if not use_summary_path:
        try:
            retrieved_context = retrieve_context(
                query=expanded_query,
                source_ids=normalized_source_ids,
                user_id=user_id,
                top_k=effective_top_k,
            )
        except ValueError:
            retrieved_context = None
            use_summary_path = has_context_signal

    if use_summary_path:
        model_output = _run_summary_completion(
            query=expanded_query,
            source_summaries=source_summaries,
            relationship_insights=relationship_insights,
            recent_turns=recent_turns,
            generated_learning_context=generated_learning_context,
            long_form=long_form,
            user_id=user_id,
        )
        answer = (model_output.answer or "").strip()
        follow_up_question = (model_output.follow_up_question or "").strip()
    elif retrieved_context is not None:
        concept_ids = retrieved_context.concept_ids
        model_output = _run_unified_completion(
            query=expanded_query,
            retrieved_context=retrieved_context,
            recent_turns=recent_turns,
            generated_learning_context=generated_learning_context,
            long_form=long_form,
            user_id=user_id,
        )
        answer = (model_output.answer or "").strip()
        follow_up_question = (model_output.follow_up_question or "").strip()
    else:
        # No retrieval, no summaries — the relaxed system prompt would have
        # produced a Socratic redirect if it had any context; without any, we
        # surface a short, honest pointer instead of a robotic refusal.
        answer = (
            "Your uploaded materials do not appear to cover this directly. "
            "Tell me which source you want me to dig into, or rephrase the question against something specific from the lecture or document."
        )
        follow_up_question = "What is the most concrete passage in your materials that hints at the answer you're looking for?"

    if not answer and has_context_signal:
        # Last-resort synthesis from summaries — only when the model returned
        # truly empty content. Replaces the old dead-end refusal string.
        answer = _build_best_effort_answer(
            query=expanded_query,
            source_summaries=source_summaries,
            relationship_insights=relationship_insights,
            generated_learning_context=generated_learning_context,
            long_form=long_form,
        )

    answer = _sanitize_continuous_text(answer)

    if not follow_up_question:
        follow_up_question = None
    else:
        follow_up_question = _sanitize_continuous_text(follow_up_question)

    if _is_deepening_query(normalized_query) and recent_turns and answer:
        previous_answer = _sanitize_continuous_text(recent_turns[-1].answer)
        answer = _sanitize_continuous_text(
            _strip_redundant_sentences(
                answer,
                previous_answer,
                similarity_threshold=DEEPENING_REDUNDANCY_SIMILARITY_THRESHOLD,
                token_overlap_threshold=DEEPENING_REDUNDANCY_TOKEN_OVERLAP_THRESHOLD,
                min_char_ratio=DEEPENING_REDUNDANCY_MIN_CHAR_RATIO,
                allow_shorter_output=True,
            )
        )

    if retrieved_context is not None and not _is_insufficient_answer(answer):
        learning_bridge = _sanitize_continuous_text(_build_learning_bridge(retrieved_context) or "")
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

    # Per-query telemetry — appended to a chat-specific JSONL so the diagnostic
    # harness can score "feel": route taken, retrieval signal strength, whether
    # the dead-end refusal pattern leaked through, answer length distribution.
    route = "broad" if use_summary_path and is_broad_query else (
        "summary_fallback" if use_summary_path else (
            "grounded" if retrieved_context is not None else "no_context"
        )
    )
    top_retrieval_score = 0.0
    if retrieved_context is not None and retrieved_context.hits:
        top_retrieval_score = round(
            max(hit.score for hit in retrieved_context.hits), 4
        )
    SessionLogger(f"chat-{session_id}", user_id=user_id).record(
        "chat_query",
        {
            "query_length": len(normalized_query),
            "answer_length": len(answer),
            "route": route,
            "top_retrieval_score": top_retrieval_score,
            "n_retrieval_hits": (
                len(retrieved_context.hits) if retrieved_context else 0
            ),
            "had_follow_up": bool(follow_up_question),
            "contains_refusal_pattern": _is_insufficient_answer(answer),
            "long_form": long_form,
            "is_deepening": _is_deepening_query(normalized_query),
            "n_turns": len(state.turns),
        },
    )

    return AskResponse(
        session_id=state.session_id,
        answer=answer,
        follow_up_question=follow_up_question,
        concept_ids=concept_ids,
        model_name=INTERACTION_MODEL,
    )
