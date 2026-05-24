from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterable

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.models import (
    ConceptExtractionPayload,
    CrossReferenceResult,
    LinkedConcept,
    LinkingEdgeRecord,
)
from config import CACHE_PROCESSED_SOURCES, LINKING_MODEL, db_engine, openai_client
from prompts.cross_reference import (
    CROSS_REFERENCE_BATCH_JSON_SCHEMA,
    CROSS_REFERENCE_BATCH_SYSTEM_PROMPT,
    CROSS_REFERENCE_JSON_SCHEMA,
    CROSS_REFERENCE_SYSTEM_PROMPT,
)


MAX_TARGETS_PER_SOURCE_CONCEPT = 3
# Session-wide ceiling; the practical per-pair budget is also scaled by source
# count in link_source_pair() so multi-source sessions don't starve.
MAX_COMPARISONS_PER_RUN = 15
LINKING_BATCH_SIZE = 5
MIN_TOPIC_TOKEN_LENGTH = 3
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def ensure_linking_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS concept_relationship_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_concept_id TEXT NOT NULL,
                    target_concept_id TEXT NOT NULL,
                    source_source_id INTEGER NOT NULL,
                    target_source_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL CHECK (
                        relation_type IN (
                            'reinforces',
                            'new_info',
                            'contradiction',
                            'partial_overlap'
                        )
                    ),
                    explanation TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (source_concept_id, target_concept_id)
                )
                """
            )
        )
        edge_columns = {
            row[1] for row in connection.execute(text("PRAGMA table_info(concept_relationship_edges)")).fetchall()
        }
        if "user_id" not in edge_columns:
            connection.execute(
                text("ALTER TABLE concept_relationship_edges ADD COLUMN user_id TEXT DEFAULT 'legacy'")
            )
            connection.execute(
                text(
                    """
                    UPDATE concept_relationship_edges
                    SET user_id = (
                        SELECT sources.user_id
                        FROM sources
                        WHERE sources.id = concept_relationship_edges.source_source_id
                    )
                    WHERE user_id IS NULL
                    """
                )
            )
            connection.execute(
                text(
                    """
                    UPDATE concept_relationship_edges
                    SET user_id = 'legacy'
                    WHERE user_id IS NULL OR user_id = ''
                    """
                )
            )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_source_concept
                ON concept_relationship_edges (source_concept_id)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_target_concept
                ON concept_relationship_edges (target_concept_id)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_user_id
                ON concept_relationship_edges (user_id)
                """
            )
        )


def load_extracted_concepts(source_id: int, user_id: str) -> tuple[str, ConceptExtractionPayload]:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT source_type, extraction_json
                FROM concept_extractions
                WHERE source_id = :source_id AND user_id = :user_id
                """
            ),
            {"source_id": source_id, "user_id": user_id},
        ).mappings().first()

    if row is None:
        raise ValueError(f"No extracted concepts found for source {source_id}.")

    source_type = str(row["source_type"])
    if source_type not in {"video", "document"}:
        raise ValueError(f"Unsupported source type in concept extraction: {source_type}")

    extraction_json = row["extraction_json"]
    if not extraction_json:
        raise ValueError(f"Extraction JSON is empty for source {source_id}.")

    payload = ConceptExtractionPayload.model_validate_json(str(extraction_json))
    return source_type, payload


def build_linkable_concepts(
    source_id: int,
    source_type: str,
    payload: ConceptExtractionPayload,
) -> list[LinkedConcept]:
    concepts: list[LinkedConcept] = []
    for index, concept in enumerate(payload.concepts, start=1):
        concepts.append(
            LinkedConcept(
                concept_id=f"{source_id}:{index}",
                source_id=source_id,
                source_type=source_type,
                concept_index=index,
                term=concept.term,
                definition=concept.definition,
                key_ideas=concept.key_ideas,
            )
        )
    return concepts


def _tokenize(values: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in TOKEN_PATTERN.findall(value.lower()):
            if len(token) >= MIN_TOPIC_TOKEN_LENGTH:
                tokens.add(token)
    return tokens


def tokenize_topic_terms(concept: LinkedConcept) -> set[str]:
    return _tokenize([concept.term, *concept.key_ideas])


def build_candidate_pairs(
    source_concepts: list[LinkedConcept],
    target_concepts: list[LinkedConcept],
    max_targets_per_source_concept: int = MAX_TARGETS_PER_SOURCE_CONCEPT,
    max_comparisons: int = MAX_COMPARISONS_PER_RUN,
) -> list[tuple[LinkedConcept, LinkedConcept]]:
    if not source_concepts or not target_concepts:
        return []

    token_to_target_indices: dict[str, set[int]] = defaultdict(set)
    for target_index, target_concept in enumerate(target_concepts):
        for token in tokenize_topic_terms(target_concept):
            token_to_target_indices[token].add(target_index)

    candidate_pairs: list[tuple[LinkedConcept, LinkedConcept]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for source_concept in source_concepts:
        source_tokens = tokenize_topic_terms(source_concept)
        target_scores: dict[int, int] = defaultdict(int)

        for token in source_tokens:
            for target_index in token_to_target_indices.get(token, set()):
                target_scores[target_index] += 1

        if not target_scores:
            continue

        ranked_targets = sorted(
            target_scores.items(),
            key=lambda item: (-item[1], target_concepts[item[0]].concept_index),
        )

        selected_count = 0
        for target_index, _ in ranked_targets:
            target_concept = target_concepts[target_index]
            pair_key = (source_concept.concept_id, target_concept.concept_id)
            if pair_key in seen_pairs:
                continue

            candidate_pairs.append((source_concept, target_concept))
            seen_pairs.add(pair_key)
            selected_count += 1

            if selected_count >= max_targets_per_source_concept:
                break
            if len(candidate_pairs) >= max_comparisons:
                return candidate_pairs

    return candidate_pairs


def _parse_concept_id(concept_id: str) -> tuple[int, int]:
    parts = concept_id.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid concept_id format: {concept_id}")

    try:
        source_id = int(parts[0])
        concept_index = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid concept_id format: {concept_id}") from exc

    if source_id <= 0 or concept_index <= 0:
        raise ValueError(f"Invalid concept_id values: {concept_id}")

    return source_id, concept_index


def get_concept_by_id(concept_id: str, user_id: str) -> LinkedConcept:
    source_id, concept_index = _parse_concept_id(concept_id)
    source_type, payload = load_extracted_concepts(source_id, user_id)
    concepts = build_linkable_concepts(source_id, source_type, payload)

    if concept_index > len(concepts):
        raise ValueError(f"Concept index out of range for concept_id {concept_id}")

    return concepts[concept_index - 1]


def compare_concept_pair_batch(
    pairs: list[tuple[LinkedConcept, LinkedConcept]],
    user_id: str,
    batch_size: int = LINKING_BATCH_SIZE,
) -> list[CrossReferenceResult]:
    """Compare multiple concept pairs in batched API calls. Returns results in input order."""
    if not pairs:
        return []

    results: dict[int, CrossReferenceResult] = {}

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        batch_payload = [
            {
                "pair_index": batch_start + i,
                "source_concept": source.model_dump(),
                "target_concept": target.model_dump(),
            }
            for i, (source, target) in enumerate(batch)
        ]

        completion = openai_client.chat.completions.create(
            model=LINKING_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": CROSS_REFERENCE_BATCH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Classify relationships for these concept pairs:\n\n"
                        f"{json.dumps(batch_payload, ensure_ascii=True)}"
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": CROSS_REFERENCE_BATCH_JSON_SCHEMA,
            },
        )
        log_api_usage(
            response=completion,
            user_id=user_id,
            call_stage="linking",
            model_name=LINKING_MODEL,
        )

        content = completion.choices[0].message.content
        if not content:
            continue

        parsed = json.loads(content)
        for item in parsed.get("results", []):
            try:
                idx = int(item["pair_index"])
                results[idx] = CrossReferenceResult(
                    relation_type=item["relation_type"],
                    explanation=str(item.get("explanation", "")),
                    confidence=float(item.get("confidence", 0.5)),
                )
            except (KeyError, ValueError):
                continue

    return [
        results[i]
        for i in range(len(pairs))
        if i in results
    ]


def compare_concept_pair(
    source_concept: LinkedConcept,
    target_concept: LinkedConcept,
    user_id: str,
) -> CrossReferenceResult:
    comparison_payload = {
        "source_concept": source_concept.model_dump(),
        "target_concept": target_concept.model_dump(),
    }

    completion = openai_client.chat.completions.create(
        model=LINKING_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": CROSS_REFERENCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Compare these two concepts and classify their relationship.\n\n"
                    f"{json.dumps(comparison_payload, ensure_ascii=True)}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": CROSS_REFERENCE_JSON_SCHEMA,
        },
    )
    log_api_usage(
        response=completion,
        user_id=user_id,
        call_stage="linking",
        model_name=LINKING_MODEL,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty linking content.")

    return CrossReferenceResult.model_validate_json(content)


def store_relationship_edge(
    source_concept: LinkedConcept,
    target_concept: LinkedConcept,
    result: CrossReferenceResult,
    user_id: str,
) -> LinkingEdgeRecord:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO concept_relationship_edges (
                    user_id,
                    source_concept_id,
                    target_concept_id,
                    source_source_id,
                    target_source_id,
                    relation_type,
                    explanation,
                    confidence,
                    model_name
                ) VALUES (
                    :user_id,
                    :source_concept_id,
                    :target_concept_id,
                    :source_source_id,
                    :target_source_id,
                    :relation_type,
                    :explanation,
                    :confidence,
                    :model_name
                )
                ON CONFLICT(source_concept_id, target_concept_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    relation_type = excluded.relation_type,
                    explanation = excluded.explanation,
                    confidence = excluded.confidence,
                    model_name = excluded.model_name,
                    updated_at = CURRENT_TIMESTAMP
                """
            ),
            {
                "user_id": user_id,
                "source_concept_id": source_concept.concept_id,
                "target_concept_id": target_concept.concept_id,
                "source_source_id": source_concept.source_id,
                "target_source_id": target_concept.source_id,
                "relation_type": result.relation_type,
                "explanation": result.explanation,
                "confidence": result.confidence,
                "model_name": LINKING_MODEL,
            },
        )

    return LinkingEdgeRecord(
        source_concept_id=source_concept.concept_id,
        target_concept_id=target_concept.concept_id,
        source_source_id=source_concept.source_id,
        target_source_id=target_concept.source_id,
        relation_type=result.relation_type,
        explanation=result.explanation,
        confidence=result.confidence,
        model_name=LINKING_MODEL,
    )


def load_existing_relationship_edge(
    source_concept_id: str,
    target_concept_id: str,
    user_id: str,
) -> LinkingEdgeRecord | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    source_concept_id,
                    target_concept_id,
                    source_source_id,
                    target_source_id,
                    relation_type,
                    explanation,
                    confidence,
                    model_name
                FROM concept_relationship_edges
                WHERE source_concept_id = :source_concept_id
                  AND target_concept_id = :target_concept_id
                  AND user_id = :user_id
                LIMIT 1
                """
            ),
            {
                "source_concept_id": source_concept_id,
                "target_concept_id": target_concept_id,
                "user_id": user_id,
            },
        ).mappings().first()

    if row is None:
        return None

    return LinkingEdgeRecord(
        source_concept_id=str(row["source_concept_id"]),
        target_concept_id=str(row["target_concept_id"]),
        source_source_id=int(row["source_source_id"]),
        target_source_id=int(row["target_source_id"]),
        relation_type=str(row["relation_type"]),
        explanation=str(row["explanation"]),
        confidence=float(row["confidence"]),
        model_name=str(row["model_name"]),
    )


def link_single_pair(
    source_concept_id: str,
    target_concept_id: str,
    user_id: str,
) -> LinkingEdgeRecord:
    ensure_linking_tables()
    if CACHE_PROCESSED_SOURCES:
        existing_edge = load_existing_relationship_edge(source_concept_id, target_concept_id, user_id)
        if existing_edge is not None:
            return existing_edge

    source_concept = get_concept_by_id(source_concept_id, user_id)
    target_concept = get_concept_by_id(target_concept_id, user_id)

    result = compare_concept_pair(source_concept, target_concept, user_id)
    return store_relationship_edge(source_concept, target_concept, result, user_id)


def link_source_pair(
    video_source_id: int,
    document_source_id: int,
    user_id: str,
    *,
    max_comparisons: int | None = None,
) -> dict[str, int]:
    ensure_linking_tables()

    video_source_type, video_payload = load_extracted_concepts(video_source_id, user_id)
    document_source_type, document_payload = load_extracted_concepts(document_source_id, user_id)

    if video_source_type != "video":
        raise ValueError(f"Source {video_source_id} is not a video concept source.")
    if document_source_type != "document":
        raise ValueError(f"Source {document_source_id} is not a document concept source.")

    video_concepts = build_linkable_concepts(video_source_id, video_source_type, video_payload)
    document_concepts = build_linkable_concepts(
        document_source_id,
        document_source_type,
        document_payload,
    )

    candidate_pairs = build_candidate_pairs(
        video_concepts,
        document_concepts,
        max_comparisons=max_comparisons or MAX_COMPARISONS_PER_RUN,
    )

    # Filter out already-cached pairs before batching
    pairs_to_compare: list[tuple[LinkedConcept, LinkedConcept]] = []
    if CACHE_PROCESSED_SOURCES:
        for source_concept, target_concept in candidate_pairs:
            existing_edge = load_existing_relationship_edge(
                source_concept.concept_id,
                target_concept.concept_id,
                user_id,
            )
            if existing_edge is None:
                pairs_to_compare.append((source_concept, target_concept))
    else:
        pairs_to_compare = list(candidate_pairs)

    results = compare_concept_pair_batch(pairs_to_compare, user_id)

    stored_edges = 0
    for (source_concept, target_concept), result in zip(pairs_to_compare, results):
        store_relationship_edge(source_concept, target_concept, result, user_id)
        stored_edges += 1

    return {
        "video_source_id": video_source_id,
        "document_source_id": document_source_id,
        "candidate_pairs": len(candidate_pairs),
        "stored_edges": stored_edges,
    }


def get_relationships_for_source_concept(
    source_concept_id: str,
    user_id: str,
) -> list[LinkingEdgeRecord]:
    with db_engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT
                    source_concept_id,
                    target_concept_id,
                    source_source_id,
                    target_source_id,
                    relation_type,
                    explanation,
                    confidence,
                    model_name
                FROM concept_relationship_edges
                WHERE source_concept_id = :source_concept_id
                  AND user_id = :user_id
                ORDER BY confidence DESC, updated_at DESC, id DESC
                """
            ),
            {"source_concept_id": source_concept_id, "user_id": user_id},
        ).mappings().all()

    return [
        LinkingEdgeRecord(
            source_concept_id=str(row["source_concept_id"]),
            target_concept_id=str(row["target_concept_id"]),
            source_source_id=int(row["source_source_id"]),
            target_source_id=int(row["target_source_id"]),
            relation_type=str(row["relation_type"]),
            explanation=str(row["explanation"]),
            confidence=float(row["confidence"]),
            model_name=str(row["model_name"]),
        )
        for row in rows
    ]
