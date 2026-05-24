from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import text

from app.cost_logging import log_api_usage
from app.models import (
    ConceptExtractionPayload,
    LinkedConcept,
    LinkingEdgeRecord,
    RetrievedContext,
    RetrievalHit,
)
from config import RETRIEVAL_MODEL, chroma_client, db_engine, openai_client


COLLECTION_NAME = "interaction_retrieval_concepts"
EMBEDDING_BATCH_SIZE = 64
QUERY_MULTIPLIER = 2
RAW_PRIORITY_RATIO = 0.67
# Diagnostic-surfaced tune: chat interaction was using ~1.5k prompt tokens per
# query (the biggest line item by far) and the model only produced ~110
# completion tokens per call. Most of the prompt was retrieved context past
# what the model actually used. Trim aggressively — answers stay grounded but
# don't drown in tangential hits.
MAX_CONTEXT_CHARS = 4500
MAX_HIT_TEXT_CHARS = 280
MIN_CONCEPT_HIT_SCORE = 0.56
MIN_RAW_HIT_SCORE = 0.53

CONCEPT_FIELD_TYPES = {"term", "definition", "key_idea"}
RAW_FIELD_TYPE = "raw_chunk"


def _build_in_clause(values: list[int | str], prefix: str) -> tuple[str, dict[str, int | str]]:
    params: dict[str, int | str] = {}
    keys: list[str] = []

    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        keys.append(f":{key}")

    return ", ".join(keys), params


def _get_collection() -> Any:
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def _embed_texts(texts: list[str], user_id: str) -> list[list[float]]:
    if not texts:
        return []

    embedding_response = openai_client.embeddings.create(
        model=RETRIEVAL_MODEL,
        input=texts,
    )
    log_api_usage(
        response=embedding_response,
        user_id=user_id,
        call_stage="retrieval",
        model_name=RETRIEVAL_MODEL,
    )
    return [item.embedding for item in embedding_response.data]


def load_concepts_for_sources(source_ids: list[int], user_id: str) -> list[LinkedConcept]:
    normalized_source_ids = sorted({source_id for source_id in source_ids if source_id > 0})
    if not normalized_source_ids:
        raise ValueError("At least one valid source_id is required for retrieval.")

    in_clause, params = _build_in_clause(normalized_source_ids, "source_id")
    query = text(
        f"""
        SELECT source_id, source_type, extraction_json
        FROM concept_extractions
        WHERE source_id IN ({in_clause})
          AND user_id = :user_id
        """
    )
    params["user_id"] = user_id

    with db_engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    concepts: list[LinkedConcept] = []
    for row in rows:
        source_id = int(row["source_id"])
        source_type = str(row["source_type"])
        extraction_json = str(row["extraction_json"])

        payload = ConceptExtractionPayload.model_validate_json(extraction_json)
        for concept_index, concept in enumerate(payload.concepts, start=1):
            concepts.append(
                LinkedConcept(
                    concept_id=f"{source_id}:{concept_index}",
                    source_id=source_id,
                    source_type=source_type,
                    concept_index=concept_index,
                    term=concept.term,
                    definition=concept.definition,
                    key_ideas=concept.key_ideas,
                )
            )

    return concepts


def load_raw_chunks_for_sources(source_ids: list[int], user_id: str) -> list[dict[str, Any]]:
    normalized_source_ids = sorted({source_id for source_id in source_ids if source_id > 0})
    if not normalized_source_ids:
        raise ValueError("At least one valid source_id is required for retrieval.")

    in_clause, params = _build_in_clause(normalized_source_ids, "source_id")
    query = text(
        f"""
        SELECT
            source_text_chunks.source_id,
            sources.source_type,
            source_text_chunks.chunk_type,
            source_text_chunks.chunk_index,
            source_text_chunks.chunk_text,
            source_text_chunks.timestamp_start,
            source_text_chunks.timestamp_end,
            source_text_chunks.page_number
        FROM source_text_chunks
        JOIN sources ON sources.id = source_text_chunks.source_id
        WHERE source_text_chunks.source_id IN ({in_clause})
          AND source_text_chunks.user_id = :user_id
        ORDER BY
            source_text_chunks.source_id ASC,
            source_text_chunks.chunk_type ASC,
            source_text_chunks.chunk_index ASC
        """
    )
    params["user_id"] = user_id

    with db_engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return [
        {
            "source_id": int(row["source_id"]),
            "source_type": str(row["source_type"]),
            "chunk_type": str(row["chunk_type"]),
            "chunk_index": int(row["chunk_index"]),
            "chunk_text": str(row["chunk_text"]),
            "timestamp_start": (
                float(row["timestamp_start"])
                if row["timestamp_start"] is not None
                else None
            ),
            "timestamp_end": (
                float(row["timestamp_end"])
                if row["timestamp_end"] is not None
                else None
            ),
            "page_number": (
                int(row["page_number"])
                if row["page_number"] is not None
                else None
            ),
        }
        for row in rows
        if str(row["chunk_text"] or "").strip()
    ]


def load_relationships_for_concepts(concept_ids: list[str], user_id: str) -> list[LinkingEdgeRecord]:
    if not concept_ids:
        return []

    normalized_ids = sorted(set(concept_ids))
    in_clause, params = _build_in_clause(normalized_ids, "concept_id")
    query = text(
        f"""
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
        WHERE user_id = :user_id
          AND (source_concept_id IN ({in_clause}) OR target_concept_id IN ({in_clause}))
        ORDER BY confidence DESC, updated_at DESC, id DESC
        """
    )
    params["user_id"] = user_id

    with db_engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

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


def build_concept_embedding_documents(
    concepts: list[LinkedConcept],
    user_id: str,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for concept in concepts:
        documents.append(
            {
                "id": f"{concept.concept_id}:term",
                "text": concept.term,
                "metadata": {
                    "source_id": concept.source_id,
                    "source_type": concept.source_type,
                    "concept_id": concept.concept_id,
                    "user_id": user_id,
                    "field_type": "term",
                },
            }
        )
        documents.append(
            {
                "id": f"{concept.concept_id}:definition",
                "text": concept.definition,
                "metadata": {
                    "source_id": concept.source_id,
                    "source_type": concept.source_type,
                    "concept_id": concept.concept_id,
                    "user_id": user_id,
                    "field_type": "definition",
                },
            }
        )

        for key_idea_index, key_idea in enumerate(concept.key_ideas, start=1):
            documents.append(
                {
                    "id": f"{concept.concept_id}:key_idea:{key_idea_index}",
                    "text": key_idea,
                    "metadata": {
                        "source_id": concept.source_id,
                        "source_type": concept.source_type,
                        "concept_id": concept.concept_id,
                        "user_id": user_id,
                        "field_type": "key_idea",
                    },
                }
            )

    return documents


def build_raw_embedding_documents(
    raw_chunks: list[dict[str, Any]],
    user_id: str,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for chunk in raw_chunks:
        metadata: dict[str, Any] = {
            "source_id": int(chunk["source_id"]),
            "source_type": str(chunk["source_type"]),
            "user_id": user_id,
            "field_type": RAW_FIELD_TYPE,
            "chunk_type": str(chunk["chunk_type"]),
            "chunk_index": int(chunk["chunk_index"]),
        }
        if chunk.get("page_number") is not None:
            metadata["page_number"] = int(chunk["page_number"])
        if chunk.get("timestamp_start") is not None:
            metadata["timestamp_start"] = float(chunk["timestamp_start"])
        if chunk.get("timestamp_end") is not None:
            metadata["timestamp_end"] = float(chunk["timestamp_end"])

        documents.append(
            {
                "id": (
                    f"raw:{chunk['source_id']}:{chunk['chunk_type']}:{chunk['chunk_index']}"
                ),
                "text": str(chunk["chunk_text"]),
                "metadata": metadata,
            }
        )

    return documents


def build_embedding_documents(
    concepts: list[LinkedConcept],
    raw_chunks: list[dict[str, Any]],
    user_id: str,
) -> list[dict[str, Any]]:
    return build_raw_embedding_documents(raw_chunks, user_id) + build_concept_embedding_documents(
        concepts,
        user_id,
    )


def upsert_retrieval_embeddings(
    source_ids: list[int],
    user_id: str,
    *,
    preloaded_concepts: list[LinkedConcept] | None = None,
    preloaded_raw_chunks: list[dict[str, Any]] | None = None,
) -> int:
    concepts = preloaded_concepts or load_concepts_for_sources(source_ids, user_id)
    raw_chunks = preloaded_raw_chunks or load_raw_chunks_for_sources(source_ids, user_id)
    embedding_docs = build_embedding_documents(concepts, raw_chunks, user_id)
    if not embedding_docs:
        return 0

    collection = _get_collection()
    doc_ids = [entry["id"] for entry in embedding_docs]
    existing = collection.get(ids=doc_ids, include=[])
    existing_ids = set(existing.get("ids", []))

    missing_docs = [entry for entry in embedding_docs if entry["id"] not in existing_ids]
    if not missing_docs:
        return 0

    for start in range(0, len(missing_docs), EMBEDDING_BATCH_SIZE):
        batch = missing_docs[start : start + EMBEDDING_BATCH_SIZE]
        texts = [entry["text"] for entry in batch]
        embeddings = _embed_texts(texts, user_id)

        collection.upsert(
            ids=[entry["id"] for entry in batch],
            documents=texts,
            metadatas=[entry["metadata"] for entry in batch],
            embeddings=embeddings,
        )

    return len(missing_docs)


def _raw_hit_key(hit: RetrievalHit) -> str:
    return f"{hit.source_id}:{hit.chunk_type}:{hit.chunk_index}"


def _score_to_similarity(distance: float | None) -> float:
    return 1.0 / (1.0 + max(float(distance or 0.0), 0.0))


def _truncate_hit_text(text_value: str, max_chars: int = MAX_HIT_TEXT_CHARS) -> str:
    normalized = " ".join(text_value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _format_raw_hit_provenance(hit: RetrievalHit) -> str:
    metadata_fragments = [f"{hit.source_type} #{hit.source_id}"]
    if hit.chunk_type == "transcript" and hit.timestamp_start is not None and hit.timestamp_end is not None:
        metadata_fragments.append(f"t={hit.timestamp_start:.1f}-{hit.timestamp_end:.1f}s")
    if hit.chunk_type == "document" and hit.page_number is not None:
        metadata_fragments.append(f"page={hit.page_number}")
    if hit.chunk_index is not None:
        metadata_fragments.append(f"chunk={hit.chunk_index}")
    return " | ".join(metadata_fragments)


def _build_limited_section(
    title: str,
    lines: list[str],
    *,
    remaining_chars: int,
) -> tuple[str, int]:
    if not lines or remaining_chars <= 0:
        return "", remaining_chars

    section_lines = [title]
    consumed = len(title) + 1

    for line in lines:
        line_with_break = line + "\n"
        if consumed + len(line_with_break) > remaining_chars:
            break
        section_lines.append(line)
        consumed += len(line_with_break)

    if len(section_lines) == 1:
        return "", remaining_chars

    section_text = "\n".join(section_lines).strip()
    return section_text, remaining_chars - len(section_text) - 2


def retrieve_context(query: str, source_ids: list[int], user_id: str, top_k: int = 8) -> RetrievedContext:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Retrieval query cannot be empty.")

    normalized_source_ids = sorted({source_id for source_id in source_ids if source_id > 0})
    if not normalized_source_ids:
        raise ValueError("At least one valid source_id is required for retrieval.")

    concepts = load_concepts_for_sources(normalized_source_ids, user_id)
    raw_chunks = load_raw_chunks_for_sources(normalized_source_ids, user_id)

    if not concepts and not raw_chunks:
        raise ValueError("No retrieval context found for the provided query and sources.")

    concept_map = {concept.concept_id: concept for concept in concepts}

    upsert_retrieval_embeddings(
        normalized_source_ids,
        user_id,
        preloaded_concepts=concepts,
        preloaded_raw_chunks=raw_chunks,
    )

    query_embedding = _embed_texts([normalized_query], user_id)[0]
    collection = _get_collection()
    raw_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k * QUERY_MULTIPLIER, top_k),
        where={
            "$and": [
                {"source_id": {"$in": normalized_source_ids}},
                {"user_id": user_id},
            ]
        },
        include=["metadatas", "documents", "distances"],
    )

    ids = raw_result.get("ids", [[]])[0]
    metadatas = raw_result.get("metadatas", [[]])[0]
    documents = raw_result.get("documents", [[]])[0]
    distances = raw_result.get("distances", [[]])[0]

    candidate_concept_hits: list[RetrievalHit] = []
    candidate_raw_hits: list[RetrievalHit] = []

    for entry_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
        if metadata is None:
            continue

        field_type = str(metadata.get("field_type", "")).strip()
        score = _score_to_similarity(float(distance) if distance is not None else None)

        if field_type in CONCEPT_FIELD_TYPES:
            concept_id = str(metadata.get("concept_id", "")).strip()
            if concept_id not in concept_map:
                continue

            concept = concept_map[concept_id]
            candidate_concept_hits.append(
                RetrievalHit(
                    concept_id=concept_id,
                    source_id=concept.source_id,
                    source_type=concept.source_type,
                    field_type=field_type,
                    text=str(document),
                    score=score,
                )
            )
            continue

        if field_type == RAW_FIELD_TYPE:
            source_id = int(metadata.get("source_id", 0) or 0)
            if source_id <= 0:
                continue

            source_type = str(metadata.get("source_type", "")).strip()
            if source_type not in {"video", "document"}:
                continue

            chunk_type = str(metadata.get("chunk_type", "")).strip() or "document"
            if chunk_type not in {"transcript", "document"}:
                chunk_type = "document"

            chunk_index = metadata.get("chunk_index")
            chunk_index_int = int(chunk_index) if chunk_index is not None else None

            page_number = metadata.get("page_number")
            page_number_int = int(page_number) if page_number is not None else None

            timestamp_start = metadata.get("timestamp_start")
            timestamp_end = metadata.get("timestamp_end")

            candidate_raw_hits.append(
                RetrievalHit(
                    concept_id=None,
                    source_id=source_id,
                    source_type=source_type,
                    field_type=RAW_FIELD_TYPE,
                    text=str(document),
                    score=score,
                    chunk_type=chunk_type,
                    chunk_index=chunk_index_int,
                    timestamp_start=(
                        float(timestamp_start)
                        if timestamp_start is not None
                        else None
                    ),
                    timestamp_end=(
                        float(timestamp_end)
                        if timestamp_end is not None
                        else None
                    ),
                    page_number=page_number_int,
                )
            )

    strong_concept_hits = [hit for hit in candidate_concept_hits if hit.score >= MIN_CONCEPT_HIT_SCORE]
    strong_raw_hits = [hit for hit in candidate_raw_hits if hit.score >= MIN_RAW_HIT_SCORE]
    if strong_concept_hits or strong_raw_hits:
        candidate_concept_hits = strong_concept_hits
        candidate_raw_hits = strong_raw_hits

    if not candidate_concept_hits and not candidate_raw_hits:
        raise ValueError("No retrieval context found for the provided query and sources.")

    concept_best_hit: dict[str, RetrievalHit] = {}
    for hit in candidate_concept_hits:
        existing_hit = concept_best_hit.get(hit.concept_id)
        if existing_hit is None or hit.score > existing_hit.score:
            concept_best_hit[hit.concept_id] = hit

    raw_best_hit: dict[str, RetrievalHit] = {}
    for hit in candidate_raw_hits:
        key = _raw_hit_key(hit)
        existing_hit = raw_best_hit.get(key)
        if existing_hit is None or hit.score > existing_hit.score:
            raw_best_hit[key] = hit

    ranked_concept_hits = sorted(
        concept_best_hit.values(),
        key=lambda hit: hit.score,
        reverse=True,
    )
    ranked_raw_hits = sorted(
        raw_best_hit.values(),
        key=lambda hit: hit.score,
        reverse=True,
    )

    raw_target = min(
        len(ranked_raw_hits),
        max(1 if ranked_raw_hits else 0, int(top_k * RAW_PRIORITY_RATIO)),
    )
    concept_target = min(len(ranked_concept_hits), max(top_k - raw_target, 0))

    selected_hits: list[RetrievalHit] = []
    selected_hits.extend(ranked_raw_hits[:raw_target])
    selected_hits.extend(ranked_concept_hits[:concept_target])

    selected_raw_keys = {_raw_hit_key(hit) for hit in selected_hits if hit.field_type == RAW_FIELD_TYPE}
    selected_concept_keys = {
        hit.concept_id
        for hit in selected_hits
        if hit.field_type in CONCEPT_FIELD_TYPES and hit.concept_id
    }

    remaining_raw_hits = [
        hit for hit in ranked_raw_hits if _raw_hit_key(hit) not in selected_raw_keys
    ]
    remaining_concept_hits = [
        hit
        for hit in ranked_concept_hits
        if hit.concept_id and hit.concept_id not in selected_concept_keys
    ]
    combined_remaining_hits = sorted(
        [*remaining_raw_hits, *remaining_concept_hits],
        key=lambda hit: hit.score,
        reverse=True,
    )

    for hit in combined_remaining_hits:
        if len(selected_hits) >= top_k:
            break
        if hit.field_type == RAW_FIELD_TYPE and _raw_hit_key(hit) in selected_raw_keys:
            continue
        if hit.field_type in CONCEPT_FIELD_TYPES and hit.concept_id in selected_concept_keys:
            continue

        selected_hits.append(hit)
        if hit.field_type == RAW_FIELD_TYPE:
            selected_raw_keys.add(_raw_hit_key(hit))
        elif hit.concept_id:
            selected_concept_keys.add(hit.concept_id)

    selected_hits = sorted(selected_hits, key=lambda hit: hit.score, reverse=True)

    selected_concept_ids: list[str] = []
    for hit in selected_hits:
        if hit.field_type in CONCEPT_FIELD_TYPES and hit.concept_id and hit.concept_id not in selected_concept_ids:
            selected_concept_ids.append(hit.concept_id)

    relationships = load_relationships_for_concepts(selected_concept_ids, user_id)

    # Keep relationship list focused on currently selected concepts.
    concept_id_set = set(selected_concept_ids)
    filtered_relationships = [
        relationship
        for relationship in relationships
        if relationship.source_concept_id in concept_id_set
        or relationship.target_concept_id in concept_id_set
    ]

    return RetrievedContext(
        query=normalized_query,
        source_ids=normalized_source_ids,
        concept_ids=selected_concept_ids,
        hits=selected_hits,
        relationships=filtered_relationships,
        raw_hit_count=len([hit for hit in selected_hits if hit.field_type == RAW_FIELD_TYPE]),
        concept_hit_count=len([hit for hit in selected_hits if hit.field_type in CONCEPT_FIELD_TYPES]),
    )


def build_context_text(retrieved_context: RetrievedContext, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    raw_hits = [hit for hit in retrieved_context.hits if hit.field_type == RAW_FIELD_TYPE]
    concept_hits = [
        hit for hit in retrieved_context.hits if hit.field_type in CONCEPT_FIELD_TYPES and hit.concept_id
    ]

    remaining_chars = max_chars

    raw_lines = [
        (
            f"- ({_format_raw_hit_provenance(hit)}, score={hit.score:.3f}) "
            f"{_truncate_hit_text(hit.text)}"
        )
        for hit in raw_hits
    ]
    raw_section, remaining_chars = _build_limited_section(
        "Supporting Raw Material:",
        raw_lines,
        remaining_chars=remaining_chars,
    )

    concepts_by_id: dict[str, list[str]] = defaultdict(list)
    concept_label_map: dict[str, str] = {}
    for hit in concept_hits:
        concepts_by_id[str(hit.concept_id)].append(
            f"- ({hit.field_type}, score={hit.score:.3f}) {_truncate_hit_text(hit.text)}"
        )
        if hit.field_type == "term" and hit.concept_id and hit.concept_id not in concept_label_map:
            concept_label_map[str(hit.concept_id)] = _truncate_hit_text(hit.text, 120)

    concept_blocks: list[str] = []
    for concept_id in retrieved_context.concept_ids:
        concept_lines = concepts_by_id.get(concept_id, [])
        if concept_lines:
            concept_label = concept_label_map.get(concept_id, "related concept")
            concept_blocks.append(f"Concept: {concept_label}\n" + "\n".join(concept_lines))

    concept_section, remaining_chars = _build_limited_section(
        "Structured Concepts:",
        concept_blocks,
        remaining_chars=remaining_chars,
    )

    relationship_lines = [
        (
            f"- {concept_label_map.get(edge.source_concept_id, 'related concept')} -> "
            f"{concept_label_map.get(edge.target_concept_id, 'related concept')}: "
            f"{edge.relation_type} (confidence={edge.confidence:.2f}) | {_truncate_hit_text(edge.explanation, 280)}"
        )
        for edge in retrieved_context.relationships
    ]
    relationship_section, _ = _build_limited_section(
        "Linked Relationships:",
        relationship_lines or ["- None"],
        remaining_chars=remaining_chars,
    )

    sections = [section for section in [raw_section, concept_section, relationship_section] if section]
    if not sections:
        return "Supporting Raw Material:\n- None\n\nStructured Concepts:\n- None\n\nLinked Relationships:\n- None"
    return "\n\n".join(sections)
