from __future__ import annotations

import re
import threading
from typing import Any

from sqlalchemy import text

from app.models import RetrievedContext, RetrievalHit
import config
from config import chroma_client, db_engine, get_llm_client


# Collections are keyed by embedding provider+model: vectors from different
# models share neither dimensionality nor geometry, so a provider switch must
# land in its own collection instead of colliding with (or silently querying)
# stale vectors from the previous model. Chunks re-embed lazily per collection.
COLLECTION_PREFIX = "interaction_retrieval"
EMBEDDING_BATCH_SIZE = 64
QUERY_MULTIPLIER = 2
MAX_CONTEXT_CHARS = 4500
MAX_HIT_TEXT_CHARS = 280
# Collections use cosine distance (d = 1 - cos), so score = 1/(1+d). This floor
# keeps roughly the cosine cutoff (~0.55) the old l2 threshold encoded, loosened
# a little for local embedders whose similarity distributions run wider.
MIN_RAW_HIT_SCORE = 0.66
RAW_FIELD_TYPE = "raw_chunk"

# (user_id, source_id, collection_name) triples whose chunks are known embedded
# — skips the per-chat-turn Chroma existence sweep over every chunk id.
_fully_embedded: set[tuple[str, int, str]] = set()
_fully_embedded_lock = threading.Lock()


def _collection_name() -> str:
    slug = re.sub(
        r"[^a-zA-Z0-9._-]", "-", f"{config.embedding_provider()}-{config.retrieval_model()}"
    ).strip("-_.")
    return f"{COLLECTION_PREFIX}__{slug}"[:120]


def _prefixed_for_embedding(texts: list[str], *, kind: str) -> list[str]:
    """Apply the task prefixes local embedding models are trained with.

    nomic-embed-text and embeddinggemma measurably degrade without their
    document/query prefixes; OpenAI models take raw text."""
    model = config.retrieval_model()
    if "nomic-embed" in model:
        prefix = "search_document: " if kind == "document" else "search_query: "
        return [prefix + text_value for text_value in texts]
    if "embeddinggemma" in model:
        if kind == "document":
            return [f"title: none | text: {text_value}" for text_value in texts]
        return [f"task: search result | query: {text_value}" for text_value in texts]
    return texts


def _build_in_clause(values: list[int | str], prefix: str) -> tuple[str, dict[str, int | str]]:
    params: dict[str, int | str] = {}
    keys: list[str] = []
    for idx, value in enumerate(values):
        key = f"{prefix}_{idx}"
        params[key] = value
        keys.append(f":{key}")
    return ", ".join(keys), params


def _get_collection() -> Any:
    return chroma_client.get_or_create_collection(
        name=_collection_name(),
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": config.embedding_provider(),
            "embedding_model": config.retrieval_model(),
        },
    )


def _embed_texts(texts: list[str], *, kind: str = "document") -> list[list[float]]:
    if not texts:
        return []
    client = get_llm_client(config.embedding_provider())
    return client.embed(
        model=config.retrieval_model(),
        inputs=_prefixed_for_embedding(texts, kind=kind),
    )


def load_raw_chunks_for_sources(source_ids: list[int], user_id: str) -> list[dict[str, Any]]:
    normalized = sorted({sid for sid in source_ids if sid > 0})
    if not normalized:
        raise ValueError("At least one valid source_id is required for retrieval.")

    in_clause, params = _build_in_clause(normalized, "source_id")
    query = text(f"""
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
    """)
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
                float(row["timestamp_start"]) if row["timestamp_start"] is not None else None
            ),
            "timestamp_end": (
                float(row["timestamp_end"]) if row["timestamp_end"] is not None else None
            ),
            "page_number": (
                int(row["page_number"]) if row["page_number"] is not None else None
            ),
        }
        for row in rows
        if str(row["chunk_text"] or "").strip()
    ]


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

        documents.append({
            "id": f"raw:{chunk['source_id']}:{chunk['chunk_type']}:{chunk['chunk_index']}",
            "text": str(chunk["chunk_text"]),
            "metadata": metadata,
        })
    return documents


def upsert_retrieval_embeddings(
    source_ids: list[int],
    user_id: str,
    *,
    preloaded_raw_chunks: list[dict[str, Any]] | None = None,
) -> int:
    collection_name = _collection_name()
    normalized = sorted({sid for sid in source_ids if sid > 0})
    with _fully_embedded_lock:
        if all((user_id, sid, collection_name) in _fully_embedded for sid in normalized):
            return 0

    raw_chunks = preloaded_raw_chunks or load_raw_chunks_for_sources(source_ids, user_id)
    embedding_docs = build_raw_embedding_documents(raw_chunks, user_id)
    if not embedding_docs:
        return 0

    collection = _get_collection()
    doc_ids = [entry["id"] for entry in embedding_docs]
    existing = collection.get(ids=doc_ids, include=[])
    existing_ids = set(existing.get("ids", []))

    missing_docs = [entry for entry in embedding_docs if entry["id"] not in existing_ids]
    for start in range(0, len(missing_docs), EMBEDDING_BATCH_SIZE):
        batch = missing_docs[start : start + EMBEDDING_BATCH_SIZE]
        texts = [entry["text"] for entry in batch]
        embeddings = _embed_texts(texts, kind="document")
        collection.upsert(
            ids=[entry["id"] for entry in batch],
            documents=texts,
            metadatas=[entry["metadata"] for entry in batch],
            embeddings=embeddings,
        )

    with _fully_embedded_lock:
        for sid in normalized:
            _fully_embedded.add((user_id, sid, collection_name))
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
    parts = [f"{hit.source_type} #{hit.source_id}"]
    if hit.chunk_type == "transcript" and hit.timestamp_start is not None and hit.timestamp_end is not None:
        parts.append(f"t={hit.timestamp_start:.1f}-{hit.timestamp_end:.1f}s")
    if hit.chunk_type == "document" and hit.page_number is not None:
        parts.append(f"page={hit.page_number}")
    if hit.chunk_index is not None:
        parts.append(f"chunk={hit.chunk_index}")
    return " | ".join(parts)


def retrieve_context(
    query: str,
    source_ids: list[int],
    user_id: str,
    top_k: int = 8,
) -> RetrievedContext:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Retrieval query cannot be empty.")

    normalized_source_ids = sorted({sid for sid in source_ids if sid > 0})
    if not normalized_source_ids:
        raise ValueError("At least one valid source_id is required for retrieval.")

    collection_name = _collection_name()
    with _fully_embedded_lock:
        all_embedded = all(
            (user_id, sid, collection_name) in _fully_embedded
            for sid in normalized_source_ids
        )
    if not all_embedded:
        raw_chunks = load_raw_chunks_for_sources(normalized_source_ids, user_id)
        if not raw_chunks:
            raise ValueError("No retrieval context found for the provided query and sources.")
        upsert_retrieval_embeddings(
            normalized_source_ids, user_id, preloaded_raw_chunks=raw_chunks,
        )

    query_embedding = _embed_texts([normalized_query], kind="query")[0]
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

    candidate_hits: list[RetrievalHit] = []
    for _entry_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
        if metadata is None:
            continue
        field_type = str(metadata.get("field_type", "")).strip()
        if field_type != RAW_FIELD_TYPE:
            continue

        source_id = int(metadata.get("source_id", 0) or 0)
        if source_id <= 0:
            continue
        source_type = str(metadata.get("source_type", "")).strip()
        if source_type not in {"video", "document"}:
            continue

        chunk_type = str(metadata.get("chunk_type", "")).strip() or "document"
        if chunk_type not in {"transcript", "document"}:
            chunk_type = "document"

        score = _score_to_similarity(float(distance) if distance is not None else None)
        candidate_hits.append(RetrievalHit(
            source_id=source_id,
            source_type=source_type,
            text=str(document),
            score=score,
            chunk_type=chunk_type,
            chunk_index=int(metadata["chunk_index"]) if metadata.get("chunk_index") is not None else None,
            timestamp_start=float(metadata["timestamp_start"]) if metadata.get("timestamp_start") is not None else None,
            timestamp_end=float(metadata["timestamp_end"]) if metadata.get("timestamp_end") is not None else None,
            page_number=int(metadata["page_number"]) if metadata.get("page_number") is not None else None,
        ))

    strong_hits = [h for h in candidate_hits if h.score >= MIN_RAW_HIT_SCORE]
    if strong_hits:
        candidate_hits = strong_hits

    if not candidate_hits:
        raise ValueError("No retrieval context found for the provided query and sources.")

    best_by_key: dict[str, RetrievalHit] = {}
    for hit in candidate_hits:
        key = _raw_hit_key(hit)
        existing = best_by_key.get(key)
        if existing is None or hit.score > existing.score:
            best_by_key[key] = hit

    selected = sorted(best_by_key.values(), key=lambda h: h.score, reverse=True)[:top_k]

    return RetrievedContext(
        query=normalized_query,
        source_ids=normalized_source_ids,
        hits=selected,
        raw_hit_count=len(selected),
    )


def build_context_text(retrieved_context: RetrievedContext, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    lines: list[str] = []
    consumed = 0
    for hit in retrieved_context.hits:
        line = (
            f"- ({_format_raw_hit_provenance(hit)}, score={hit.score:.3f}) "
            f"{_truncate_hit_text(hit.text)}"
        )
        if consumed + len(line) > max_chars and lines:
            break
        lines.append(line)
        consumed += len(line) + 1

    if not lines:
        return "Supporting Material:\n- None"
    return "Supporting Material:\n" + "\n".join(lines)
