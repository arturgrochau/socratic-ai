"""
Per-run in-memory similarity index for anti-redundancy in generation.

Scope: one `generate_tailored_learning` invocation. Discarded after.

Why not ChromaDB? Each generation run produces ~5-15 paragraphs per source.
A pure-numpy cosine index sized to that is faster, has zero persistence
side effects, and removes the cleanup question.

Usage from generation:

    rag = MiniRag(client=get_llm_client(), model=RETRIEVAL_MODEL)
    rag.add_paragraphs(section_idx=0, title="Why X matters",
                       paragraphs=["...para 1...", "...para 2..."])
    hits = rag.find_similar("query text", top_k=3, min_score=0.78)
    for h in hits:
        # h.text, h.section_idx, h.title, h.score
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _EmbedClient(Protocol):
    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]: ...


@dataclass
class MiniRagHit:
    text: str
    section_idx: int
    title: str
    score: float


class MiniRag:
    """Cosine-similarity index over generated paragraphs."""

    def __init__(self, client: _EmbedClient, model: str) -> None:
        self._client = client
        self._model = model
        self._records: list[tuple[np.ndarray, str, int, str]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_paragraphs(
        self,
        *,
        section_idx: int,
        title: str,
        paragraphs: list[str],
    ) -> None:
        clean = [p.strip() for p in paragraphs if p and p.strip()]
        if not clean:
            return
        vectors = self._client.embed(model=self._model, inputs=clean)
        for text, vec in zip(clean, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm == 0.0:
                continue
            unit = arr / norm
            self._records.append((unit, text, section_idx, title))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_similar(
        self,
        text: str,
        *,
        top_k: int = 3,
        min_score: float = 0.78,
    ) -> list[MiniRagHit]:
        if not self._records or not text.strip():
            return []
        vectors = self._client.embed(model=self._model, inputs=[text.strip()])
        query = np.asarray(vectors[0], dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query_unit = query / norm

        scored: list[MiniRagHit] = []
        for vec, ptext, idx, title in self._records:
            score = float(np.dot(vec, query_unit))
            if score >= min_score:
                scored.append(MiniRagHit(text=ptext, section_idx=idx, title=title, score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)


class _StaticEmbedClient:
    """Test helper. Maps known input strings to fixed embedding vectors."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in inputs:
            if text not in self._vectors:
                raise KeyError(f"_StaticEmbedClient missing vector for {text!r}")
            out.append(self._vectors[text])
        return out
