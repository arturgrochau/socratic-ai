"""Unit tests for app.mini_rag using deterministic synthetic vectors."""
from __future__ import annotations

import unittest

from app.mini_rag import MiniRag, _StaticEmbedClient


class MiniRagTests(unittest.TestCase):
    def test_empty_index_returns_no_hits(self) -> None:
        client = _StaticEmbedClient({"query": [1.0, 0.0]})
        rag = MiniRag(client=client, model="test-embed")
        self.assertEqual(rag.find_similar("query"), [])

    def test_finds_identical_paragraph(self) -> None:
        vectors = {
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "alpha (query)": [1.0, 0.0, 0.0],
        }
        client = _StaticEmbedClient(vectors)
        rag = MiniRag(client=client, model="test-embed")
        rag.add_paragraphs(section_idx=0, title="A", paragraphs=["alpha", "beta"])
        hits = rag.find_similar("alpha (query)", top_k=2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].text, "alpha")
        self.assertEqual(hits[0].section_idx, 0)
        self.assertEqual(hits[0].title, "A")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)

    def test_orthogonal_vectors_below_threshold(self) -> None:
        vectors = {
            "alpha": [1.0, 0.0],
            "orthogonal-query": [0.0, 1.0],
        }
        client = _StaticEmbedClient(vectors)
        rag = MiniRag(client=client, model="test-embed")
        rag.add_paragraphs(section_idx=0, title="A", paragraphs=["alpha"])
        hits = rag.find_similar("orthogonal-query", min_score=0.5)
        self.assertEqual(hits, [])

    def test_ranking_by_similarity(self) -> None:
        # Three paragraphs with varying similarity to the query.
        vectors = {
            "near": [0.95, 0.31, 0.0],
            "medium": [0.7, 0.7, 0.0],
            "far": [0.1, 0.99, 0.0],
            "query": [1.0, 0.0, 0.0],
        }
        client = _StaticEmbedClient(vectors)
        rag = MiniRag(client=client, model="test-embed")
        rag.add_paragraphs(
            section_idx=0, title="S",
            paragraphs=["near", "medium", "far"],
        )
        hits = rag.find_similar("query", top_k=2, min_score=0.5)
        self.assertEqual([h.text for h in hits], ["near", "medium"])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_section_attribution_preserved(self) -> None:
        vectors = {
            "p1": [1.0, 0.0],
            "p2": [0.99, 0.14],
            "p3": [0.97, 0.24],
            "q": [1.0, 0.0],
        }
        client = _StaticEmbedClient(vectors)
        rag = MiniRag(client=client, model="test-embed")
        rag.add_paragraphs(section_idx=0, title="First", paragraphs=["p1"])
        rag.add_paragraphs(section_idx=1, title="Second", paragraphs=["p2"])
        rag.add_paragraphs(section_idx=2, title="Third", paragraphs=["p3"])
        hits = rag.find_similar("q", top_k=3, min_score=0.5)
        attribution = {(h.section_idx, h.title) for h in hits}
        self.assertEqual(
            attribution,
            {(0, "First"), (1, "Second"), (2, "Third")},
        )

    def test_blank_paragraphs_ignored(self) -> None:
        client = _StaticEmbedClient({"real": [1.0, 0.0]})
        rag = MiniRag(client=client, model="test-embed")
        rag.add_paragraphs(section_idx=0, title="A", paragraphs=["", "  ", "real"])
        self.assertEqual(len(rag), 1)


if __name__ == "__main__":
    unittest.main()
