"""Unit tests for the code-level section validator and the chat depth heuristic
(both replaced LLM calls), plus the map-reduce ledger merge determinism."""
from __future__ import annotations

import unittest

from app.ledger import KnowledgeUnit, append_units, next_unit_id, parse_units
from app.section_validator import validate_arc
from prompts.sections import PER_SOURCE_SECTIONS


def _clean_structured() -> dict:
    return {
        "summary": "Feedback loops drive system behavior. This matters because stability depends on their balance.",
        "deep_dive": (
            "Delays inside a loop produce oscillation when they exceed the response time. "
            "Thresholds trigger regime shifts that look discontinuous from outside. "
            "Coupling between loops dominates large-system behavior."
        ),
        "under_surface": "It is unresolved how loop coupling interacts with measurement noise in practice.",
        "key_terms": [
            {"term": "feedback", "layman": "an effect changing its own cause", "technical": "output routed to input"},
        ],
        "reflection_points": [
            {"question": "Why does delay create oscillation?", "explanation": "trap", "depth_level": "advanced"},
        ],
    }


class ValidateArcTests(unittest.TestCase):
    def test_clean_arc_passes(self) -> None:
        self.assertEqual(validate_arc(PER_SOURCE_SECTIONS, _clean_structured()), [])

    def test_over_sentence_budget_flagged(self) -> None:
        structured = _clean_structured()
        # Summary budget is max 5 sentences (+2 grace) -> 9 sentences must flag.
        structured["summary"] = " ".join(f"Sentence number {i} is here." for i in range(9))
        violations = validate_arc(PER_SOURCE_SECTIONS, structured)
        self.assertTrue(any("Core Thesis" in v and "over budget" in v for v in violations))

    def test_over_item_budget_flagged(self) -> None:
        structured = _clean_structured()
        structured["key_terms"] = [
            {"term": f"t{i}", "layman": "x", "technical": "y"} for i in range(12)
        ]
        violations = validate_arc(PER_SOURCE_SECTIONS, structured)
        self.assertTrue(any("Key Concepts" in v for v in violations))

    def test_restatement_flagged(self) -> None:
        structured = _clean_structured()
        # under_surface (check_restatement=True) copying the deep dive must flag.
        structured["under_surface"] = structured["deep_dive"]
        violations = validate_arc(PER_SOURCE_SECTIONS, structured)
        self.assertTrue(any("Open Questions" in v and "restates" in v for v in violations))

    def test_empty_sections_skipped(self) -> None:
        self.assertEqual(validate_arc(PER_SOURCE_SECTIONS, {"summary": ""}), [])


class HeuristicDepthTests(unittest.TestCase):
    def _tier(self, query: str, turns: int = 0) -> str:
        from app.interaction import _heuristic_depth
        from app.models import InteractionTurnRecord

        recent = [InteractionTurnRecord(query="q", answer="a") for _ in range(turns)]
        return _heuristic_depth(query, recent)

    def test_lookup(self) -> None:
        self.assertEqual(self._tier("What is a knowledge ledger?"), "lookup")
        self.assertEqual(self._tier("define entropy"), "lookup")

    def test_explain_default(self) -> None:
        self.assertEqual(self._tier("how does the chunking pipeline handle PDFs"), "explain")

    def test_analyze(self) -> None:
        self.assertEqual(self._tier("compare the two sources on feedback delay"), "analyze")
        self.assertEqual(self._tier("why does coupling dominate at scale"), "analyze")

    def test_deepen_needs_prior_turns(self) -> None:
        self.assertEqual(self._tier("go deeper on that", turns=1), "deepen")
        # Without prior turns, "deeper" can't refer to a prior answer.
        self.assertNotEqual(self._tier("go deeper on that", turns=0), "deepen")


class MapReduceMergeTests(unittest.TestCase):
    """The reduce step (parse_units + append_units in window order) must be
    deterministic and drop cross-window duplicates."""

    def test_merge_dedups_and_keeps_order(self) -> None:
        window_payloads = [
            [{"type": "foundational", "claim": "Feedback loops drive all system behavior.", "evidence": ""}],
            [
                # Near-duplicate of window 1's claim -> dropped by overlap dedup.
                {"type": "foundational", "claim": "Feedback loops drive system behavior overall.", "evidence": ""},
                {"type": "mechanism", "claim": "Delays inside loops produce oscillation.", "evidence": ""},
            ],
        ]
        ledger: list[KnowledgeUnit] = []
        for payload in window_payloads:
            new_units = parse_units(payload, source_id=1, start_index=next_unit_id(ledger))
            append_units(ledger, new_units)

        claims = [u.claim for u in ledger]
        self.assertEqual(len(ledger), 2, f"expected dedup to 2 units, got {claims}")
        self.assertTrue(claims[0].startswith("Feedback loops drive all"))
        self.assertEqual(ledger[0].id, "u1")
        self.assertEqual(ledger[1].type, "mechanism")


if __name__ == "__main__":
    unittest.main()
