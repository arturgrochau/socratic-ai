from __future__ import annotations

import unittest

from app.generation import _compute_distributed_row_positions, _deduplicate_section_text
from frontend.app import _should_show_cross_source_section


class DistributedGroundingSelectionTests(unittest.TestCase):
    def test_distributed_positions_cover_early_mid_late_segments(self) -> None:
        positions = _compute_distributed_row_positions(total_rows=30, max_chunks=10)

        self.assertEqual(len(positions), 10)
        self.assertTrue(any(position < 10 for position in positions))
        self.assertTrue(any(10 <= position < 20 for position in positions))
        self.assertTrue(any(position >= 20 for position in positions))

    def test_distributed_positions_use_all_rows_when_short_source(self) -> None:
        positions = _compute_distributed_row_positions(total_rows=5, max_chunks=10)
        self.assertEqual(positions, [0, 1, 2, 3, 4])


class DeduplicationTests(unittest.TestCase):
    def test_deduplication_removes_redundant_sentences(self) -> None:
        current_text = (
            "Dopamine can bias attention toward immediate rewards. "
            "Dopamine can bias attention toward immediate rewards. "
            "You can retrain attention by reinforcing process-level milestones."
        )
        prior_texts = ["Dopamine can bias attention toward immediate rewards."]

        result = _deduplicate_section_text(current_text, prior_texts)
        self.assertIn("retrain attention", result)
        self.assertNotIn("Dopamine can bias attention toward immediate rewards.", result)

    def test_deduplication_uses_safety_fallback_when_too_much_removed(self) -> None:
        current_text = (
            "The mechanism starts with an external trigger. "
            "The mechanism then moves into reinforcement loops. "
            "Practical design must respect those loops."
        )
        prior_texts = [
            "The mechanism starts with an external trigger.",
            "The mechanism then moves into reinforcement loops.",
        ]

        result = _deduplicate_section_text(current_text, prior_texts)
        self.assertEqual(result, current_text)


class CrossSourceVisibilityTests(unittest.TestCase):
    def test_single_source_never_shows_cross_source_section(self) -> None:
        should_show = _should_show_cross_source_section(
            generation_result={"source_ids": [12]},
            insights_payload={"synthesis_text": "Synthesis content exists"},
            quiz_payload={"questions": [{"question": "x"}]},
        )
        self.assertFalse(should_show)

    def test_multi_source_shows_cross_source_section_when_content_exists(self) -> None:
        should_show = _should_show_cross_source_section(
            generation_result={"source_ids": [12, 14]},
            insights_payload={"comparative_analysis": "Meaningful contrast"},
            quiz_payload={"questions": []},
        )
        self.assertTrue(should_show)


if __name__ == "__main__":
    unittest.main()
