from __future__ import annotations

import unittest

from app.generation import (
    _compute_distributed_row_positions,
    _deduplicate_section_text,
    _normalize_continuous_prose,
    _should_use_model_progression_outline,
    _truncate_to_max_words,
)
from frontend.app import (
    _build_first_principles_evidence_summary,
    _build_quick_elaboration_prompt,
    _build_quick_quiz_prompt,
    _build_quiz_grading_prompt,
    _compose_assistant_chat_message,
    _normalize_continuous_text_for_display,
    _resolve_source_label_map,
    _should_show_cross_source_section,
)


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

    def test_deduplication_catches_semantic_overlap(self) -> None:
        current_text = (
            "Dopamine drives short-term reward seeking and narrows attention toward outcomes. "
            "By reinforcing process-level milestones, you can retrain attention toward long-term growth."
        )
        prior_texts = [
            "Short-term reward seeking is driven by dopamine and narrows attention toward outcomes."
        ]

        result = _deduplicate_section_text(current_text, prior_texts)
        self.assertNotIn("dopamine drives short-term reward seeking", result.lower())
        self.assertIn("process-level milestones", result.lower())

    def test_generation_prose_normalizer_removes_markdown_artifacts(self) -> None:
        raw = "###Mechanism\n- First causal step\n- Second causal step\n\n## Why this matters\nSignal###Leak\nA \u2014 B"
        normalized = _normalize_continuous_prose(raw)
        self.assertNotIn("###", normalized)
        self.assertNotIn("- First", normalized)
        self.assertNotIn("\u2014", normalized)
        self.assertIn("First causal step", normalized)
        self.assertIn("Second causal step", normalized)

    def test_word_truncation_respects_limit(self) -> None:
        raw = " ".join(f"word{index}" for index in range(1, 1201))
        truncated = _truncate_to_max_words(raw, max_words=900)
        self.assertLessEqual(len(truncated.split()), 900)


class ProgressionOutlineEfficiencyTests(unittest.TestCase):
    def test_short_sources_skip_model_outline_stage(self) -> None:
        should_use_model = _should_use_model_progression_outline(
            total_rows=6,
            grounding_chunks=["a", "b", "c"],
        )
        self.assertFalse(should_use_model)

    def test_long_sources_use_model_outline_stage(self) -> None:
        should_use_model = _should_use_model_progression_outline(
            total_rows=24,
            grounding_chunks=[str(index) for index in range(7)],
        )
        self.assertTrue(should_use_model)


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


class ChatBehaviorTests(unittest.TestCase):
    def test_quick_quiz_prompt_uses_recent_assistant_context(self) -> None:
        messages = [
            {"role": "assistant", "content": "Initial explanation."},
            {"role": "user", "content": "Tell me more."},
            {
                "role": "assistant",
                "content": "Deeper explanation.\n\n**Socratic next question:** What would you test next?",
            },
        ]

        prompt = _build_quick_quiz_prompt(messages)

        self.assertIn("Ask exactly one challenging question now", prompt)
        self.assertIn("Deeper explanation.", prompt)
        self.assertNotIn("Socratic next question", prompt)

    def test_quiz_grading_prompt_blocks_next_question(self) -> None:
        prompt = _build_quiz_grading_prompt(
            ask_messages=[{"role": "assistant", "content": "Question: Explain x."}],
            user_answer="My attempt is ...",
        )
        self.assertIn("Do not ask another quiz question", prompt)
        self.assertIn("My answer:", prompt)

    def test_quick_elaboration_prompt_requests_non_redundant_depth(self) -> None:
        prompt = _build_quick_elaboration_prompt(
            [{"role": "assistant", "content": "This is the previous explanation."}]
        )
        self.assertIn("Do not restate or paraphrase prior wording", prompt)
        self.assertIn("net-new causal steps", prompt)

    def test_assistant_message_can_suppress_follow_up_in_quiz_mode(self) -> None:
        composed = _compose_assistant_chat_message(
            "Here is your quiz question.",
            "What assumption is most fragile?",
            include_follow_up=False,
        )
        self.assertEqual(composed, "Here is your quiz question.")

    def test_assistant_message_includes_follow_up_outside_quiz_mode(self) -> None:
        composed = _compose_assistant_chat_message(
            "Explanation.",
            "What changes if constraints tighten?",
            include_follow_up=True,
        )
        self.assertIn("Socratic next question", composed)

    def test_frontend_continuous_text_normalizer_removes_headers(self) -> None:
        raw = "##Core idea\n###Why it works\n1. First point\n2. Second point\ninline###hash\nA \u2014 B"
        normalized = _normalize_continuous_text_for_display(raw)
        self.assertNotIn("##", normalized)
        self.assertNotIn("1.", normalized)
        self.assertNotIn("\u2014", normalized)
        self.assertNotIn("###", normalized)
        self.assertIn("Core idea", normalized)
        self.assertIn("Second point", normalized)

    def test_first_principles_evidence_summary_omits_source_labels(self) -> None:
        summary = _build_first_principles_evidence_summary(
            [
                ("Video: Test", "Mechanism starts with delayed feedback.", ["feedback"]),
                ("Document: Notes", "Constraint pressure changes the strategy.", ["constraint"]),
            ]
        )
        self.assertIn("Mechanism starts", summary)
        self.assertNotIn("Video:", summary)


class SourceLabelFormattingTests(unittest.TestCase):
    def test_source_label_map_uses_type_prefixed_labels(self) -> None:
        labels = _resolve_source_label_map(
            {"source_id": 7, "generated_title": "Loss Landscapes"},
            [
                {"source_id": 10, "generated_title": "Optimization Notes"},
                {"source_id": 11, "source_name": "Paper Appendix"},
            ],
        )

        self.assertEqual(labels[7], "Video: Loss Landscapes")
        self.assertEqual(labels[10], "Document: Optimization Notes")
        self.assertEqual(labels[11], "Document: Paper Appendix")


if __name__ == "__main__":
    unittest.main()
