from __future__ import annotations

import unittest

from app.generation import (
    _assert_interaction_stage_progression,
    _assert_progressive_cross_structure,
    _apply_source_hard_cuts,
    _build_progressive_cross_stage_texts,
    _build_source_intent_map,
    _classify_interaction_types,
    _classify_deep_dive_operation,
    _compute_distributed_row_positions,
    _deduplicate_section_text,
    _enforce_section_novelty,
    _extract_claims,
    _extract_variable_families,
    _filter_deep_dive_sentences,
    _filter_under_surface_sentences,
    _normalize_continuous_prose,
    _pairwise_claim_overlap_ratio,
    _section_redundancy_ratio,
    _section_has_required_variables,
    _source_restatement_ratio,
    _tighten_progressive_cross_stage_texts,
    _tighten_stage_text,
    _should_use_model_progression_outline,
    _truncate_to_max_words,
    _pairwise_overlap_ratio,
)
from app.interaction import _expand_followup_query, _strip_redundant_sentences
from app.models import (
    AttributedSentence,
    CombinedInsightSection,
    CombinedQuizSection,
    InsightIntersection,
    InteractionTurnRecord,
    ReflectionPoint,
)
from frontend.app import (
    _build_generation_export_json,
    _build_generation_export_markdown,
    _build_first_principles_evidence_summary,
    _build_quick_elaboration_prompt,
    _build_quick_quiz_prompt,
    _build_quiz_grading_prompt,
    _compose_assistant_chat_message,
    _normalize_continuous_text_for_display,
    _parse_summary_sections,
    _text_overlap_ratio,
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
        self.assertNotIn(" - ", normalized)
        self.assertIn("First causal step", normalized)
        self.assertIn("Second causal step", normalized)

    def test_word_truncation_respects_limit(self) -> None:
        raw = " ".join(f"word{index}" for index in range(1, 1201))
        truncated = _truncate_to_max_words(raw, max_words=900)
        self.assertLessEqual(len(truncated.split()), 900)

    def test_section_novelty_reduces_overlap_ratio(self) -> None:
        current = (
            "Core mechanism starts with delayed reward weighting. "
            "Core mechanism starts with delayed reward weighting. "
            "Failure appears when assumptions drift under noisy feedback."
        )
        prior = ["Core mechanism starts with delayed reward weighting."]

        ratio_before = _section_redundancy_ratio(current, prior)
        refined = _enforce_section_novelty(current, prior)
        ratio_after = _section_redundancy_ratio(refined, prior)

        self.assertGreater(ratio_before, 0.0)
        self.assertLessEqual(ratio_after, ratio_before)

    def test_pairwise_overlap_ratio_detects_near_duplicates(self) -> None:
        left = "The model emphasizes boundary conditions and constraint-sensitive behavior."
        right = "Constraint-sensitive behavior and boundary conditions are emphasized by the model."
        score = _pairwise_overlap_ratio(left, right)
        self.assertGreater(score, 0.5)

    def test_extract_claims_filters_short_or_duplicate_sentences(self) -> None:
        text = (
            "Short. "
            "This mechanism predicts learning-rate collapse under delayed feedback loops. "
            "This mechanism predicts learning-rate collapse under delayed feedback loops. "
            "A second substantive claim appears when constraints tighten and variance rises."
        )

        claims = _extract_claims(text)

        self.assertEqual(len(claims), 2)
        self.assertTrue(any("learning-rate collapse" in claim for claim in claims))
        self.assertTrue(any("constraints tighten" in claim for claim in claims))

    def test_pairwise_claim_overlap_ratio_detects_shared_claims(self) -> None:
        left = (
            "Delayed rewards distort optimization in early training phases. "
            "Regularization restores stability under noisy constraints."
        )
        right = (
            "Regularization restores stability under noisy constraints. "
            "Constraint shocks can still destabilize downstream transfer."
        )

        ratio = _pairwise_claim_overlap_ratio(left, right)
        self.assertAlmostEqual(ratio, 0.5, places=2)

    def test_source_hard_cut_drops_overlap_prone_sections(self) -> None:
        summary = "Core behavior depends on delayed feedback stability and constrained updates."
        deep = "Delayed feedback stability and constrained updates are central constraints in this mechanism."
        under = "Delayed feedback stability and constrained updates are central constraints in this mechanism."
        reflection_points = [
            ReflectionPoint(
                question="What fails first?",
                explanation="Delayed feedback stability and constrained updates are central constraints in this mechanism.",
                under_the_hood="Delayed feedback stability and constrained updates are central constraints in this mechanism.",
                depth_level="foundational",
            )
        ]

        (
            cut_deep,
            cut_under,
            cut_reflection,
            cut_checklist,
            cut_terms,
            cut_labels,
        ) = _apply_source_hard_cuts(
            summary_text=summary,
            deep_dive_text=deep,
            under_surface_explainer=under,
            reflection_points=reflection_points,
            diagnostic_checklist=["Check whether delayed feedback is stable."],
            key_term_explanations=[],
        )

        self.assertTrue(cut_labels)
        self.assertTrue((not cut_reflection) or (not cut_under) or (not cut_deep))
        if not cut_under:
            self.assertEqual(cut_checklist, [])
            self.assertEqual(cut_terms, [])


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


class ControllerGuardTests(unittest.TestCase):
    def test_build_source_intent_map_has_expected_roles(self) -> None:
        intent_map = _build_source_intent_map(
            summary_text="The mechanism stabilizes when feedback delay is bounded.",
            progression_outline="Early phase sets assumptions, late phase stresses constraints.",
        )
        self.assertIn("deep_dive", intent_map)
        self.assertIn("under_surface", intent_map)
        self.assertIn("reflection", intent_map)
        self.assertIn("required_variables", intent_map["deep_dive"])

    def test_extract_variable_families_detects_constraints_and_assumptions(self) -> None:
        families = _extract_variable_families(
            "A hidden assumption appears under latency limits, and failure occurs when drift grows."
        )
        self.assertIn("assumption", families)
        self.assertIn("constraint", families)
        self.assertIn("failure_mode", families)

    def test_section_has_required_variables_requires_new_signal(self) -> None:
        has_new = _section_has_required_variables(
            section_text="The system fails under instability when boundary constraints tighten.",
            prior_texts=["The summary explains baseline mechanism behavior."],
            required_families=["constraint", "failure_mode"],
        )
        self.assertTrue(has_new)

    def test_classify_interaction_types_detects_dependency_and_tradeoff(self) -> None:
        types = _classify_interaction_types(
            "This setup depends on calibrated inputs and introduces a trade-off between latency and stability."
        )
        self.assertIn("dependency", types)
        self.assertIn("tradeoff", types)

    def test_source_restatement_ratio_is_higher_for_rephrased_source_text(self) -> None:
        ratio = _source_restatement_ratio(
            "The model depends on calibrated inputs and stable delay windows.",
            [
                "Calibrated inputs and stable delay windows are required by the model.",
                "A separate document discusses downstream monitoring.",
            ],
        )
        self.assertGreater(ratio, 0.2)

    def test_interaction_stage_progression_requires_new_types(self) -> None:
        stage_types = _assert_interaction_stage_progression(
            stage_name="cross-source:dependency",
            stage_text="This method depends on the upstream signal and requires stable calibration.",
            used_types={"connection"},
            required_new_types={"dependency", "transfer"},
            source_texts=["Source summary describes baseline mechanisms."],
            max_source_restatement_ratio=0.9,
        )
        self.assertIn("dependency", stage_types)

    def test_classify_deep_dive_operation(self) -> None:
        self.assertEqual(
            _classify_deep_dive_operation("The system fails when delay exceeds the threshold."),
            "failure_mode",
        )
        self.assertEqual(
            _classify_deep_dive_operation("Intervention requires calibrating the boundary window."),
            "intervention",
        )

    def test_filter_deep_dive_sentences_removes_explanatory_lines(self) -> None:
        filtered = _filter_deep_dive_sentences(
            (
                "A constraint means the model has limited headroom. "
                "The system fails when delayed updates cross the stability threshold. "
                "Intervention mitigates collapse by recalibrating update cadence."
            ),
            ["The summary already introduced baseline mechanism behavior."],
        )
        self.assertNotIn("means the model", filtered.lower())
        self.assertIn("fails when", filtered.lower())
        self.assertIn("intervention", filtered.lower())

    def test_filter_under_surface_sentences_keeps_assumption_extension(self) -> None:
        filtered = _filter_under_surface_sentences(
            (
                "This concept refers to delayed feedback. "
                "An implicit assumption breaks under distribution drift unless calibration is refreshed."
            ),
            ["Delayed feedback appears in the summary."],
        )
        self.assertNotIn("refers to", filtered.lower())
        self.assertIn("implicit assumption", filtered.lower())

    def test_progressive_cross_stage_map_and_validation(self) -> None:
        insights = CombinedInsightSection(
            intersections=[
                InsightIntersection(
                    intersection_title="Signal coupling",
                    why_it_matters="Sources connect around calibration reliability.",
                    integrated_explanation="The sources align on how calibration links upstream delay and downstream stability.",
                    attributed_sentences=[
                        AttributedSentence(
                            text="Calibration links delay and stability under load.",
                            source_id=1,
                            source_type="video",
                            emphasis_terms=["calibration"],
                        ),
                        AttributedSentence(
                            text="Document notes connect latency windows to reliability.",
                            source_id=2,
                            source_type="document",
                            emphasis_terms=["latency"],
                        ),
                    ],
                )
            ],
            layman_bridge="The two sources connect through a shared calibration bridge.",
            synthesis_text="A constraint appears when throughput rises and breakdown risk grows.",
            comparative_analysis="Trade-off pressure appears because stability improves at the expense of latency.",
            application_scenarios=[],
            model_name="gpt-4o-mini",
            schema_version=10,
        )
        quiz = CombinedQuizSection(
            questions=[],
            study_advice="Decide which control to prioritize when implication risk increases.",
            model_name="gpt-4o-mini",
            schema_version=10,
        )

        stage_texts = _build_progressive_cross_stage_texts(insights, quiz)
        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Summary context unrelated to mapping wording."],
            mapping_lock_claims=None,
        )
        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_allows_mixed_decision_transfer_language(self) -> None:
        stage_texts = {
            "mapping": "The sources connect through calibration assumptions and shared reliability framing.",
            "constraint": "A constraint emerges when load increases, and failure risk rises under tighter windows.",
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": (
                "When applying these transfer steps, decide which control to prioritize first. "
                "Which recommendation best limits implication risk under shifting constraints?"
            ),
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_allows_mapping_intersection_language(self) -> None:
        stage_texts = {
            "mapping": (
                "A shared intersection appears across sources: both describe a common reliability mechanism "
                "that overlaps in how boundary assumptions drive outcomes."
            ),
            "constraint": "A constraint emerges when load increases, and failure risk rises under tighter windows.",
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Decide which intervention to prioritize based on implication risk and timing.",
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_allows_mapping_with_light_decision_language(self) -> None:
        stage_texts = {
            "mapping": (
                "The sources connect through a shared intersection in reliability assumptions. "
                "This mapping shows which overlap matters most when both mechanisms align."
            ),
            "constraint": "A constraint emerges when load increases, and failure risk rises under tighter windows.",
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Decide which intervention to prioritize based on implication risk and timing.",
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_allows_mapping_with_both_across_signal(self) -> None:
        stage_texts = {
            "mapping": (
                "Both sources describe the same reliability mechanism across contexts, "
                "between upstream calibration and downstream stability behavior."
            ),
            "constraint": "A constraint emerges when load increases, and failure risk rises under tighter windows.",
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Decide which intervention to prioritize based on implication risk and timing.",
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_allows_mapping_without_explicit_mapping_tokens(self) -> None:
        stage_texts = {
            "mapping": (
                "Reliability behavior appears in the video and the notes with matching assumptions "
                "about calibration drift under load."
            ),
            "constraint": "A constraint emerges when load increases, and failure risk rises under tighter windows.",
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Decide which intervention to prioritize based on implication risk and timing.",
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_allows_constraint_without_explicit_constraint_tokens(self) -> None:
        stage_texts = {
            "mapping": "The sources connect through calibration assumptions and shared reliability framing.",
            "constraint": (
                "Both sources share the same reliability pattern across contexts, and this overlap "
                "becomes fragile as load and noise increase."
            ),
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Decide which intervention to prioritize based on implication risk and timing.",
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_progressive_cross_stage_rejects_decision_stage_without_decision_signal(self) -> None:
        stage_texts = {
            "mapping": "The sources connect through calibration assumptions and shared reliability framing.",
            "constraint": "A constraint emerges when load increases, and failure risk rises under tighter windows.",
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Transfer mitigation and adaptation continue across scenarios with repeated application steps.",
        }

        with self.assertRaises(ValueError):
            _assert_progressive_cross_structure(
                stage_texts=stage_texts,
                source_texts=["Baseline source summary content."],
                mapping_lock_claims=None,
            )

    def test_progressive_cross_stage_allows_mixed_constraint_decision_language(self) -> None:
        stage_texts = {
            "mapping": "The sources connect through calibration assumptions and shared reliability framing.",
            "constraint": (
                "A key constraint appears under load when failure risk compounds at boundary conditions. "
                "You should prioritize the bottleneck check first when this breakdown starts to spread."
            ),
            "transfer": "Adapt mitigation steps to transfer controls between scenarios when friction appears.",
            "decision": "Decide which intervention to prioritize based on implication risk and timing.",
        }

        mapping_claims = _assert_progressive_cross_structure(
            stage_texts=stage_texts,
            source_texts=["Baseline source summary content."],
            mapping_lock_claims=None,
        )

        self.assertTrue(mapping_claims)

    def test_tighten_stage_text_removes_mapping_reanchor_in_constraint_stage(self) -> None:
        tightened, _ = _tighten_stage_text(
            stage_name="constraint",
            text_value=(
                "The sources connect through a shared mechanism across both materials. "
                "A key constraint appears when noise rises and failure risk increases."
            ),
            prior_stage_texts=["The sources connect through a shared mechanism across both materials."],
        )

        self.assertIn("constraint", tightened.lower())
        self.assertNotIn("shared mechanism across both materials", tightened.lower())

    def test_progressive_tightening_caps_mapping_and_constraint_paragraphs(self) -> None:
        tightened = _tighten_progressive_cross_stage_texts(
            stage_texts={
                "mapping": (
                    "Both sources connect through shared calibration assumptions. "
                    "Both sources connect through shared calibration assumptions again. "
                    "Across both sources, overlapping control loops align outcomes. "
                    "Between both materials, the same mapping pattern appears under drift."
                ),
                "constraint": (
                    "A constraint emerges when load increases and boundary limits tighten. "
                    "Failure appears when load increases and boundary limits tighten. "
                    "A tradeoff appears because stabilizing one path degrades another under pressure. "
                    "Constraint risk compounds when assumptions no longer hold."
                ),
                "transfer": (
                    "Apply the mitigation sequence to adapt controls in deployment. "
                    "Then adapt monitoring to transfer the control law safely."
                ),
                "decision": (
                    "Decide which control to prioritize first under uncertainty. "
                    "Choose the tradeoff with lower failure impact for this context."
                ),
            }
        )

        mapping_paragraphs = [part for part in tightened["mapping"].split("\n\n") if part.strip()]
        constraint_paragraphs = [part for part in tightened["constraint"].split("\n\n") if part.strip()]
        self.assertLessEqual(len(mapping_paragraphs), 2)
        self.assertLessEqual(len(constraint_paragraphs), 2)


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
        self.assertNotIn(" - ", normalized)
        self.assertIn("Core idea", normalized)
        self.assertIn("Second point", normalized)

    def test_parse_summary_sections_extracts_expected_titles(self) -> None:
        summary = (
            "## Core Thesis and Scope\n"
            "This source defines the central claim.\n\n"
            "### Key Mechanisms and How They Work\n"
            "Mechanism details appear here.\n\n"
            "Practical Implications and Limitations\n"
            "Boundaries and tradeoffs are discussed."
        )
        parsed = _parse_summary_sections(summary)
        titles = [title for title, _ in parsed]
        self.assertEqual(
            titles,
            [
                "Core Thesis and Scope",
                "Key Mechanisms and How They Work",
                "Practical Implications and Limitations",
            ],
        )

    def test_first_principles_evidence_summary_omits_source_labels(self) -> None:
        summary = _build_first_principles_evidence_summary(
            [
                ("Video: Test", "Mechanism starts with delayed feedback.", ["feedback"]),
                ("Document: Notes", "Constraint pressure changes the strategy.", ["constraint"]),
            ]
        )
        self.assertIn("Mechanism starts", summary)
        self.assertNotIn("Video:", summary)

    def test_text_overlap_ratio_captures_shared_terms(self) -> None:
        score = _text_overlap_ratio(
            "Transfer constraints shape outcomes in practical deployments.",
            "Practical deployments are shaped by transfer constraints and assumptions.",
        )
        self.assertGreater(score, 0.4)

    def test_expand_followup_query_includes_prior_claims_and_dimensions(self) -> None:
        expanded = _expand_followup_query(
            "Elaborate further",
            [
                InteractionTurnRecord(
                    query="How does this work?",
                    answer=(
                        "The system depends on delayed feedback alignment. "
                        "When alignment breaks, instability emerges under constraint pressure."
                    ),
                    follow_up_question=None,
                )
            ],
        )
        self.assertIn("Prior claims to avoid repeating", expanded)
        self.assertIn("hidden assumption", expanded)

    def test_strict_redundancy_strip_can_return_shorter_output(self) -> None:
        current = (
            "The mechanism depends on delayed feedback stability. "
            "The mechanism depends on delayed feedback stability. "
            "A hidden assumption is stable calibration."
        )
        reference = "The mechanism depends on delayed feedback stability."
        stripped = _strip_redundant_sentences(
            current,
            reference,
            similarity_threshold=0.74,
            token_overlap_threshold=0.5,
            min_char_ratio=0.9,
            allow_shorter_output=True,
        )
        self.assertIn("hidden assumption", stripped.lower())
        self.assertNotIn("depends on delayed feedback stability", stripped.lower())


class ExportBuilderTests(unittest.TestCase):
    def test_generation_export_markdown_includes_sources_and_cross_sections(self) -> None:
        generation_result = {
            "source_ids": [1, 2],
            "video": {
                "source_id": 1,
                "source_type": "video",
                "source_name": "Video Source Name",
                "generated_title": "Video Topic",
                "summary_text": "Video summary.",
                "deep_dive_text": "Video deep dive.",
                "key_terms": ["alpha", "beta"],
                "under_surface_explainer": "Video mechanism.",
                "diagnostic_checklist": ["Check signal drift"],
                "reflection_points": [
                    {
                        "question": "Why does this work?",
                        "explanation": "Because constraints align.",
                        "under_the_hood": "Underlying mechanism.",
                    }
                ],
            },
            "documents": [
                {
                    "source_id": 2,
                    "source_type": "document",
                    "source_name": "Doc Source Name",
                    "generated_title": "Doc Topic",
                    "summary_text": "Doc summary.",
                    "deep_dive_text": "Doc deep dive.",
                    "key_terms": ["gamma"],
                    "under_surface_explainer": "Doc mechanism.",
                    "reflection_points": [],
                }
            ],
            "insights": {
                "intersections": [
                    {
                        "intersection_title": "Shared mechanism",
                        "why_it_matters": "It improves transfer.",
                        "integrated_explanation": "The overlap is causal.",
                        "attributed_sentences": [
                            {"text": "Evidence text.", "source_id": 1}
                        ],
                    }
                ],
                "layman_bridge": "Bridge text.",
                "synthesis_text": "Synthesis text.",
                "comparative_analysis": "Comparative text.",
                "application_scenarios": [
                    {
                        "scenario_title": "Scenario A",
                        "scenario_prompt": "Use in production",
                        "transfer_steps": ["Step one", "Step two"],
                        "common_pitfall": "Overfit assumptions",
                    }
                ],
            },
            "quiz": {
                "questions": [
                    {
                        "question": "What should you do?",
                        "options": ["A", "B", "C", "D"],
                        "answer_index": 1,
                        "explanation": "Because B is right.",
                        "under_the_hood": "Mechanism detail.",
                    }
                ],
                "study_advice": "Review mistakes.",
            },
        }

        markdown_export = _build_generation_export_markdown(generation_result, user_id="demo-user")

        self.assertIn("# Socratic Study Snapshot", markdown_export)
        self.assertIn("# Source Learning", markdown_export)
        self.assertIn("## Video: Video Topic", markdown_export)
        self.assertIn("## Document: Doc Topic", markdown_export)
        self.assertIn("# Cross-Source Synthesis and Assessment", markdown_export)
        self.assertIn("## 1) Mapping", markdown_export)
        self.assertIn("## 2) Constraint or Breakdown", markdown_export)
        self.assertIn("## 3) Transfer and Adaptation", markdown_export)
        self.assertIn("## 4) Decision and Implication", markdown_export)

    def test_generation_export_json_wraps_generation_payload(self) -> None:
        generation_result = {
            "source_ids": [3],
            "video": {"generated_title": "Only source"},
            "documents": [],
            "insights": {},
            "quiz": {},
        }

        json_export = _build_generation_export_json(generation_result, user_id="demo-user")

        self.assertIn('"user_id": "demo-user"', json_export)
        self.assertIn('"generation_result"', json_export)
        self.assertIn('"source_ids": [', json_export)


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
