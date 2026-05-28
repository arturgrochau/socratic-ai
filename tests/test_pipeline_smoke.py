"""
End-to-end smoke test for the v2.1 outline-first generation pipeline using
a mock LLM client. No API key required.

Verifies that:
  * The pipeline runs outline → section drafts (with mini-RAG context)
    → critic → optional refine → key concepts → reflection points for a
    single source.
  * Two-source synthesis populates the v2.1 quiz fields
    (option_rationales + deeper_why) and produces a non-empty intersections
    list.
  * Section titles are content-driven (not the v2.0 fixed scaffolding).
  * No banned opener appears at the start of any section body.
  * Every KeyTermExplanation has a non-empty `explanation`.

Runs in CI alongside the other unit tests (no `-m e2e` marker).
"""
from __future__ import annotations

import importlib
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock


@dataclass
class _FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class _FakeResult:
    content: str
    raw: Any = None
    usage: _FakeUsage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _FakeUsage()
        if self.raw is None:
            self.raw = _FakeRaw(self.usage)


@dataclass
class _FakeRaw:
    usage: _FakeUsage


# ---------------------------------------------------------------------------
# Canned responses keyed by JSON schema name. Plain-text (no schema) calls
# return prose suitable for the section draft + refine steps.
# ---------------------------------------------------------------------------

_OUTLINE_RESPONSE = {
    "sections": [
        {
            "title": "How feedback loops shape system behavior",
            "scope": "Set up positive vs. negative feedback and stability.",
            "expected_paragraphs": 3,
        },
        {
            "title": "Why delays cause oscillation even in stable loops",
            "scope": "Explain delay-driven overshoot inside a single loop.",
            "expected_paragraphs": 3,
        },
        {
            "title": "When loop topology overrides loop polarity",
            "scope": "Move from within-loop to cross-loop dynamics.",
            "expected_paragraphs": 3,
        },
    ],
}

_CRITIC_PASS = {
    "banned_phrase_flags": [],
    "redundancy_flags": [],
    "scope_drift": False,
    "tone_issues": [],
    "needs_refinement": False,
    "instructions": "",
}

_KEY_CONCEPTS_RESPONSE = {
    "key_concepts": [
        {
            "term": "Positive feedback",
            "explanation": (
                "A self-amplifying loop where the output of a process makes "
                "its own cause stronger. Like a microphone next to a speaker: "
                "the louder it gets, the louder it gets, until something else "
                "stops it."
            ),
        },
        {
            "term": "Negative feedback",
            "explanation": (
                "A self-correcting loop that pushes back against its own "
                "cause. A thermostat is the classic example: as the room "
                "warms up, the heater works less."
            ),
        },
        {
            "term": "Regime shift",
            "explanation": (
                "A sudden jump from one stable behavior to another, often "
                "because a slow variable quietly crossed a threshold. The "
                "system looks fine right up until it doesn't."
            ),
        },
    ],
}

_REFLECTION_RESPONSE = {
    "reflection_points": [
        {
            "question": "Why might a system with strong negative feedback still oscillate?",
            "explanation": (
                "Delay between cause and corrective action can push the "
                "response past the setpoint before it has time to register, "
                "creating overshoot. The careful thinker notices that "
                "stability is a property of timing, not just polarity."
            ),
            "depth_level": "intermediate",
        },
        {
            "question": "What conditions cause a positive feedback loop to terminate naturally?",
            "explanation": (
                "Either the resource feeding the loop runs out, the amplified "
                "variable saturates against a physical ceiling, or a slower "
                "negative loop activates and overtakes it. Spotting which "
                "limit is binding tells you where to intervene."
            ),
            "depth_level": "advanced",
        },
        {
            "question": "How would you distinguish positive feedback from a correlated trend?",
            "explanation": (
                "Test whether the relationship survives an intervention. "
                "Correlated trends fall apart when you perturb the system; "
                "true feedback re-establishes itself."
            ),
            "depth_level": "foundational",
        },
        {
            "question": "Where would you expect cross-loop topology to dominate over within-loop delay?",
            "explanation": (
                "In systems where many loops share variables and interactions "
                "are tight. Cascading grid failures are the canonical case — "
                "no single loop explains the outage; the connection pattern does."
            ),
            "depth_level": "advanced",
        },
    ],
}

_SYNTHESIS_RESPONSE = {
    "synthesis_text": (
        "Both sources treat feedback as the engine of system behavior, but "
        "they disagree on where the interesting structure lives. The first "
        "centers loop polarity and delay; the second emphasizes how loops "
        "interconnect. Together they imply that isolated loops are a teaching "
        "abstraction, and that real systems live in the interaction between "
        "loops."
    ),
    "intersections": [
        {
            "title": "Delay vs. topology as the dominant explanation",
            "why_it_matters": "Choosing the wrong frame leads to interventions at the wrong leverage point.",
            "integrated_explanation": (
                "Delay matters within a loop and topology matters across loops. "
                "Non-trivial systems require both analyses simultaneously."
            ),
        },
        {
            "title": "Stability is multi-scale",
            "why_it_matters": "Local stability can mask global fragility.",
            "integrated_explanation": (
                "The first source's local-stability picture coexists with the "
                "second source's topological-vulnerability picture. Both scales "
                "have to be examined."
            ),
        },
    ],
    "questions": [
        {
            "question": "When does loop topology matter more than loop polarity?",
            "options": [
                "When delays are negligible and loops interact strongly",
                "When a single loop dominates all dynamics",
                "When the system has no external perturbations",
                "When the time horizon is very short",
            ],
            "answer_index": 0,
            "explanation": (
                "Topology dominates when interactions between loops exceed any "
                "single loop's dynamics — which happens precisely when delays "
                "do not isolate them."
            ),
            "option_rationales": [
                "Correctly identifies that cross-loop interactions dominate when delays do not buffer them.",
                "Confuses dominance with absence — a dominant loop doesn't make topology irrelevant; it makes it static.",
                "Mistakes 'no perturbations' for stability; topology shapes how perturbations propagate.",
                "Confuses short horizons with topological simplicity; cross-loop effects show up fastest at short horizons.",
            ],
            "deeper_why": (
                "Cross-loop coupling raises the dimension of the dynamical "
                "system: a topology change that costs nothing locally can "
                "shift global attractors. This is why network analyses often "
                "surface fragilities that loop-by-loop reviews miss."
            ),
        },
        {
            "question": "Which scenario most clearly requires both frames together?",
            "options": [
                "Diagnosing a thermostat that never overshoots",
                "Modeling a population with a single growth driver",
                "Predicting cascading failures across an interconnected grid",
                "Estimating the half-life of a decay process",
            ],
            "answer_index": 2,
            "explanation": (
                "Cascading failures emerge from how loops connect, while the "
                "speed of cascade depends on within-loop delays. Both frames "
                "are essential."
            ),
            "option_rationales": [
                "Treats a single-loop diagnostic as if it were multi-loop; the thermostat is precisely the case where one frame suffices.",
                "A single-driver population has no cross-loop interaction to analyze.",
                "Correctly identifies a problem that requires both within-loop delay analysis and cross-loop topology analysis.",
                "Decay processes are well-described by simple rate laws — neither frame is needed.",
            ],
            "deeper_why": (
                "In an interconnected grid the failure speed is set by the "
                "fastest within-loop response, but the failure shape is set by "
                "the connection topology. Interventions that ignore one or the "
                "other tend to fix the wrong outage class."
            ),
        },
    ],
    "application_scenarios": [
        {
            "scenario_title": "Diagnosing a recurring incident in a production system",
            "scenario_prompt": "The on-call team sees the same outage shape every few weeks despite each fix working.",
            "transfer_steps": [
                "Map the visible loops and their delays.",
                "Identify which loop interactions activate only at high load.",
                "Look for a slow variable crossing a threshold between incidents.",
            ],
            "common_pitfall": "Fixing the most recent visible loop while the cross-loop interaction driving the cycle stays untouched.",
        },
        {
            "scenario_title": "Designing a regulatory intervention",
            "scenario_prompt": "A policy team wants to dampen a market boom-bust cycle.",
            "transfer_steps": [
                "Distinguish loops within actors from loops across actors.",
                "Identify which delays transparency can shorten.",
                "Pilot the intervention and measure for topology shifts, not just polarity shifts.",
            ],
            "common_pitfall": "Treating the system as one big loop when boom-bust emerges from coupling between many smaller loops.",
        },
    ],
}


class _FakeClient:
    """LLMClient stand-in. Routes by json_schema.name; falls back to prose."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> _FakeResult:
        if json_schema is None:
            # Section draft or refine. Return prose that opens with a concrete
            # claim — never a banned phrase. We seed the body with section-
            # specific anchor words so the mini-RAG sees realistic content.
            self.calls.append("plain")
            return _FakeResult(
                content=(
                    "Feedback loops shape system behavior by linking outputs "
                    "back to inputs, either amplifying small changes or "
                    "damping them.\n\n"
                    "The polarity of the loop determines whether perturbations "
                    "grow or decay over time. Delay between cause and "
                    "corrective response is what turns a stable loop into an "
                    "oscillating one."
                )
            )
        schema_name = json_schema.get("name", "")
        self.calls.append(schema_name)
        if schema_name == "source_outline":
            return _FakeResult(content=json.dumps(_OUTLINE_RESPONSE))
        if schema_name == "section_critique":
            return _FakeResult(content=json.dumps(_CRITIC_PASS))
        if schema_name == "key_concepts":
            return _FakeResult(content=json.dumps(_KEY_CONCEPTS_RESPONSE))
        if schema_name == "reflection_points":
            return _FakeResult(content=json.dumps(_REFLECTION_RESPONSE))
        if schema_name == "cross_source_synthesis":
            return _FakeResult(content=json.dumps(_SYNTHESIS_RESPONSE))
        raise AssertionError(f"Unexpected schema name in smoke test: {schema_name!r}")

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        # Deterministic length-aware embedding so the mini-RAG has something
        # meaningful to compare. Not used for retrieval correctness here.
        out: list[list[float]] = []
        for text in inputs:
            length = max(len(text), 1)
            out.append([
                (length % 7) / 10.0,
                (length % 11) / 10.0,
                (length % 13) / 10.0,
            ])
        return out


class PipelineSmokeTests(unittest.TestCase):
    """v2.1 outline-first pipeline end-to-end with a mocked LLM."""

    def _setup_isolated_env(self, tmpdir: str) -> tuple[Any, Any]:
        import os

        os.environ["DATABASE_URL"] = f"sqlite:///{tmpdir}/smoke.db"
        os.environ["SESSION_LOG_DIR"] = f"{tmpdir}/sessions"
        os.environ["ENABLE_SESSION_LOG"] = "true"
        os.environ["CACHE_PROCESSED_SOURCES"] = "false"
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-smoke")

        import config
        importlib.reload(config)

        from app import session_logger as session_logger_mod
        importlib.reload(session_logger_mod)
        from app import cost_logging as cost_logging_mod
        importlib.reload(cost_logging_mod)
        from app import models as models_mod
        importlib.reload(models_mod)
        from app import ingestion as ingestion_mod
        importlib.reload(ingestion_mod)
        from app import generation as generation_mod
        importlib.reload(generation_mod)

        return config, generation_mod

    def _seed_source(
        self,
        config_mod: Any,
        *,
        source_id_filename: str,
        user_id: str,
        chunks: list[str],
    ) -> int:
        from sqlalchemy import text as sql_text

        with config_mod.db_engine.begin() as connection:
            result = connection.execute(
                sql_text(
                    "INSERT INTO sources (user_id, source_type, filename) "
                    "VALUES (:u, 'document', :f)"
                ),
                {"u": user_id, "f": source_id_filename},
            )
            source_id = int(result.lastrowid)  # type: ignore[arg-type]
            for idx, chunk in enumerate(chunks):
                connection.execute(
                    sql_text(
                        "INSERT INTO source_text_chunks "
                        "(user_id, source_id, chunk_index, chunk_type, chunk_text) "
                        "VALUES (:u, :s, :i, 'document', :t)"
                    ),
                    {"u": user_id, "s": source_id, "i": idx, "t": chunk},
                )
        return source_id

    def test_single_document_pipeline(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_mod, gen_mod = self._setup_isolated_env(tmpdir)

            from app.cost_logging import ensure_cost_logging_tables
            from app.ingestion import ensure_ingestion_tables
            ensure_cost_logging_tables()
            ensure_ingestion_tables()
            gen_mod.ensure_generation_tables()

            user_id = "smoke-single"
            source_id = self._seed_source(
                config_mod,
                source_id_filename="feedback_loops.pdf",
                user_id=user_id,
                chunks=[
                    "Feedback loops are the building blocks of system behavior.",
                    "Positive feedback amplifies; negative feedback regulates.",
                    "Delays in feedback produce oscillations.",
                    "Thresholds can trigger regime shifts in complex systems.",
                ],
            )

            fake_client = _FakeClient()
            with mock.patch.object(config_mod, "get_llm_client", return_value=fake_client), \
                 mock.patch.object(gen_mod, "get_llm_client", return_value=fake_client):
                response = gen_mod.generate_tailored_learning(
                    video_source_id=None,
                    document_source_ids=[source_id],
                    user_id=user_id,
                )

            self.assertEqual(len(response.documents), 1)
            doc = response.documents[0]

            # Content-driven sections (v2.1).
            self.assertGreaterEqual(len(doc.sections), 3, "expected outline-driven sections")
            generic = {"summary", "deep dive", "overview", "conclusion"}
            non_generic_titles = [s.title for s in doc.sections if s.title.lower() not in generic]
            self.assertTrue(
                non_generic_titles,
                f"At least one section title should be content-driven; got {[s.title for s in doc.sections]}",
            )

            # No banned opener at the start of any section body.
            from prompts.banned_phrases import BANNED_OPENERS
            for s in doc.sections:
                first = s.body.lstrip().lower()
                for phrase in BANNED_OPENERS:
                    self.assertFalse(
                        first.startswith(phrase.lower()),
                        f"Banned opener {phrase!r} leaked into section '{s.title}'",
                    )

            # Key term explanations: single-field, all non-empty.
            self.assertGreaterEqual(len(doc.key_term_explanations), 3)
            for kte in doc.key_term_explanations:
                self.assertTrue(kte.term.strip(), "key term name missing")
                self.assertTrue(kte.explanation.strip(), "key term explanation missing")

            # Reflection points span depth levels.
            depths = {p.depth_level for p in doc.reflection_points}
            self.assertGreaterEqual(
                len(depths), 2,
                f"Expected varied reflection depths; got {depths}",
            )

            # Single-source path: no synthesis content.
            self.assertEqual(response.insights.synthesis_text, "")
            self.assertEqual(response.quiz.questions, [])

    def test_two_document_synthesis(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_mod, gen_mod = self._setup_isolated_env(tmpdir)

            from app.cost_logging import ensure_cost_logging_tables
            from app.ingestion import ensure_ingestion_tables
            ensure_cost_logging_tables()
            ensure_ingestion_tables()
            gen_mod.ensure_generation_tables()

            user_id = "smoke-two"
            id_a = self._seed_source(
                config_mod,
                source_id_filename="source_a.pdf",
                user_id=user_id,
                chunks=[
                    "Source A discusses loop polarity and delay.",
                    "Within a loop, response time determines stability.",
                    "Damping is achieved through negative feedback.",
                ],
            )
            id_b = self._seed_source(
                config_mod,
                source_id_filename="source_b.pdf",
                user_id=user_id,
                chunks=[
                    "Source B treats loop topology as the central concept.",
                    "Cross-loop interactions dominate large-system behavior.",
                    "Cascade failures arise from coupling structure.",
                ],
            )

            fake_client = _FakeClient()
            with mock.patch.object(config_mod, "get_llm_client", return_value=fake_client), \
                 mock.patch.object(gen_mod, "get_llm_client", return_value=fake_client):
                response = gen_mod.generate_tailored_learning(
                    video_source_id=None,
                    document_source_ids=[id_a, id_b],
                    user_id=user_id,
                )

            self.assertEqual(len(response.documents), 2)
            self.assertTrue(response.insights.synthesis_text.strip())
            self.assertGreaterEqual(len(response.insights.intersections), 1)
            self.assertGreaterEqual(len(response.insights.application_scenarios), 1)

            # v2.1 quiz fields: option_rationales (4 entries) + deeper_why.
            self.assertGreaterEqual(len(response.quiz.questions), 2)
            for q in response.quiz.questions:
                self.assertEqual(
                    len(q.option_rationales), 4,
                    f"Each question must carry 4 option_rationales; got {len(q.option_rationales)}",
                )
                for rat in q.option_rationales:
                    self.assertTrue(rat.strip(), "option_rationale must be non-empty")
                self.assertTrue(q.deeper_why.strip(), "deeper_why must be non-empty")

            # Quiz answers should not all collapse to the same index.
            answer_indices = {q.answer_index for q in response.quiz.questions}
            self.assertGreater(len(answer_indices), 1)

            # Session JSONL was written.
            log_dir = Path(tmpdir) / "sessions"
            jsonl_files = list(log_dir.glob("gen-*.jsonl"))
            self.assertTrue(jsonl_files, "no session JSONL produced")


if __name__ == "__main__":
    unittest.main()
