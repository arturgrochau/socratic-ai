"""
End-to-end smoke test for the generation pipeline using a mock LLM client.

No API key required. Verifies that:
  * Per-source iterative accumulation + consolidation produces a complete
    SourceLearningSection with layman + technical key term explanations.
  * Multi-source synthesis produces a CombinedInsightSection and a quiz.
  * DB tables are created, sections are persisted, and reflection points
    spanning depth_levels make it through.

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
            # Make raw quack like the OpenAI completion for log_api_usage.
            self.raw = _FakeRaw(self.usage)


@dataclass
class _FakeRaw:
    usage: _FakeUsage


# ---------------------------------------------------------------------------
# Canned responses keyed by JSON schema name. None => plain accumulation step.
# ---------------------------------------------------------------------------

_CONSOLIDATION_RESPONSE = {
    "summary": (
        "The material covers how feedback loops shape complex systems. It shows "
        "that small perturbations can be amplified through positive feedback and "
        "dampened through negative feedback. Stability emerges when these forces "
        "balance over time."
    ),
    "deep_dive": (
        "Feedback loops operate through delays and thresholds. When the delay is "
        "long compared to the response time, oscillation appears. Threshold "
        "effects produce regime shifts that look discontinuous from the outside "
        "but emerge from continuous internal dynamics. Failure modes cluster "
        "around mistaking correlation for causation in feedback diagrams."
    ),
    "key_terms": [
        {
            "term": "Positive feedback",
            "layman": "When an effect makes its own cause stronger, like a microphone screech.",
            "technical": "A loop where the output of a process reinforces the input, leading to exponential growth or runaway dynamics absent a limiting factor.",
        },
        {
            "term": "Negative feedback",
            "layman": "When an effect pushes back against its own cause, like a thermostat.",
            "technical": "A regulatory loop where the output of a process opposes the input, producing convergence toward a setpoint or equilibrium.",
        },
        {
            "term": "Regime shift",
            "layman": "When a system suddenly switches to a new mode of behavior.",
            "technical": "A discontinuous transition between attractor basins, often triggered when a slow variable crosses a critical threshold.",
        },
    ],
    "under_surface": (
        "Most explanations of feedback assume the loop structure is given. The "
        "harder question is how loop topology itself emerges, and what conditions "
        "cause loops to form, dissolve, or invert sign."
    ),
    "reflection_points": [
        {
            "question": "Why might a system with strong negative feedback still oscillate?",
            "explanation": "Delay between cause and corrective action can overshoot the setpoint, producing oscillation despite the loop pulling toward equilibrium.",
            "depth_level": "intermediate",
        },
        {
            "question": "What conditions would cause a positive feedback loop to terminate naturally?",
            "explanation": "Exhaustion of the resource feeding the loop, saturation of the amplifying variable, or activation of a slower negative loop.",
            "depth_level": "advanced",
        },
        {
            "question": "How would you tell positive feedback apart from a one-off correlated trend?",
            "explanation": "Test whether the relationship persists across intervention or only appears under specific co-movements.",
            "depth_level": "foundational",
        },
    ],
}

_SYNTHESIS_RESPONSE = {
    "synthesis_text": (
        "Both sources treat feedback as the engine of system-level behavior, "
        "but they disagree on where the interesting structure lives. The first "
        "frames it in terms of loop polarity and delay; the second emphasizes "
        "the topology of how loops interconnect. Together they suggest that "
        "isolated loops are a teaching abstraction, and that real systems live "
        "in the interaction between loops."
    ),
    "intersections": [
        {
            "title": "Delay vs. topology as the dominant explanation",
            "why_it_matters": "Choosing the wrong frame leads to interventions that target the wrong leverage point.",
            "integrated_explanation": "The first source treats delay as the master variable while the second treats topology as primary. The synthesis is that delay matters within a loop and topology matters across loops, and both must be analyzed for non-trivial systems.",
        },
        {
            "title": "Stability is multi-scale",
            "why_it_matters": "A system can be locally stable yet globally fragile.",
            "integrated_explanation": "Local stability described in the first source coexists with topological vulnerabilities described in the second. Real-world stability requires examining both scales simultaneously.",
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
            "explanation": "Topology dominates when interaction effects between loops exceed the dynamics of any single loop, which happens precisely when delays do not isolate them.",
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
            "explanation": "Cascading failures emerge from how loops connect, while the speed of cascade depends on within-loop delays. Both frames are essential.",
        },
        {
            "question": "What is the strongest evidence the two sources are complementary?",
            "options": [
                "They use the same vocabulary throughout",
                "They cite identical case studies",
                "Their primary variables operate at different scales",
                "They reach opposing conclusions on every question",
            ],
            "answer_index": 2,
            "explanation": "Operating at different scales is exactly what makes the frames composable rather than competing.",
        },
    ],
    "application_scenarios": [
        {
            "scenario_title": "Diagnosing a recurring incident in a production system",
            "scenario_prompt": "An on-call team sees the same outage shape every few weeks despite each fix appearing to work.",
            "transfer_steps": [
                "Map the visible loops and their delays.",
                "Identify which loop interactions become active only at high load.",
                "Look for a slow variable that crosses a threshold between incidents.",
            ],
            "common_pitfall": "Fixing the most recent visible loop while ignoring the cross-loop interaction that drives the cycle.",
        },
        {
            "scenario_title": "Designing a regulatory intervention",
            "scenario_prompt": "A policy team wants to dampen a market boom-bust cycle.",
            "transfer_steps": [
                "Distinguish loops that operate within actors from loops that operate across actors.",
                "Identify which delays are shortenable by transparency vs. which require structural change.",
                "Pilot the intervention on a subset and measure for topology shifts, not just polarity shifts.",
            ],
            "common_pitfall": "Treating the system as one big loop when boom-bust dynamics emerge from how many smaller loops are coupled.",
        },
    ],
}


class _FakeClient:
    """LLMClient stand-in. Routes by json_schema name (or lack thereof)."""

    name = "fake"

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
            # Accumulation step — return plain prose.
            return _FakeResult(
                content=(
                    "The system under study exhibits feedback dynamics. Small "
                    "changes propagate through the loop and can either amplify "
                    "or dampen depending on the polarity of the connections. "
                    "Stability is a delicate balance of these competing forces."
                )
            )
        schema_name = json_schema.get("name", "")
        if schema_name == "source_consolidation":
            return _FakeResult(content=json.dumps(_CONSOLIDATION_RESPONSE))
        if schema_name == "cross_source_synthesis":
            return _FakeResult(content=json.dumps(_SYNTHESIS_RESPONSE))
        raise AssertionError(f"Unexpected schema name in smoke test: {schema_name!r}")

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        # Not used by the generation pipeline; included for protocol completeness.
        return [[0.0] * 8 for _ in inputs]


class PipelineSmokeTests(unittest.TestCase):
    """Generation pipeline runs end-to-end with a mocked LLM."""

    def _setup_isolated_env(self, tmpdir: str) -> tuple[Any, Any]:
        """Point config at a temp DB and reload the relevant modules."""
        import os

        os.environ["DATABASE_URL"] = f"sqlite:///{tmpdir}/smoke.db"
        os.environ["SESSION_LOG_DIR"] = f"{tmpdir}/sessions"
        os.environ["ENABLE_SESSION_LOG"] = "true"
        os.environ["CACHE_PROCESSED_SOURCES"] = "false"
        # Avoid the import-time OpenAI key check if not set.
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-smoke")

        # Reload modules so they pick up the new env. Order matters:
        # config first (rebuilds db_engine), then anything that imported it.
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

            # Bootstrap tables.
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
            self.assertTrue(doc.summary_text.strip(), "summary empty")
            self.assertTrue(doc.deep_dive_text.strip(), "deep_dive empty")
            self.assertTrue(doc.under_surface_explainer.strip(), "under_surface empty")
            self.assertGreaterEqual(len(doc.key_term_explanations), 3)
            for kte in doc.key_term_explanations:
                self.assertTrue(kte.term.strip(), "key term name missing")
                self.assertTrue(kte.layman.strip(), "key term layman explanation missing")
                self.assertTrue(kte.technical.strip(), "key term technical definition missing")
            self.assertGreaterEqual(len(doc.reflection_points), 3)
            depths = {p.depth_level for p in doc.reflection_points}
            # Reflection points should span at least 2 distinct depth levels.
            self.assertGreaterEqual(
                len(depths), 2,
                f"Expected varied reflection depths; got {depths}",
            )

            # Single-source path: no synthesis, but empty insights/quiz objects exist.
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
            self.assertTrue(
                response.insights.synthesis_text.strip(),
                "synthesis_text empty in multi-source run",
            )
            self.assertGreaterEqual(
                len(response.insights.intersections), 1,
                "expected at least one intersection",
            )
            self.assertGreaterEqual(
                len(response.insights.application_scenarios), 1,
                "expected at least one application scenario",
            )
            self.assertGreaterEqual(
                len(response.quiz.questions), 1,
                "expected at least one quiz question",
            )
            # Quiz answer indices should not all be identical (sanity check).
            answer_indices = {q.answer_index for q in response.quiz.questions}
            self.assertGreater(
                len(answer_indices), 1,
                "quiz answers all point at the same index",
            )

            # Session JSONL was written.
            log_dir = Path(tmpdir) / "sessions"
            jsonl_files = list(log_dir.glob("gen-*.jsonl"))
            self.assertTrue(jsonl_files, "no session JSONL produced")


if __name__ == "__main__":
    unittest.main()
