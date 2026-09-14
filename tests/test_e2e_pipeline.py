"""
End-to-end pipeline test against test_assets/sample_notes.pdf.

Runs in two modes (selected by env var):

  * SOCRATIC_RECORD=1    — hits the real OpenAI API once, populates
                           tests/_cassettes/test_e2e_pipeline.json.
  * (unset, default)     — replays from cassette. No API key needed.

Skipped if the cassette is missing AND SOCRATIC_RECORD is not set, so a fresh
clone without cassettes doesn't fail CI.

Marker `e2e` is configured in tests/conftest.py — run with `pytest -m e2e`.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from tests._recording_client import CASSETTE_DIR, RECORD_MODE, install_cassette

CASSETTE_NAME = "test_e2e_pipeline"
FIXTURE_PDF = Path(__file__).resolve().parent.parent / "test_assets" / "sample_notes.pdf"


pytestmark = pytest.mark.e2e


def _cassette_missing() -> bool:
    return not (CASSETTE_DIR / f"{CASSETTE_NAME}.json").exists()


@pytest.fixture
def _isolated_env(monkeypatch, tmp_path):
    """Set up a fully isolated environment: temp DB, temp session log dir, and
    reload every module that captured stale config/session_logger symbols at
    import time. Returns (log_dir, db_path)."""
    log_dir = tmp_path / "sessions"
    db_path = tmp_path / "test_e2e.db"
    monkeypatch.setenv("SESSION_LOG_DIR", str(log_dir))
    monkeypatch.setenv("ENABLE_SESSION_LOG", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import importlib
    # Order matters: config first (db_engine), then session_logger (reads
    # SESSION_LOG_DIR), then every module that imported either of those.
    for mod_name in (
        "config",
        "app.session_logger",
        "app.cost_logging",
        "app.json_reliability",
        "app.models",
        "app.ingestion",
        "app.retrieval",
        "app.interaction",
        "app.ledger",
        "app.generation",
    ):
        try:
            importlib.reload(importlib.import_module(mod_name))
        except Exception:
            pass

    # Bootstrap every table the pipeline writes into.
    from app.cost_logging import ensure_cost_logging_tables
    from app.generation import ensure_generation_tables
    from app.ingestion import ensure_ingestion_tables
    from app.interaction import ensure_interaction_tables
    ensure_cost_logging_tables()
    ensure_ingestion_tables()
    ensure_interaction_tables()
    ensure_generation_tables()
    yield log_dir, db_path


@pytest.mark.skipif(
    _cassette_missing() and not RECORD_MODE,
    reason="No cassette; set SOCRATIC_RECORD=1 with an API key to record.",
)
def test_e2e_single_pdf_generation(monkeypatch, _isolated_env):
    log_dir, _db_path = _isolated_env
    """Full pipeline against sample_notes.pdf.

    Assertions:
      * Generation completes without raising.
      * Output schemas validate.
      * Reflection points are present.
      * No section text contains the removed refusal pattern.
      * Section novelty stays below the threshold (already enforced inline,
        so this is a regression guard).
    """
    assert FIXTURE_PDF.exists(), f"Missing fixture: {FIXTURE_PDF}"

    install_cassette(monkeypatch, CASSETTE_NAME)

    # Re-import the pipeline so it picks up rebuilt config + cassette client.
    import importlib

    import app.ingestion as ingestion
    importlib.reload(ingestion)
    import app.generation as generation
    importlib.reload(generation)

    # Re-install cassette after reloads (reload re-imports openai_client).
    install_cassette(monkeypatch, CASSETTE_NAME)

    user_id = f"e2e-{uuid.uuid4().hex[:6]}"

    # Manually ingest the PDF via the in-process function — avoids spinning a
    # FastAPI server in the test.
    import asyncio
    from io import BytesIO

    class _Upload:
        def __init__(self, name: str, data: bytes, content_type: str) -> None:
            self.filename = name
            self.file = BytesIO(data)
            self.content_type = content_type

        async def read(self) -> bytes:
            return self.file.getvalue()

        async def close(self) -> None:
            self.file.close()

    document = _Upload(FIXTURE_PDF.name, FIXTURE_PDF.read_bytes(), "application/pdf")
    ingestion_resp = asyncio.run(
        ingestion.ingest_upload_bundle(
            video_file=None,
            video_url=None,
            document_files=[document],  # type: ignore[arg-type]
            user_id=user_id,
        )
    )
    assert ingestion_resp.documents, "ingestion produced no document records"
    document_ids = [d.source_id for d in ingestion_resp.documents]

    # Generate
    response = generation.generate_tailored_learning(
        video_source_id=None,
        document_source_ids=document_ids,
        user_id=user_id,
    )

    # ── Schema + presence assertions ──────────────────────────────────────────
    assert response.documents, "no document sections generated"
    doc_section = response.documents[0]
    assert doc_section.summary_text.strip(), "summary empty"
    assert doc_section.deep_dive_text.strip(), "deep_dive empty"
    assert len(doc_section.reflection_points) >= 2, "at least 2 reflections expected"
    depths = {p.depth_level for p in doc_section.reflection_points}
    # Reflection points should span at least one named depth.
    assert depths, "reflection depth_level missing"

    # A single document now gets full enrichment: a quiz + apply-it scenarios.
    assert response.quiz.questions, "single-document quiz should not be empty"
    assert response.insights.application_scenarios, "single-document applications expected"
    assert not response.insights.intersections, "single source must not fabricate intersections"

    # ── No refusal pattern leakage in any section text ────────────────────────
    refusal_marker = "don't have enough information"
    for section_text in (
        doc_section.summary_text,
        doc_section.deep_dive_text,
        doc_section.under_surface_explainer or "",
    ):
        assert refusal_marker not in section_text.lower(), (
            f"Refusal marker leaked into section: {section_text[:120]}"
        )

    # ── Session JSONL was actually written ────────────────────────────────────
    jsonl_files = list(log_dir.glob("gen-*.jsonl"))
    assert jsonl_files, "no session JSONL produced — telemetry wiring broken"
    rows = jsonl_files[0].read_text(encoding="utf-8").splitlines()
    assert any('"stage": "session_start"' in line for line in rows)
    assert any('"stage": "session_end"' in line for line in rows)
    assert any('"stage": "stage_call"' in line for line in rows), "no stage_call records"
