"""First-run setup logic: RAM tiers, environment detection, model pulls."""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from app import setup


@pytest.mark.parametrize(
    "ram, expected",
    [
        (None, "qwen3:8b"),        # unknown hardware: the safe middle
        (8.0, "qwen3:4b"),
        (11.9, "qwen3:4b"),
        (12.0, "qwen3:8b"),
        (16.0, "qwen3:8b"),
        (23.9, "qwen3:8b"),
        (24.0, "qwen3:14b"),
        (31.9, "qwen3:14b"),
        (32.0, "qwen3:30b-a3b-instruct-2507-q4_K_M"),
        (128.0, "qwen3:30b-a3b-instruct-2507-q4_K_M"),
    ],
)
def test_recommend_tier_boundaries(ram, expected) -> None:
    assert setup.recommend_tier(ram).generation_model == expected


def test_small_machines_get_a_smaller_context_window() -> None:
    assert setup.recommend_tier(16.0).num_ctx == 8192
    assert setup.recommend_tier(64.0).num_ctx == 16384


def test_detect_environment_reports_missing_pieces(monkeypatch) -> None:
    monkeypatch.setattr(setup, "detect_ram_gb", lambda: 16.0)
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup, "OLLAMA_APP", setup.Path("/nonexistent/Ollama.app"))
    with patch("app.ollama_probe.list_models", side_effect=ConnectionError("refused")):
        report = setup.detect_environment()
    assert report.ffmpeg.found is False
    assert report.ollama.installed is False and report.ollama.running is False
    assert report.recommended.generation_model == "qwen3:8b"
    # Nothing can be verified while the daemon is down: everything is "missing".
    assert report.missing_models == ["qwen3:8b", "nomic-embed-text"]


def test_detect_environment_treats_latest_tag_as_present(monkeypatch) -> None:
    monkeypatch.setattr(setup, "detect_ram_gb", lambda: 48.0)
    with patch("app.ollama_probe.list_models", return_value=["nomic-embed-text:latest", "qwen3:1.7b"]):
        report = setup.detect_environment()
    assert report.ollama.running is True
    assert report.missing_models == ["qwen3:30b-a3b-instruct-2507-q4_K_M"]


def _fake_pull_transport(lines: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull"
        body = "".join(json.dumps(line) + "\n" for line in lines)
        return httpx.Response(200, content=body.encode())

    return httpx.MockTransport(handler)


def test_pull_job_reports_progress_and_completion() -> None:
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling abc", "total": 1000, "completed": 250},
        {"status": "pulling abc", "total": 1000, "completed": 1000},
        {"status": "verifying sha256 digest"},
        {"status": "success"},
    ]
    job = setup.PullJob(job_id="t1", models=["qwen3:8b"])
    setup._run_pull(job, "http://test:11434", client=httpx.Client(transport=_fake_pull_transport(lines)))
    assert job.status == "done"
    assert job.percent == 100.0
    assert job.finished == ["qwen3:8b"]
    assert job.error is None


def test_pull_job_surfaces_daemon_errors() -> None:
    transport = _fake_pull_transport([{"error": "pull model manifest: file does not exist"}])
    job = setup.PullJob(job_id="t2", models=["nope:latest"])
    setup._run_pull(job, "http://test:11434", client=httpx.Client(transport=transport))
    assert job.status == "failed"
    assert "does not exist" in (job.error or "")
    assert job.finished == []


def test_complete_setup_applies_tier_only_on_first_run(monkeypatch, tmp_path) -> None:
    import importlib

    import config

    monkeypatch.setenv("SOCRATIC_CONFIG_DIR", str(tmp_path / "cfg"))
    importlib.reload(config)
    importlib.reload(setup)
    monkeypatch.setattr(setup, "detect_ram_gb", lambda: 16.0)

    settings = setup.complete_setup("local")
    assert settings.setup_complete is True
    assert settings.local_generation_model == "qwen3:8b"
    assert settings.local_num_ctx == 8192
    assert config.ollama_num_ctx() == 8192

    # Re-running setup later must not clobber a model the user picked by hand.
    from dataclasses import replace

    config.apply_settings(replace(config.get_settings(), local_generation_model="qwen3:14b"))
    again = setup.complete_setup("local")
    assert again.local_generation_model == "qwen3:14b"
