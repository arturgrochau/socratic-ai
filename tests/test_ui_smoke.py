"""In-process smoke test of the NiceGUI pages.

Uses NiceGUI's `user` fixture (loaded via -p nicegui.testing.user_plugin) to
build each page through the real connected-render path — this catches runtime
errors in the wizard / dashboard / chat / settings builders that a plain HTTP
GET (which only returns the shell) would miss. main.py is the page source.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-ui-smoke")
os.environ.setdefault("SOCRATIC_CONFIG_DIR", "/tmp/socratic-ui-smoke-cfg")

import pytest

pytestmark = pytest.mark.asyncio


async def test_index_renders_step_one(user) -> None:
    await user.open("/")
    await user.should_see("Step 1 of 2")
    await user.should_see("Skip video →")


async def test_skip_video_requires_document(user) -> None:
    await user.open("/")
    await user.should_see("Skip video →")  # wait for the connected render
    user.find("Skip video →").click()
    await user.should_see("Step 2 of 2")
    # With no video, a document is mandatory until one is uploaded.
    await user.should_see("Upload a document above to enable Generate.")


async def test_document_upload_lands_and_enables_generate(user, monkeypatch) -> None:
    # Regression guard for the NiceGUI 3.x upload-event API AND the upload-on-add
    # flow: a real upload must ingest (mocked here, since the in-process User
    # simulation can't reach the loopback HTTP backend) and flip Generate on.
    from nicegui import ui
    from nicegui.elements.upload_files import SmallFileUpload

    from frontend import api_client

    async def _fake_upload_document(*, name: str, mime_type: str, data: bytes) -> dict:
        return {"name": name, "source_id": 1}

    monkeypatch.setattr(api_client, "upload_document", _fake_upload_document)

    await user.open("/")
    await user.should_see("Skip video →")
    user.find("Skip video →").click()
    await user.should_see("Step 2 of 2")

    upload_el = next(iter(user.find(ui.upload).elements))
    await upload_el.handle_uploads(
        [SmallFileUpload(name="notes.pdf", content_type="application/pdf", _data=b"hello world")]
    )
    await user.should_see("notes.pdf")          # saved row appears
    await user.should_see("Estimated time")     # Generate is now enabled (hint flips)


@pytest.mark.nicegui_main_file("tests/_ui_quiz_probe.py")
async def test_quiz_lock_and_correction(user) -> None:
    await user.open("/quizprobe")
    await user.should_see("ZZQUESTION about coupling?")
    await user.should_see("alphaOPT")
    # Pick the wrong option (alpha). First pick locks the question.
    user.find("alphaOPT").click()
    # The chosen-wrong correction and the correct-answer rationale both appear.
    await user.should_see("TRAPALPHA")
    await user.should_see("RIGHTBETA")
    # Single question answered => the handoff button is offered.
    await user.should_see("Discuss in Socratic chat")


async def test_settings_page_renders(user) -> None:
    # The settings form fields load from the API over HTTP, which the in-process
    # User simulation can't reach — so we only assert the page builder runs and
    # paints its header/intro (the endpoint itself is covered in test_settings_api).
    await user.open("/settings")
    await user.should_see("Settings")
    await user.should_see("Local mode")


def _report(**overrides) -> dict:
    base = {
        "platform": "darwin", "arch": "arm64", "ram_gb": 16.0,
        "ffmpeg": {"found": False, "path": None, "version": None},
        "ollama": {"installed": False, "running": False, "host": "http://localhost:11434", "models": [], "detail": "refused"},
        "mlx_available": True,
        "recommended": {"name": "standard", "generation_model": "qwen3:8b", "embedding_model": "nomic-embed-text",
                        "download_gb": 5.5, "num_ctx": 8192, "note": ""},
        "missing_models": ["qwen3:8b", "nomic-embed-text"],
        "setup_complete": False, "mode": "local",
    }
    base.update(overrides)
    return base


async def test_index_redirects_to_welcome_until_setup_is_complete(user, monkeypatch) -> None:
    from frontend import api_client

    async def _status() -> dict:
        return _report()

    monkeypatch.setattr(api_client, "get_setup_status", _status)
    await user.open("/")
    await user.should_see("Welcome to Socratic AI")
    await user.should_see("qwen3:8b")           # the RAM-tiered recommendation is shown
    await user.should_see("5.5 GB")


async def test_welcome_local_checklist_shows_fixes(user, monkeypatch) -> None:
    from frontend import api_client

    async def _status() -> dict:
        return _report()

    monkeypatch.setattr(api_client, "get_setup_status", _status)
    await user.open("/welcome")
    await user.should_see("Continue")
    user.find("Continue").click()
    await user.should_see("Ollama not running")
    await user.should_see("Download Ollama for macOS")   # not installed -> install link
    await user.should_see("2 model(s) to download")
    await user.should_see("ffmpeg missing")
    await user.should_see("brew install ffmpeg")


async def test_welcome_ready_machine_offers_download_button(user, monkeypatch) -> None:
    from frontend import api_client

    async def _status() -> dict:
        return _report(
            ffmpeg={"found": True, "path": "/opt/homebrew/bin/ffmpeg", "version": "7.1"},
            ollama={"installed": True, "running": True, "host": "http://localhost:11434",
                    "models": ["nomic-embed-text:latest"], "detail": ""},
            missing_models=["qwen3:8b"],
        )

    monkeypatch.setattr(api_client, "get_setup_status", _status)
    await user.open("/welcome")
    await user.should_see("Continue")
    user.find("Continue").click()
    await user.should_see("Ollama running")
    await user.should_see("Download 5.5 GB")
    await user.should_see("ffmpeg found")
