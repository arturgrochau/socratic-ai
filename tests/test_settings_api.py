"""Tests for the runtime settings API and config hot-swap behavior.

These avoid the full app startup (DB/chroma) by mounting only the settings
router, and isolate persistence via SOCRATIC_CONFIG_DIR.
"""
from __future__ import annotations

import importlib
from dataclasses import replace

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _daemon_up() -> patch:
    """Mock the Ollama reachability probe so tests don't need a live daemon."""
    ok = MagicMock()
    ok.raise_for_status.return_value = None
    ok.json.return_value = {"models": []}
    return patch("requests.get", return_value=ok)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate persisted settings; provide a key so API mode can be selected.
    monkeypatch.setenv("SOCRATIC_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-abcd1234")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("WHISPER_PROVIDER", raising=False)

    import config
    importlib.reload(config)
    import routes.settings as settings_mod
    importlib.reload(settings_mod)

    app = FastAPI()
    app.include_router(settings_mod.router)
    return TestClient(app), config


def test_local_by_default_and_masked_key(client):
    http, config = client
    resp = http.get("/settings")
    assert resp.status_code == 200
    body = resp.json()
    # Local is the default: no provider env set => everything on-device.
    assert body["mode"] == "local"
    assert body["llm_provider"] == "ollama"
    assert body["embedding_provider"] == "ollama"
    assert body["whisper_provider"] == "local"
    # The full secret must never be serialized.
    assert body["openai_api_key_set"] is True
    assert "sk-test-secret-abcd1234" not in resp.text
    assert body["openai_api_key_masked"] == "sk-...1234"


def test_mode_toggle_switches_all_providers_and_models(client):
    http, config = client
    # Local mode resolves the local bundle.
    assert config.generation_model() == config.get_settings().local_generation_model

    resp = http.put("/settings/mode", json={"mode": "api"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "api"
    assert body["llm_provider"] == "openai"
    assert body["whisper_provider"] == "openai"
    # The resolved models follow the mode — the known-backlog provider-switch
    # integration gap: a switched provider must be used downstream.
    assert config.generation_model() == config.get_settings().openai_generation_model
    assert type(config.get_llm_client()).__name__ == "OpenAIClient"

    with _daemon_up():
        resp = http.put("/settings/mode", json={"mode": "local"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "local"
    assert config.generation_model() == config.get_settings().local_generation_model
    assert type(config.get_llm_client()).__name__ == "OllamaClient"


def test_mode_local_requires_reachable_daemon(client):
    http, config = client
    http.put("/settings/mode", json={"mode": "api"})
    import requests

    with patch("requests.get", side_effect=requests.ConnectionError("refused")):
        resp = http.put("/settings/mode", json={"mode": "local"})
    assert resp.status_code == 422
    assert "not reachable" in resp.json()["detail"]
    assert config.get_settings().llm_provider == "openai"  # unchanged


def test_mode_api_requires_key(client):
    http, config = client
    # Drop the stored key, then try to switch to API mode.
    config.apply_settings(replace(config.get_settings(), openai_api_key=None), persist=False)
    resp = http.put("/settings/mode", json={"mode": "api"})
    assert resp.status_code == 422
    assert config.get_settings().llm_provider == "ollama"  # unchanged


def test_put_rejects_unknown_provider(client):
    http, _config = client
    resp = http.put("/settings", json={"llm_provider": "banana"})
    assert resp.status_code == 422


def test_put_partial_update_preserves_key_and_other_fields(client):
    http, config = client
    before = config.get_settings().openai_api_key
    resp = http.put("/settings", json={"openai_chat_model": "gpt-4o"})
    assert resp.status_code == 200
    assert config.get_settings().openai_api_key == before
    assert resp.json()["openai_chat_model"] == "gpt-4o"
    # Untouched fields keep their values.
    assert config.get_settings().llm_provider == "ollama"


def test_put_new_key_updates_and_masks(client):
    http, config = client
    resp = http.put("/settings", json={"openai_api_key": "sk-brand-new-key-wxyz9876"})
    assert resp.status_code == 200
    assert config.get_settings().openai_api_key == "sk-brand-new-key-wxyz9876"
    assert resp.json()["openai_api_key_masked"] == "sk-...9876"
    assert "sk-brand-new-key-wxyz9876" not in resp.text


def test_apply_settings_invalidates_caches(client):
    _http, config = client
    config.get_llm_client()  # populate cache
    config._aux_probe = (0.0, True)
    assert config._llm_client_cache  # non-empty
    config.apply_settings(
        replace(config.get_settings(), local_generation_model="qwen3:8b-q4_K_M")
    )
    assert config._llm_client_cache == {}
    assert config._aux_probe is None
    assert config.generation_model() == "qwen3:8b-q4_K_M"


def test_settings_persist_across_reload(client):
    http, config = client
    http.put("/settings/mode", json={"mode": "api"})

    # Reload config; the persisted choice (api mode) should win over defaults.
    importlib.reload(config)
    assert config.get_settings().llm_provider == "openai"
    assert config.current_mode() == "api"


def test_legacy_persisted_settings_migrate(client):
    http, config = client
    # Simulate a pre-2.3 settings.json (single-bundle field names).
    legacy = {
        "llm_provider": "ollama",
        "embedding_provider": "ollama",
        "whisper_provider": "ollama",
        "generation_model": "llama3.1:8b",
        "retrieval_model": "nomic-embed-text",
        "aux_local_model": "qwen3:4b-instruct-2507-q4_K_M",
        "aux_llm_provider": "ollama",
    }
    path = config._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(legacy), encoding="utf-8")
    importlib.reload(config)
    settings = config.get_settings()
    assert settings.local_generation_model == "llama3.1:8b"
    assert settings.local_retrieval_model == "nomic-embed-text"
    assert settings.local_aux_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert settings.whisper_provider == "local"  # legacy "ollama" alias
