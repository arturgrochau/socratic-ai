"""Tests for the runtime settings API and config hot-swap behavior.

These avoid the full app startup (DB/chroma) by mounting only the settings
router, and isolate persistence via SOCRATIC_CONFIG_DIR.
"""
from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate persisted settings and provide a key so import doesn't raise.
    monkeypatch.setenv("SOCRATIC_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-abcd1234")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    import config
    importlib.reload(config)
    import routes.settings as settings_mod
    importlib.reload(settings_mod)

    app = FastAPI()
    app.include_router(settings_mod.router)
    return TestClient(app), config


def test_get_settings_masks_key(client):
    http, config = client
    resp = http.get("/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["openai_api_key_set"] is True
    # The full secret must never be serialized.
    assert "sk-test-secret-abcd1234" not in resp.text
    assert body["openai_api_key_masked"] == "sk-...1234"
    assert body["llm_provider"] == "openai"


def test_put_switches_provider_and_clears_cache(client):
    http, config = client
    # Prime the cache with the openai client.
    first = config.get_llm_client()
    assert type(first).__name__ == "OpenAIClient"

    payload = config.get_settings().__dict__.copy()
    payload.pop("openai_api_key", None)
    payload["llm_provider"] = "ollama"
    resp = http.put("/settings", json=payload)
    assert resp.status_code == 200
    assert resp.json()["llm_provider"] == "ollama"

    # Settings singleton updated and the cached client rebuilt for the new provider.
    assert config.get_settings().llm_provider == "ollama"
    assert type(config.get_llm_client()).__name__ == "OllamaClient"


def test_put_blank_key_preserves_existing(client):
    http, config = client
    before = config.get_settings().openai_api_key
    payload = config.get_settings().__dict__.copy()
    payload.pop("openai_api_key", None)  # omit => preserve
    payload["chat_model"] = "gpt-4o"
    resp = http.put("/settings", json=payload)
    assert resp.status_code == 200
    assert config.get_settings().openai_api_key == before
    assert resp.json()["chat_model"] == "gpt-4o"


def test_put_new_key_updates_and_masks(client):
    http, config = client
    payload = config.get_settings().__dict__.copy()
    payload["openai_api_key"] = "sk-brand-new-key-wxyz9876"
    resp = http.put("/settings", json=payload)
    assert resp.status_code == 200
    assert config.get_settings().openai_api_key == "sk-brand-new-key-wxyz9876"
    assert resp.json()["openai_api_key_masked"] == "sk-...9876"
    assert "sk-brand-new-key-wxyz9876" not in resp.text


def test_apply_settings_invalidates_caches(client):
    _http, config = client
    config.get_llm_client()  # populate cache
    # Force the reachability flag to a known value, then apply settings.
    config._aux_local_reachable = True
    assert config._llm_client_cache  # non-empty
    config.apply_settings(replace(config.get_settings(), generation_model="gpt-4o"))
    assert config._llm_client_cache == {}
    assert config._aux_local_reachable is None
    assert config.generation_model() == "gpt-4o"


def test_settings_persist_across_reload(client, tmp_path, monkeypatch):
    http, config = client
    payload = config.get_settings().__dict__.copy()
    payload.pop("openai_api_key", None)
    payload["llm_provider"] = "ollama"
    http.put("/settings", json=payload)

    # Reload config; the persisted choice (ollama) should win over the env default.
    importlib.reload(config)
    assert config.get_settings().llm_provider == "ollama"
