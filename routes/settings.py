from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config


router = APIRouter(tags=["settings"])


def _mask_key(key: str | None) -> str:
    """Never expose the full API key. Show only a short fingerprint."""
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:3]}...{key[-4:]}"


class SettingsView(BaseModel):
    """Read view of the runtime settings. The API key is always masked."""

    llm_provider: str
    embedding_provider: str
    whisper_provider: str
    ollama_host: str
    openai_api_key_masked: str
    openai_api_key_set: bool
    generation_model: str
    chat_model: str
    retrieval_model: str
    aux_use_local: bool
    aux_llm_provider: str
    aux_model: str
    aux_local_model: str


class SettingsUpdate(BaseModel):
    """Mutable settings. `openai_api_key` is optional — leave blank/None to
    preserve the currently stored key (so the UI never has to round-trip it)."""

    llm_provider: str
    embedding_provider: str
    whisper_provider: str
    ollama_host: str
    openai_api_key: str | None = None
    generation_model: str
    chat_model: str
    retrieval_model: str
    aux_use_local: bool
    aux_llm_provider: str
    aux_model: str
    aux_local_model: str


def _view(settings: config.Settings) -> SettingsView:
    return SettingsView(
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        whisper_provider=settings.whisper_provider,
        ollama_host=settings.ollama_host,
        openai_api_key_masked=_mask_key(settings.openai_api_key),
        openai_api_key_set=bool(settings.openai_api_key),
        generation_model=settings.generation_model,
        chat_model=settings.chat_model,
        retrieval_model=settings.retrieval_model,
        aux_use_local=settings.aux_use_local,
        aux_llm_provider=settings.aux_llm_provider,
        aux_model=settings.aux_model,
        aux_local_model=settings.aux_local_model,
    )


@router.get("/settings", response_model=SettingsView)
def read_settings() -> SettingsView:
    return _view(config.get_settings())


@router.put("/settings", response_model=SettingsView)
def update_settings(body: SettingsUpdate) -> SettingsView:
    current = config.get_settings()
    fields = body.model_dump()
    incoming_key = fields.pop("openai_api_key", None)
    # Blank/None => keep the existing key.
    new_key = incoming_key.strip() if isinstance(incoming_key, str) and incoming_key.strip() else current.openai_api_key
    new = replace(current, openai_api_key=new_key, **fields)
    config.apply_settings(new)
    return _view(config.get_settings())


@router.get("/settings/test")
def test_connection(provider: str) -> dict[str, object]:
    """Lightweight connectivity probe for a provider, used by the Settings panel."""
    provider = (provider or "").strip().lower()
    settings = config.get_settings()
    try:
        if provider == "ollama":
            import requests

            resp = requests.get(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=3)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {"ok": True, "provider": provider, "detail": f"{len(models)} model(s) available"}
        if provider == "openai":
            if not settings.openai_api_key:
                return {"ok": False, "provider": provider, "detail": "No API key configured."}
            client = config.get_llm_client("openai")
            client.raw.models.list()
            return {"ok": True, "provider": provider, "detail": "Authenticated."}
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any probe failure to the UI
        return {"ok": False, "provider": provider, "detail": str(exc)[:300]}
