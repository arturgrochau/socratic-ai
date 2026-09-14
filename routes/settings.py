from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from app import ollama_probe

router = APIRouter(tags=["settings"])

VALID_LLM_PROVIDERS = {"openai", "ollama"}
VALID_WHISPER_PROVIDERS = {"local", "openai", "none"}


def _mask_key(key: str | None) -> str:
    """Never expose the full API key. Show only a short fingerprint."""
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:3]}...{key[-4:]}"


class SettingsView(BaseModel):
    """Read view of the runtime settings. The API key is always masked."""

    mode: str
    llm_provider: str
    embedding_provider: str
    whisper_provider: str
    ollama_host: str
    openai_api_key_masked: str
    openai_api_key_set: bool
    openai_generation_model: str
    openai_chat_model: str
    openai_retrieval_model: str
    openai_aux_model: str
    local_generation_model: str
    local_chat_model: str
    local_retrieval_model: str
    local_aux_model: str
    aux_use_local: bool
    # Resolved active models, for display.
    generation_model: str
    chat_model: str
    retrieval_model: str


class SettingsUpdate(BaseModel):
    """Partial update: omitted/None fields keep their current value.
    `openai_api_key` left blank/None preserves the stored key (the UI never
    has to round-trip it)."""

    llm_provider: str | None = None
    embedding_provider: str | None = None
    whisper_provider: str | None = None
    ollama_host: str | None = None
    openai_api_key: str | None = None
    openai_generation_model: str | None = None
    openai_chat_model: str | None = None
    openai_retrieval_model: str | None = None
    openai_aux_model: str | None = None
    local_generation_model: str | None = None
    local_chat_model: str | None = None
    local_retrieval_model: str | None = None
    local_aux_model: str | None = None
    aux_use_local: bool | None = None


class ModeUpdate(BaseModel):
    mode: str  # "local" | "api"


def _view(settings: config.Settings) -> SettingsView:
    return SettingsView(
        mode=config.current_mode(),
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        whisper_provider=settings.whisper_provider,
        ollama_host=settings.ollama_host,
        openai_api_key_masked=_mask_key(settings.openai_api_key),
        openai_api_key_set=bool(settings.openai_api_key),
        openai_generation_model=settings.openai_generation_model,
        openai_chat_model=settings.openai_chat_model,
        openai_retrieval_model=settings.openai_retrieval_model,
        openai_aux_model=settings.openai_aux_model,
        local_generation_model=settings.local_generation_model,
        local_chat_model=settings.local_chat_model,
        local_retrieval_model=settings.local_retrieval_model,
        local_aux_model=settings.local_aux_model,
        aux_use_local=settings.aux_use_local,
        generation_model=config.generation_model(),
        chat_model=config.chat_model(),
        retrieval_model=config.retrieval_model(),
    )


def _validate(settings: config.Settings) -> None:
    if settings.llm_provider not in VALID_LLM_PROVIDERS:
        raise HTTPException(status_code=422, detail=f"Unknown llm_provider: {settings.llm_provider!r}")
    if settings.embedding_provider not in VALID_LLM_PROVIDERS:
        raise HTTPException(
            status_code=422, detail=f"Unknown embedding_provider: {settings.embedding_provider!r}"
        )
    if settings.whisper_provider not in VALID_WHISPER_PROVIDERS:
        raise HTTPException(
            status_code=422, detail=f"Unknown whisper_provider: {settings.whisper_provider!r}"
        )
    non_empty = {
        "ollama_host": settings.ollama_host,
        "openai_generation_model": settings.openai_generation_model,
        "openai_chat_model": settings.openai_chat_model,
        "openai_retrieval_model": settings.openai_retrieval_model,
        "openai_aux_model": settings.openai_aux_model,
        "local_generation_model": settings.local_generation_model,
        "local_chat_model": settings.local_chat_model,
        "local_retrieval_model": settings.local_retrieval_model,
        "local_aux_model": settings.local_aux_model,
    }
    blank = sorted(name for name, value in non_empty.items() if not (value or "").strip())
    if blank:
        raise HTTPException(
            status_code=422, detail=f"These fields cannot be empty: {', '.join(blank)}"
        )
    needs_key = (
        settings.llm_provider == "openai"
        or settings.embedding_provider == "openai"
        or settings.whisper_provider == "openai"
    )
    if needs_key and not settings.openai_api_key:
        raise HTTPException(
            status_code=422,
            detail="An OpenAI API key is required for the selected providers. "
            "Add a key, or switch to Local mode.",
        )


@router.get("/settings", response_model=SettingsView)
def read_settings() -> SettingsView:
    return _view(config.get_settings())


@router.put("/settings", response_model=SettingsView)
def update_settings(body: SettingsUpdate) -> SettingsView:
    current = config.get_settings()
    fields = {
        key: value
        for key, value in body.model_dump().items()
        if value is not None and key != "openai_api_key"
    }
    incoming_key = body.openai_api_key
    # Blank/None => keep the existing key.
    new_key = (
        incoming_key.strip()
        if isinstance(incoming_key, str) and incoming_key.strip()
        else current.openai_api_key
    )
    new = replace(current, openai_api_key=new_key, **fields)
    _validate(new)
    config.apply_settings(new)
    return _view(config.get_settings())


@router.put("/settings/mode", response_model=SettingsView)
def update_mode(body: ModeUpdate) -> SettingsView:
    """One-click Local ↔ API switch. Flips every provider at once while both
    model bundles keep their choices."""
    mode = (body.mode or "").strip().lower()
    try:
        new = config.settings_for_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _validate(new)
    if mode == "local":
        # Local mode without a reachable daemon would fail on every later call
        # with opaque connection errors — reject the switch with the reason.
        try:
            ollama_probe.list_models(new.ollama_host, timeout=2)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Ollama is not reachable at {new.ollama_host} — install/start it "
                f"(https://ollama.com) or stay in API mode. ({str(exc)[:120]})",
            ) from exc
    config.apply_settings(new)
    return _view(config.get_settings())


@router.get("/settings/ollama-models")
def list_ollama_models() -> dict[str, object]:
    """Models available on the local daemon, for the Settings dropdowns."""
    settings = config.get_settings()
    try:
        models = sorted(ollama_probe.list_models(settings.ollama_host, timeout=3))
        return {"ok": True, "models": models}
    except Exception as exc:  # noqa: BLE001 — surface the probe failure to the UI
        return {"ok": False, "models": [], "detail": str(exc)[:300]}


@router.get("/settings/test")
def test_connection(provider: str) -> dict[str, object]:
    """Lightweight connectivity probe for a provider, used by the Settings panel."""
    provider = (provider or "").strip().lower()
    settings = config.get_settings()
    try:
        if provider == "ollama":
            models: set[str] = set()
            for name in ollama_probe.list_models(settings.ollama_host, timeout=3):
                models.add(name)
                if name.endswith(":latest"):  # "foo" and "foo:latest" are the same model
                    models.add(name[: -len(":latest")])
            configured = {
                settings.local_generation_model,
                settings.local_chat_model,
                settings.local_retrieval_model,
                settings.local_aux_model,
            }
            missing = sorted(m for m in configured if m and m not in models)
            if missing:
                return {
                    "ok": False,
                    "provider": provider,
                    "detail": f"Daemon up, but configured model(s) not pulled: {', '.join(missing)}",
                }
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
