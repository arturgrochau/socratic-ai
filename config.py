from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


# ── Static infrastructure config (NOT runtime-switchable) ──────────────────────
# These stay module-level globals because tests reload `config` and read
# `config.db_engine` directly, and reload-order assumptions depend on them
# existing at import time. Only provider/model/key/host settings are mutable.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
USER_ID_HEADER = os.getenv("USER_ID_HEADER", "X-User-ID")
ENABLE_COST_LOGGING = _env_bool("ENABLE_COST_LOGGING", True)
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "production")
WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS = max(
    30,
    int(os.getenv("WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS", "120")),
)
WHISPER_TRANSCRIPTION_MAX_RETRIES = max(
    1,
    int(os.getenv("WHISPER_TRANSCRIPTION_MAX_RETRIES", "2")),
)
INGESTION_VIDEO_STEP_TIMEOUT_SECONDS = max(
    WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS,
    int(os.getenv("INGESTION_VIDEO_STEP_TIMEOUT_SECONDS", "240")),
)
# CACHE_PROCESSED_SOURCES is read once at generation time; keep it static.
CACHE_PROCESSED_SOURCES = _env_bool("CACHE_PROCESSED_SOURCES", True)


# ── Runtime-switchable provider/model settings ────────────────────────────────
@dataclass
class Settings:
    """Provider/model/key/host config that the in-app Settings panel can change
    at runtime. Mutated only via apply_settings() under a lock, which also
    invalidates the cached LLM clients and the Ollama reachability probe."""

    llm_provider: str = "openai"
    embedding_provider: str = "openai"
    whisper_provider: str = "openai"
    ollama_host: str = "http://localhost:11434"
    openai_api_key: str | None = None
    generation_model: str = "gpt-4o-mini"
    chat_model: str = "gpt-4o-mini"
    retrieval_model: str = "text-embedding-3-small"
    aux_use_local: bool = False
    aux_llm_provider: str = "openai"
    aux_model: str = "gpt-4o-mini"
    aux_local_model: str = "llama3.1:8b"


# Fields that may be persisted to / loaded from the user config file.
_PERSISTED_FIELDS = tuple(Settings.__dataclass_fields__.keys())


def _config_path() -> Path:
    """Cross-platform path to the persisted settings file.

    Honors SOCRATIC_CONFIG_DIR so tests/CI can isolate (and avoid picking up a
    developer's saved settings)."""
    override = os.getenv("SOCRATIC_CONFIG_DIR")
    if override:
        base = Path(override)
    else:
        try:
            from platformdirs import user_config_dir

            base = Path(user_config_dir("socratic-ai"))
        except Exception:
            base = Path.home() / ".config" / "socratic-ai"
    return base / "settings.json"


def _settings_from_env() -> Settings:
    """Build settings from dataclass defaults overlaid with environment vars.

    Mirrors the historical env names so existing .env files / CI keep working."""
    defaults = Settings()
    llm_provider = os.getenv("LLM_PROVIDER", defaults.llm_provider).strip().lower()
    aux_use_local = _env_bool("AUX_USE_LOCAL", defaults.aux_use_local)
    generation_model = os.getenv("GENERATION_MODEL", defaults.generation_model).strip()
    return Settings(
        llm_provider=llm_provider,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", llm_provider).strip().lower(),
        whisper_provider=os.getenv("WHISPER_PROVIDER", defaults.whisper_provider).strip().lower(),
        ollama_host=os.getenv("OLLAMA_HOST", defaults.ollama_host).strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"),
        generation_model=generation_model,
        chat_model=os.getenv("CHAT_MODEL", os.getenv("INTERACTION_MODEL", defaults.chat_model)).strip(),
        retrieval_model=os.getenv("RETRIEVAL_MODEL", defaults.retrieval_model).strip(),
        aux_use_local=aux_use_local,
        aux_llm_provider=os.getenv(
            "AUX_LLM_PROVIDER", "ollama" if aux_use_local else llm_provider
        ).strip().lower(),
        aux_model=os.getenv("AUX_MODEL", generation_model).strip(),
        aux_local_model=os.getenv("AUX_LOCAL_MODEL", defaults.aux_local_model).strip(),
    )


def _overlay_persisted(base: Settings) -> Settings:
    """Overlay user-saved settings (from the in-app panel) on top of env values.

    Persisted choices win because they reflect explicit user intent."""
    path = _config_path()
    if not path.exists():
        return base
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return base
    if not isinstance(stored, dict):
        return base
    overrides = {k: v for k, v in stored.items() if k in _PERSISTED_FIELDS}
    return replace(base, **overrides) if overrides else base


def persist_settings(settings: Settings) -> None:
    """Atomically write settings to the user config file (perms 600)."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


_settings_lock = threading.RLock()
_settings: Settings = _overlay_persisted(_settings_from_env())


def get_settings() -> Settings:
    """Current runtime settings. Treat the returned object as read-only;
    use apply_settings() to change anything."""
    return _settings


# Backwards-compatible module constants. Derived from the resolved settings so
# `from config import GENERATION_MODEL` keeps working, and `importlib.reload`
# rebuilds them from env. New code should use the accessor functions below so
# runtime changes take effect.
OPENAI_API_KEY = _settings.openai_api_key
RETRIEVAL_MODEL = _settings.retrieval_model
GENERATION_MODEL = _settings.generation_model
CHAT_MODEL = _settings.chat_model
LLM_PROVIDER = _settings.llm_provider
EMBEDDING_PROVIDER = _settings.embedding_provider
WHISPER_PROVIDER = _settings.whisper_provider
OLLAMA_HOST = _settings.ollama_host
AUX_USE_LOCAL = _settings.aux_use_local
AUX_LLM_PROVIDER = _settings.aux_llm_provider
AUX_MODEL = _settings.aux_model
AUX_LOCAL_MODEL = _settings.aux_local_model


def generation_model() -> str:
    return _settings.generation_model


def chat_model() -> str:
    return _settings.chat_model


def retrieval_model() -> str:
    return _settings.retrieval_model


def embedding_provider() -> str:
    return _settings.embedding_provider


# Only OpenAI strictly requires the API key. Local providers can run without it.
_needs_openai_key = (
    _settings.llm_provider == "openai"
    or _settings.embedding_provider == "openai"
    or _settings.whisper_provider == "openai"
)
if _needs_openai_key and not _settings.openai_api_key:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in .env before starting the server, "
        "or set LLM_PROVIDER=ollama (and EMBEDDING_PROVIDER=ollama, WHISPER_PROVIDER=none) "
        "to run fully against a local Ollama daemon."
    )

# Kept for backwards compatibility — ingestion (transcription) and the test
# cassette still reference openai_client directly. Recomputed by apply_settings.
openai_client = OpenAI(api_key=_settings.openai_api_key) if _settings.openai_api_key else None  # type: ignore[assignment]
db_engine = create_engine(DATABASE_URL, future=True)


_llm_client_cache: dict[str, object] = {}


def get_llm_client(provider: str | None = None):
    """Return a cached LLMClient for the requested provider (defaults to the
    current llm_provider setting)."""
    from app.llm_client import build_client

    with _settings_lock:
        key = (provider or _settings.llm_provider).strip().lower()
        cached = _llm_client_cache.get(key)
        if cached is None:
            cached = build_client(key, settings=_settings)
            _llm_client_cache[key] = cached
        return cached


_aux_local_reachable: bool | None = None


def _local_aux_reachable() -> bool:
    """Cheap, cached one-shot reachability probe for the local Ollama daemon."""
    global _aux_local_reachable
    if _aux_local_reachable is not None:
        return _aux_local_reachable
    try:
        import requests

        requests.get(
            f"{_settings.ollama_host.rstrip('/')}/api/tags", timeout=1.5
        ).raise_for_status()
        _aux_local_reachable = True
    except Exception:
        _aux_local_reachable = False
    return _aux_local_reachable


def get_aux_client():
    """Client for high-volume auxiliary passes.

    Routes to the local Ollama daemon when aux_use_local is set and the daemon
    is reachable; otherwise falls back to the default hosted client."""
    if _settings.aux_use_local and _settings.aux_llm_provider == "ollama" and _local_aux_reachable():
        return get_llm_client("ollama")
    return get_llm_client()


def get_aux_model() -> str:
    """Model name to pair with get_aux_client()."""
    if _settings.aux_use_local and _settings.aux_llm_provider == "ollama" and _local_aux_reachable():
        return _settings.aux_local_model
    return _settings.aux_model


def apply_settings(new: Settings, *, persist: bool = True) -> Settings:
    """Swap the runtime settings, invalidate caches, and (by default) persist.

    Thread-safe: an in-flight request either sees the old client fully or the
    new one fully — never a torn read."""
    global _settings, _aux_local_reachable, openai_client
    global OPENAI_API_KEY, RETRIEVAL_MODEL, GENERATION_MODEL, CHAT_MODEL
    global LLM_PROVIDER, EMBEDDING_PROVIDER, WHISPER_PROVIDER, OLLAMA_HOST
    global AUX_USE_LOCAL, AUX_LLM_PROVIDER, AUX_MODEL, AUX_LOCAL_MODEL
    with _settings_lock:
        _settings = new
        _llm_client_cache.clear()
        _aux_local_reachable = None
        openai_client = OpenAI(api_key=new.openai_api_key) if new.openai_api_key else None  # type: ignore[assignment]
        # Keep the backwards-compat module constants coherent with the new state.
        OPENAI_API_KEY = new.openai_api_key
        RETRIEVAL_MODEL = new.retrieval_model
        GENERATION_MODEL = new.generation_model
        CHAT_MODEL = new.chat_model
        LLM_PROVIDER = new.llm_provider
        EMBEDDING_PROVIDER = new.embedding_provider
        WHISPER_PROVIDER = new.whisper_provider
        OLLAMA_HOST = new.ollama_host
        AUX_USE_LOCAL = new.aux_use_local
        AUX_LLM_PROVIDER = new.aux_llm_provider
        AUX_MODEL = new.aux_model
        AUX_LOCAL_MODEL = new.aux_local_model
        if persist:
            persist_settings(new)
        return _settings


Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def run_startup_checks() -> None:
    with db_engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    chroma_client.list_collections()
