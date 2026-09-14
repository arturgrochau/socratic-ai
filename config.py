from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from platformdirs import user_data_dir
from sqlalchemy import create_engine, text


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def data_dir() -> Path:
    """Where runtime state lives: SQLite DB, Chroma vectors, uploads, logs.

    Defaults to the per-user application data dir (macOS:
    ~/Library/Application Support/socratic-ai). SOCRATIC_DATA_DIR overrides it;
    tests and Docker set it explicitly. Never next to the code: inside a
    frozen .app bundle the code dir is read-only and cwd is `/`."""
    override = os.getenv("SOCRATIC_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir("socratic-ai"))


# A .env in the working directory is a developer convenience; a frozen app has
# no meaningful cwd, so it reads only the one in the data dir (which lets .app
# users set env overrides without a terminal).
if not _is_frozen():
    load_dotenv()
DATA_DIR = data_dir()
load_dotenv(DATA_DIR / ".env")

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


# ── Static infrastructure config (NOT runtime-switchable) ──────────────────────
# These stay module-level globals because tests reload `config` and read
# `config.db_engine` directly, and reload-order assumptions depend on them
# existing at import time. Only provider/model/key/host settings are mutable.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_data"))
UPLOAD_ROOT = Path(os.getenv("SOCRATIC_UPLOAD_DIR", str(DATA_DIR / "uploads")))
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
# Local transcription of a long lecture is CPU/GPU-bound, not network-bound;
# the cap exists to catch hangs, not to police normal runs.
INGESTION_VIDEO_STEP_TIMEOUT_SECONDS = max(
    WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS,
    int(os.getenv("INGESTION_VIDEO_STEP_TIMEOUT_SECONDS", "900")),
)
# CACHE_PROCESSED_SOURCES is read once at generation time; keep it static.
CACHE_PROCESSED_SOURCES = _env_bool("CACHE_PROCESSED_SOURCES", True)

# Concurrent in-flight LLM calls per provider. A hosted API absorbs wide
# fan-out; a local Ollama daemon serves OLLAMA_NUM_PARALLEL slots (default 1)
# and queues the rest, so submitting wide just risks queue-wait timeouts.
HOSTED_MAX_PARALLEL = max(1, int(os.getenv("SOCRATIC_HOSTED_PARALLEL", "8")))
LOCAL_MAX_PARALLEL = max(1, int(os.getenv("SOCRATIC_LOCAL_PARALLEL", "2")))

# Context window requested from Ollama per call tier. The daemon default (4096)
# silently truncates long prompts head-first, which eats the system prompt.
# Both tiers default to the SAME size on purpose: a request with a larger
# num_ctx than the loaded runner forces a new runner load mid-pipeline, and on
# a machine where other models are resident the scheduler can stall
# indefinitely waiting for memory (observed live). 16k covers the largest
# pipeline prompt (~9k tokens) with headroom; raise both via env on machines
# with free memory (a smaller request reuses a larger loaded runner).
OLLAMA_NUM_CTX = max(4096, int(os.getenv("SOCRATIC_OLLAMA_NUM_CTX", "16384")))
OLLAMA_NUM_CTX_LARGE = max(OLLAMA_NUM_CTX, int(os.getenv("SOCRATIC_OLLAMA_NUM_CTX_LARGE", "16384")))
OLLAMA_KEEP_ALIVE = os.getenv("SOCRATIC_OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TIMEOUT_SECONDS = max(60, int(os.getenv("SOCRATIC_OLLAMA_TIMEOUT_SECONDS", "600")))


# Default model bundles. Local defaults target the models present on an
# Apple-Silicon dev machine; the Settings page lists what the daemon actually
# has. One model for every local role is deliberate: the 30b-a3b MoE decodes
# as fast as a 4B dense model, and a single resident model avoids load thrash.
LOCAL_GENERATION_MODEL_DEFAULT = os.getenv(
    "SOCRATIC_LOCAL_GENERATION_DEFAULT", "qwen3:30b-a3b-instruct-2507-q4_K_M"
)
LOCAL_EMBEDDING_MODEL_DEFAULT = os.getenv(
    "SOCRATIC_LOCAL_EMBEDDING_DEFAULT", "nomic-embed-text"
)


# ── Runtime-switchable provider/model settings ────────────────────────────────
@dataclass
class Settings:
    """Provider/model/key/host config that the in-app Settings panel can change
    at runtime. Mutated only via apply_settings() under a lock, which also
    invalidates the cached LLM clients and the Ollama reachability probe.

    Two model bundles live side by side (openai_* and local_*); the *_provider
    fields select which bundle each role resolves to. The Local/API mode toggle
    only flips providers, so each mode keeps its own model choices."""

    llm_provider: str = "ollama"
    embedding_provider: str = "ollama"
    whisper_provider: str = "local"  # local | openai | none
    ollama_host: str = "http://localhost:11434"
    openai_api_key: str | None = None
    # Hosted (OpenAI) bundle — used by roles whose provider is "openai".
    openai_generation_model: str = "gpt-4o-mini"
    openai_chat_model: str = "gpt-4o-mini"
    openai_retrieval_model: str = "text-embedding-3-small"
    openai_aux_model: str = "gpt-4o-mini"
    # Local (Ollama) bundle — used by roles whose provider is "ollama".
    local_generation_model: str = LOCAL_GENERATION_MODEL_DEFAULT
    local_chat_model: str = LOCAL_GENERATION_MODEL_DEFAULT
    local_retrieval_model: str = LOCAL_EMBEDDING_MODEL_DEFAULT
    local_aux_model: str = LOCAL_GENERATION_MODEL_DEFAULT
    # API mode only: route the high-volume aux passes (ledger extraction) to a
    # local Ollama daemon when one is reachable. Moot in local mode.
    aux_use_local: bool = False


# Fields that may be persisted to / loaded from the user config file.
_PERSISTED_FIELDS = tuple(Settings.__dataclass_fields__.keys())

# Legacy persisted/env single-bundle field names → how to map them onto the
# dual-bundle layout (resolved against the file's own provider choice).
_LEGACY_WHISPER_ALIASES = {"ollama": "local"}


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


def storage_secret() -> str:
    """Per-install secret for NiceGUI's browser storage cookie.

    Generated once and kept next to settings.json. SOCRATIC_STORAGE_SECRET
    still overrides it (Docker / multi-instance deploys)."""
    override = os.getenv("SOCRATIC_STORAGE_SECRET")
    if override:
        return override
    path = _config_path().with_name("storage_secret")
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    import secrets

    value = secrets.token_urlsafe(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError:
        pass  # read-only config dir: fall back to a per-process secret
    return value


def _normalize_whisper(value: str) -> str:
    value = (value or "").strip().lower()
    return _LEGACY_WHISPER_ALIASES.get(value, value) or "local"


def _settings_from_env() -> Settings:
    """Build settings from dataclass defaults overlaid with environment vars.

    Mirrors the historical env names so existing .env files / CI keep working:
    GENERATION_MODEL / CHAT_MODEL / RETRIEVAL_MODEL apply to the bundle that
    the corresponding provider selects."""
    defaults = Settings()
    llm_provider = os.getenv("LLM_PROVIDER", defaults.llm_provider).strip().lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", llm_provider).strip().lower()
    settings = replace(
        defaults,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        whisper_provider=_normalize_whisper(
            os.getenv("WHISPER_PROVIDER", defaults.whisper_provider)
        ),
        ollama_host=os.getenv("OLLAMA_HOST", defaults.ollama_host).strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"),
        openai_aux_model=os.getenv("AUX_MODEL", defaults.openai_aux_model).strip(),
        local_aux_model=os.getenv("AUX_LOCAL_MODEL", defaults.local_aux_model).strip(),
        aux_use_local=_env_bool("AUX_USE_LOCAL", defaults.aux_use_local),
    )

    def _bundle(model: str, provider: str, role: str) -> str:
        # Route the override into the bundle the model actually belongs to:
        # pre-2.3 .env files set OpenAI model names without LLM_PROVIDER (openai
        # was the default then); those must not land in the local bundle.
        looks_hosted = model.startswith(("gpt-", "o1", "o3", "o4", "text-embedding"))
        target_local = provider == "ollama" and not looks_hosted
        return f"local_{role}" if target_local else f"openai_{role}"

    generation_model = os.getenv("GENERATION_MODEL", "").strip()
    chat_model = os.getenv("CHAT_MODEL", os.getenv("INTERACTION_MODEL", "")).strip()
    retrieval_model = os.getenv("RETRIEVAL_MODEL", "").strip()
    overrides: dict[str, str] = {}
    if generation_model:
        overrides[_bundle(generation_model, llm_provider, "generation_model")] = generation_model
    if chat_model:
        overrides[_bundle(chat_model, llm_provider, "chat_model")] = chat_model
    if retrieval_model:
        overrides[_bundle(retrieval_model, embedding_provider, "retrieval_model")] = retrieval_model
    return replace(settings, **overrides) if overrides else settings


def _migrate_legacy_stored(stored: dict) -> dict:
    """Translate a pre-dual-bundle settings.json into current field names."""
    migrated = dict(stored)
    llm_provider = str(stored.get("llm_provider", "")).strip().lower()
    embedding_provider = str(stored.get("embedding_provider", llm_provider)).strip().lower()
    if "whisper_provider" in migrated:
        migrated["whisper_provider"] = _normalize_whisper(str(migrated["whisper_provider"]))
    legacy_map = {
        "generation_model": (
            "local_generation_model" if llm_provider == "ollama" else "openai_generation_model"
        ),
        "chat_model": "local_chat_model" if llm_provider == "ollama" else "openai_chat_model",
        "retrieval_model": (
            "local_retrieval_model" if embedding_provider == "ollama" else "openai_retrieval_model"
        ),
        "aux_model": "openai_aux_model",
        "aux_local_model": "local_aux_model",
    }
    for old_key, new_key in legacy_map.items():
        value = migrated.pop(old_key, None)
        if value and new_key not in stored:
            migrated[new_key] = value
    migrated.pop("aux_llm_provider", None)
    return migrated


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
    stored = _migrate_legacy_stored(stored)
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


# ── Role → model resolution (bundle selected by the role's provider) ──────────


def generation_model() -> str:
    s = _settings
    return s.local_generation_model if s.llm_provider == "ollama" else s.openai_generation_model


def chat_model() -> str:
    s = _settings
    return s.local_chat_model if s.llm_provider == "ollama" else s.openai_chat_model


def retrieval_model() -> str:
    s = _settings
    return s.local_retrieval_model if s.embedding_provider == "ollama" else s.openai_retrieval_model


def embedding_provider() -> str:
    return _settings.embedding_provider


def current_mode() -> str:
    """'local' when generation runs on Ollama, else 'api'. Display/derived only."""
    return "local" if _settings.llm_provider == "ollama" else "api"


def settings_for_mode(mode: str) -> Settings:
    """The current settings with providers flipped for the requested mode.

    Model bundles are preserved — toggling modes never loses model choices."""
    if mode == "local":
        return replace(
            _settings,
            llm_provider="ollama",
            embedding_provider="ollama",
            whisper_provider="local",
        )
    if mode == "api":
        return replace(
            _settings,
            llm_provider="openai",
            embedding_provider="openai",
            whisper_provider="openai",
        )
    raise ValueError(f"Unknown mode: {mode!r} (expected 'local' or 'api')")


# Backwards-compatible module constants, refreshed by apply_settings.
OPENAI_API_KEY = _settings.openai_api_key
LLM_PROVIDER = _settings.llm_provider
EMBEDDING_PROVIDER = _settings.embedding_provider
WHISPER_PROVIDER = _settings.whisper_provider
OLLAMA_HOST = _settings.ollama_host
GENERATION_MODEL = generation_model()
CHAT_MODEL = chat_model()
RETRIEVAL_MODEL = retrieval_model()
AUX_USE_LOCAL = _settings.aux_use_local

if _settings.llm_provider == "openai" and not _settings.openai_api_key:
    logger.warning(
        "API mode is selected but no OpenAI key is set. Open the Settings page "
        "to add a key or switch to Local mode."
    )

# Kept for backwards compatibility — the OpenAI transcription path and the test
# cassette still reference openai_client directly. Recomputed by apply_settings.
openai_client = OpenAI(api_key=_settings.openai_api_key) if _settings.openai_api_key else None  # type: ignore[assignment]

_is_sqlite = DATABASE_URL.startswith("sqlite")
if _is_sqlite:
    _db_file = DATABASE_URL.removeprefix("sqlite:///")
    if _db_file and _db_file != ":memory:":
        Path(_db_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
db_engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"timeout": 30} if _is_sqlite else {},
)


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


# Reachability probe for the local daemon: cached with a short TTL so a
# momentarily-down daemon doesn't pin aux traffic to the hosted API forever.
_AUX_PROBE_TTL_SECONDS = 60.0
_aux_probe: tuple[float, bool] | None = None


def _local_aux_reachable() -> bool:
    global _aux_probe
    now = time.monotonic()
    if _aux_probe is not None and (now - _aux_probe[0]) < _AUX_PROBE_TTL_SECONDS:
        return _aux_probe[1]
    from app.ollama_probe import is_reachable

    ok = is_reachable(_settings.ollama_host, timeout=1.5)
    _aux_probe = (now, ok)
    return ok


def get_aux_client():
    """Client for high-volume auxiliary passes (ledger extraction).

    In local mode everything is already on Ollama. In API mode, aux passes
    route to the local daemon only when aux_use_local is set and it responds."""
    if _settings.llm_provider == "ollama":
        return get_llm_client("ollama")
    if _settings.aux_use_local and _local_aux_reachable():
        return get_llm_client("ollama")
    return get_llm_client()


def get_aux_model() -> str:
    """Model name to pair with get_aux_client()."""
    if _settings.llm_provider == "ollama":
        return _settings.local_aux_model
    if _settings.aux_use_local and _local_aux_reachable():
        return _settings.local_aux_model
    return _settings.openai_aux_model


def llm_call_slots(provider_name: str) -> threading.BoundedSemaphore:
    """Concurrency gate for in-flight LLM calls, sized per provider."""
    return _local_call_slots if provider_name == "ollama" else _hosted_call_slots


_hosted_call_slots = threading.BoundedSemaphore(HOSTED_MAX_PARALLEL)
_local_call_slots = threading.BoundedSemaphore(LOCAL_MAX_PARALLEL)


def apply_settings(new: Settings, *, persist: bool = True) -> Settings:
    """Swap the runtime settings, invalidate caches, and (by default) persist.

    Thread-safe: an in-flight request either sees the old client fully or the
    new one fully — never a torn read."""
    global _settings, _aux_probe, openai_client
    global OPENAI_API_KEY, LLM_PROVIDER, EMBEDDING_PROVIDER, WHISPER_PROVIDER, OLLAMA_HOST
    global GENERATION_MODEL, CHAT_MODEL, RETRIEVAL_MODEL, AUX_USE_LOCAL
    with _settings_lock:
        _settings = new
        _llm_client_cache.clear()
        _aux_probe = None
        openai_client = OpenAI(api_key=new.openai_api_key) if new.openai_api_key else None  # type: ignore[assignment]
        # Keep the backwards-compat module constants coherent with the new state.
        OPENAI_API_KEY = new.openai_api_key
        LLM_PROVIDER = new.llm_provider
        EMBEDDING_PROVIDER = new.embedding_provider
        WHISPER_PROVIDER = new.whisper_provider
        OLLAMA_HOST = new.ollama_host
        GENERATION_MODEL = generation_model()
        CHAT_MODEL = chat_model()
        RETRIEVAL_MODEL = retrieval_model()
        AUX_USE_LOCAL = new.aux_use_local
        if persist:
            persist_settings(new)
        return _settings


class _LazyChroma:
    """Opens the Chroma store on first use, not at import.

    Import-time construction made every `import config` (tests, the pywebview
    spawn child, `--selftest`) open the vector store. The proxy keeps
    `from config import chroma_client` working unchanged; the first touch
    happens in run_startup_checks() inside the server process."""

    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()

    def _get(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import chromadb

                    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return self._client

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


chroma_client = _LazyChroma()


def local_models_in_use() -> list[str]:
    """Distinct local model names the current settings can load."""
    s = _settings
    return list(dict.fromkeys(
        [s.local_generation_model, s.local_chat_model, s.local_aux_model, s.local_retrieval_model]
    ))


def run_startup_checks() -> None:
    with db_engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        if _is_sqlite:
            # WAL lets the telemetry writers coexist with parallel generation
            # threads instead of tripping "database is locked".
            connection.execute(text("PRAGMA journal_mode=WAL"))

    chroma_client.list_collections()
