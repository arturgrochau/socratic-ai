from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
PROCESSING_MODEL = os.getenv("PROCESSING_MODEL", "gpt-4o-mini")
LINKING_MODEL = os.getenv("LINKING_MODEL", "gpt-4o-mini")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "text-embedding-3-small")
INTERACTION_MODEL = os.getenv("INTERACTION_MODEL", "gpt-4o-mini")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
GENERATION_CRITIC_MODEL = os.getenv("GENERATION_CRITIC_MODEL", "gpt-4o")
CONSOLIDATION_MODEL = os.getenv("CONSOLIDATION_MODEL", "gpt-4o-mini")
MAX_CHUNK_DIFF_WINDOWS = int(os.getenv("MAX_CHUNK_DIFF_WINDOWS", "1"))
ENABLE_GENERATION_CRITIC_FALLBACK = os.getenv("ENABLE_GENERATION_CRITIC_FALLBACK", "true").lower() == "true"
MAX_SOURCE_CRITIC_CALLS_PER_RUN = int(os.getenv("MAX_SOURCE_CRITIC_CALLS_PER_RUN", "1"))
MAX_CROSS_CRITIC_CALLS_PER_RUN = int(os.getenv("MAX_CROSS_CRITIC_CALLS_PER_RUN", "1"))
# Default-on so the existing redundancy gate in generation.py actually fires on
# every run. Set ENABLE_STRICT_GENERATION_GATES=false to disable.
ENABLE_STRICT_GENERATION_GATES = os.getenv("ENABLE_STRICT_GENERATION_GATES", "true").lower() == "true"

# Progressive chunking: per-source content pack built in N rounds with
# cumulative chunk slices, threading the prior round's deep_dive/under_surface
# as carried-over context. Diagnostic AB on long fixtures (4+ chunks) showed
# +148% deep_dive length at +18% token cost — clear win, so default ON.
# A separate min-chunks guard in generation.py prevents triggering on small
# sources where the cost/benefit reverses.
GENERATION_PROGRESSIVE_CHUNKING = os.getenv("GENERATION_PROGRESSIVE_CHUNKING", "true").lower() == "true"
GENERATION_PROGRESSIVE_ROUNDS = max(2, int(os.getenv("GENERATION_PROGRESSIVE_ROUNDS", "2")))
GENERATION_PROGRESSIVE_MIN_CHUNKS = max(2, int(os.getenv("GENERATION_PROGRESSIVE_MIN_CHUNKS", "4")))

# Token ceiling for the diagnostic harness. Soft assertion only — runs that
# exceed this get a warn in the report, not a failure.
DIAGNOSTIC_TOKEN_CEILING = int(os.getenv("DIAGNOSTIC_TOKEN_CEILING", "40000"))
USER_ID_HEADER = os.getenv("USER_ID_HEADER", "X-User-ID")
ENABLE_COST_LOGGING = os.getenv("ENABLE_COST_LOGGING", "true").lower() == "true"
CACHE_PROCESSED_SOURCES = os.getenv("CACHE_PROCESSED_SOURCES", "true").lower() == "true"
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

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER).strip().lower()
WHISPER_PROVIDER = os.getenv("WHISPER_PROVIDER", "openai").strip().lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()

# Only OpenAI strictly requires the API key. Local providers can run without it.
_needs_openai_key = (
    LLM_PROVIDER == "openai"
    or EMBEDDING_PROVIDER == "openai"
    or WHISPER_PROVIDER == "openai"
)
if _needs_openai_key and not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in .env before starting the server, "
        "or set LLM_PROVIDER=ollama (and EMBEDDING_PROVIDER=ollama, WHISPER_PROVIDER=none) "
        "to run fully against a local Ollama daemon."
    )

# Kept for backwards compatibility — existing modules still import openai_client
# directly. The new abstraction lives in app.llm_client; prefer get_llm_client().
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None  # type: ignore[assignment]
db_engine = create_engine(DATABASE_URL, future=True)


_llm_client_cache: dict[str, object] = {}


def get_llm_client(provider: str | None = None):
    """Return a cached LLMClient for the requested provider (defaults to LLM_PROVIDER)."""
    from app.llm_client import build_client

    key = (provider or LLM_PROVIDER).strip().lower()
    cached = _llm_client_cache.get(key)
    if cached is None:
        cached = build_client(key)
        _llm_client_cache[key] = cached
    return cached

Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def run_startup_checks() -> None:
    with db_engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    chroma_client.list_collections()