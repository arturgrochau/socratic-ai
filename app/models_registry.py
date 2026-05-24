"""
Logical role -> provider+model mapping.

Centralizes every model decision so swapping the backing LLM is one edit.
Roles map to environment overrides; defaults are tuned for OpenAI's gpt-4o-mini.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str


def _env(role: str, *, default_provider: str, default_model: str) -> ModelChoice:
    provider = os.getenv(f"{role}_PROVIDER", os.getenv("LLM_PROVIDER", default_provider)).strip().lower()
    model = os.getenv(f"{role}_MODEL", default_model).strip()
    return ModelChoice(provider=provider, model=model)


# Override priority: <ROLE>_PROVIDER > LLM_PROVIDER > default
PROCESSING = _env("PROCESSING", default_provider="openai", default_model="gpt-4o-mini")
GENERATION = _env("GENERATION", default_provider="openai", default_model="gpt-4o-mini")
GENERATION_CRITIC = _env("GENERATION_CRITIC", default_provider="openai", default_model="gpt-4o")
CONSOLIDATION = _env("CONSOLIDATION", default_provider="openai", default_model="gpt-4o-mini")
LINKING = _env("LINKING", default_provider="openai", default_model="gpt-4o-mini")
INTERACTION = _env("INTERACTION", default_provider="openai", default_model="gpt-4o-mini")

# Embedding + transcription have their own provider envs because they're often
# kept on OpenAI even when chat goes local.
EMBEDDING = ModelChoice(
    provider=os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower(),
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
)
TRANSCRIPTION = ModelChoice(
    provider=os.getenv("WHISPER_PROVIDER", "openai").strip().lower(),
    model=os.getenv("WHISPER_MODEL", "whisper-1").strip(),
)


def all_choices() -> dict[str, ModelChoice]:
    return {
        "processing": PROCESSING,
        "generation": GENERATION,
        "generation_critic": GENERATION_CRITIC,
        "consolidation": CONSOLIDATION,
        "linking": LINKING,
        "interaction": INTERACTION,
        "embedding": EMBEDDING,
        "transcription": TRANSCRIPTION,
    }
