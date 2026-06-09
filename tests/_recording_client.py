"""
Cassette-based OpenAI client wrapper for offline e2e tests.

Two modes (selected by SOCRATIC_RECORD env var):

  * **Record mode** (`SOCRATIC_RECORD=1`): wraps the real OpenAI client; on
    each chat.completions.create call, hashes the request and stores
    (request_hash → response_content) in a per-test cassette JSON.
  * **Replay mode** (default): reads the cassette and returns a fake
    completion object. Fails loudly if a request hash isn't in the cassette,
    so prompt drift is impossible to miss.

Usage from a test:

    from tests._recording_client import install_cassette

    def test_thing(monkeypatch, tmp_path):
        install_cassette(monkeypatch, "test_thing")  # uses tests/_cassettes/test_thing.json
        # ... call code that uses openai_client.chat.completions.create
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Generation issues LLM calls from parallel worker threads; cassette reads of a
# plain dict are fine, but record-mode mutation + file rewrite must be serialized.
_cassette_lock = threading.Lock()


CASSETTE_DIR = Path(__file__).resolve().parent / "_cassettes"
RECORD_MODE = os.getenv("SOCRATIC_RECORD", "").lower() in {"1", "true", "yes"}


def _request_hash(model: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
    canonical = json.dumps(
        {
            "model": model,
            "messages": messages,
            "schema_name": (schema or {}).get("name"),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass
class _FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage


class _CassetteCompletions:
    def __init__(self, cassette: dict[str, dict[str, Any]], cassette_path: Path, real_client: Any | None) -> None:
        self._cassette = cassette
        self._path = cassette_path
        self._real = real_client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _FakeCompletion:
        schema = (response_format or {}).get("json_schema") if response_format else None
        key = _request_hash(model, messages, schema)

        if key in self._cassette:
            entry = self._cassette[key]
            return _build_fake(entry)

        if not RECORD_MODE or self._real is None:
            raise RuntimeError(
                f"Cassette miss for request hash {key} "
                f"(model={model}, schema={(schema or {}).get('name')!r}). "
                f"Re-record with SOCRATIC_RECORD=1."
            )

        # Record path — hit the real API and persist (serialized: parallel
        # generation threads may record concurrently).
        completion = self._real.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            **kwargs,
        )
        content = completion.choices[0].message.content or ""
        usage_obj = getattr(completion, "usage", None)
        entry = {
            "content": content,
            "usage": {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            },
        }
        with _cassette_lock:
            self._cassette[key] = entry
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._cassette, indent=2, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
        return _build_fake(entry)


def _build_fake(entry: dict[str, Any]) -> _FakeCompletion:
    usage = entry.get("usage") or {}
    return _FakeCompletion(
        choices=[_FakeChoice(message=_FakeMessage(content=entry.get("content", "")))],
        usage=_FakeUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        ),
    )


class _CassetteChat:
    def __init__(self, completions: _CassetteCompletions) -> None:
        self.completions = completions


class _CassetteEmbeddings:
    """Embeddings are short and deterministic enough to cassette by input hash."""

    def __init__(self, cassette: dict[str, dict[str, Any]], cassette_path: Path, real_client: Any | None) -> None:
        self._cassette = cassette
        self._path = cassette_path
        self._real = real_client

    def create(self, *, model: str, input: list[str] | str) -> Any:
        inputs = input if isinstance(input, list) else [input]
        key = "emb:" + hashlib.sha256(
            json.dumps({"model": model, "inputs": inputs}, sort_keys=True).encode()
        ).hexdigest()[:24]
        if key in self._cassette:
            data = self._cassette[key]
            class _D: pass
            response = _D()
            response.data = [type("E", (), {"embedding": e})() for e in data["embeddings"]]
            return response
        if not RECORD_MODE or self._real is None:
            raise RuntimeError(f"Embedding cassette miss {key}; re-record with SOCRATIC_RECORD=1.")
        real_resp = self._real.embeddings.create(model=model, input=inputs)
        embeddings = [item.embedding for item in real_resp.data]
        with _cassette_lock:
            self._cassette[key] = {"embeddings": embeddings}
            self._path.write_text(
                json.dumps(self._cassette, indent=2, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
        class _D: pass
        response = _D()
        response.data = [type("E", (), {"embedding": e})() for e in embeddings]
        return response


class CassetteClient:
    """Shape-compatible drop-in for the openai_client used in the codebase."""

    def __init__(self, cassette_path: Path) -> None:
        self._path = cassette_path
        self._cassette: dict[str, dict[str, Any]] = {}
        if cassette_path.exists():
            self._cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
        real = None
        if RECORD_MODE:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("SOCRATIC_RECORD=1 set but OPENAI_API_KEY is missing.")
            real = OpenAI(api_key=api_key)
        self.chat = _CassetteChat(_CassetteCompletions(self._cassette, cassette_path, real))
        self.embeddings = _CassetteEmbeddings(self._cassette, cassette_path, real)
        # Transcription not cassetted yet — tests that need it can skip on missing.

    @property
    def audio(self) -> Any:
        raise NotImplementedError(
            "Audio transcription is not cassetted. Use a PDF fixture for e2e tests."
        )


def install_cassette(monkeypatch: Any, cassette_name: str) -> CassetteClient:
    """Route every LLM call through a cassette.

    Two interception points are needed because the codebase reaches the model two
    ways: (1) the bare ``config.openai_client`` (transcription) and (2) the
    ``LLMClient`` wrapper returned by ``config.get_llm_client()`` /
    ``get_aux_client()`` (all generation/chat/retrieval calls). The cassette is
    shape-compatible with the bare OpenAI SDK, so we back an ``OpenAIClient``
    wrapper with it and hand that out from the factories."""
    import importlib

    cassette_path = CASSETTE_DIR / f"{cassette_name}.json"
    client = CassetteClient(cassette_path)

    # An OpenAIClient wrapper whose underlying SDK client IS the cassette.
    from app.llm_client import OpenAIClient
    wrapped = OpenAIClient.__new__(OpenAIClient)
    wrapped._client = client  # type: ignore[attr-defined]

    import config
    monkeypatch.setattr(config, "openai_client", client)
    monkeypatch.setattr(config, "get_llm_client", lambda provider=None: wrapped)
    monkeypatch.setattr(config, "get_aux_client", lambda: wrapped)

    # Modules that did `from config import openai_client / get_llm_client /
    # get_aux_client` captured those symbols at import time; rebind each one.
    for mod_name in (
        "app.generation",
        "app.interaction",
        "app.retrieval",
        "app.ingestion",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "openai_client"):
            monkeypatch.setattr(mod, "openai_client", client)
        if hasattr(mod, "get_llm_client"):
            monkeypatch.setattr(mod, "get_llm_client", lambda provider=None: wrapped)
        if hasattr(mod, "get_aux_client"):
            monkeypatch.setattr(mod, "get_aux_client", lambda: wrapped)
    return client
