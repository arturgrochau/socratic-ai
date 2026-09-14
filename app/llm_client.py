"""
LLMClient abstraction over the hosted OpenAI API and a local Ollama daemon.

Two implementations behind one protocol:
  * OpenAIClient — wraps the openai SDK; JSON enforced via response_format.
  * OllamaClient — talks to the local daemon over HTTP; JSON enforced via
    Ollama structured outputs (the JSON schema is compiled to a decoding
    grammar server-side, so malformed JSON is mechanically impossible).

Provider selection happens in config.get_llm_client(). All pipeline call
sites go through chat_json/embed; transcription has its own provider switch
in app/ingestion.py (local Whisper does not involve Ollama).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from app.json_reliability import parse_json_object


logger = logging.getLogger(__name__)


@dataclass
class ChatUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ChatResult:
    content: str
    raw: Any
    usage: ChatUsage


class LLMClient(Protocol):
    name: str

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        large_context: bool = False,
    ) -> ChatResult: ...

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]: ...

    def transcribe(self, *, model: str, audio_path: str, **kwargs: Any) -> Any: ...


class OpenAIClient:
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI  # local import keeps non-OpenAI runs lighter
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @property
    def raw(self) -> Any:
        """Escape hatch for code that still wants the bare OpenAI client."""
        return self._client

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        large_context: bool = False,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        if stream:
            kwargs["stream"] = True

        completion = self._client.chat.completions.create(**kwargs)

        if stream:
            # Streaming returns an iterator; caller is responsible for draining.
            return ChatResult(content="", raw=completion, usage=ChatUsage())

        content = completion.choices[0].message.content or ""
        usage_obj = getattr(completion, "usage", None)
        usage = ChatUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", None) if usage_obj else None,
            completion_tokens=getattr(usage_obj, "completion_tokens", None) if usage_obj else None,
            total_tokens=getattr(usage_obj, "total_tokens", None) if usage_obj else None,
        )
        return ChatResult(content=content, raw=completion, usage=usage)

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=model, input=inputs)
        return [item.embedding for item in response.data]

    def transcribe(self, *, model: str, audio_path: str, **kwargs: Any) -> Any:
        with open(audio_path, "rb") as audio:
            return self._client.audio.transcriptions.create(
                model=model,
                file=audio,
                **kwargs,
            )


# Bound completion length under grammar-constrained decoding: a grammar can
# steer a model into states it would never sample freely (e.g. an endless
# array); an explicit cap turns that into an invalid-JSON retry instead of a
# multi-minute stall. Far above any legitimate pipeline output (~2k tokens).
OLLAMA_NUM_PREDICT = 4096


class OllamaClient:
    """Ollama adapter for chat + embeddings.

    chat_json uses structured outputs (schema passed as `format`), sets an
    explicit context window (the daemon default of 4096 silently truncates
    long prompts head-first), and keeps the model warm between pipeline
    stages. Transcription is not Ollama's job — see app/ingestion.py.
    """
    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        import config

        response = requests.post(
            f"{self.host}{path}",
            json=payload,
            timeout=(10, config.OLLAMA_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return response.json()

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        large_context: bool = False,
    ) -> ChatResult:
        if stream:
            raise NotImplementedError("Streaming via OllamaClient not wired yet.")

        import config

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": config.OLLAMA_NUM_CTX_LARGE if large_context else config.OLLAMA_NUM_CTX,
        }
        if temperature > 0:
            options["top_p"] = 0.8  # qwen3-instruct guidance for sampled decoding
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "stream": False,
        }
        if json_schema is not None:
            # Unwrap the OpenAI response_format envelope ({name, strict, schema})
            # to the bare JSON schema Ollama compiles into a decoding grammar.
            payload["format"] = json_schema.get("schema", json_schema)
            payload["options"]["num_predict"] = OLLAMA_NUM_PREDICT

        last_error: Exception | None = None
        for attempt in range(2):
            data = self._post("/api/chat", payload)
            content = (data.get("message", {}) or {}).get("content", "")
            prompt_tokens = data.get("prompt_eval_count")
            completion_tokens = data.get("eval_count")
            usage = ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=((prompt_tokens or 0) + (completion_tokens or 0)) or None,
            )
            if json_schema is None:
                return ChatResult(content=content, raw=data, usage=usage)
            try:
                parse_json_object(content, stage_name="ollama")
                return ChatResult(content=content, raw=data, usage=usage)
            except ValueError as exc:
                last_error = exc
                if attempt == 0:
                    # Show the model its own invalid output; nudge sampling off
                    # the deterministic path so the retry isn't a re-roll of the
                    # exact same failure.
                    payload["messages"] = payload["messages"] + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": "That response was not valid JSON for the required schema. "
                            "Reply again with only the corrected JSON object.",
                        },
                    ]
                    payload["options"] = {**payload["options"], "temperature": max(temperature, 0.3)}
                    continue
        raise RuntimeError(f"Ollama returned invalid JSON twice: {last_error}")

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        import config

        data = self._post(
            "/api/embed",
            {"model": model, "input": inputs, "keep_alive": config.OLLAMA_KEEP_ALIVE},
        )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise RuntimeError(
                f"Ollama /api/embed returned {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"embeddings for {len(inputs)} inputs (model {model})."
            )
        return embeddings

    def transcribe(self, *, model: str, audio_path: str, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Ollama does not provide transcription. Use whisper_provider='local' or 'openai'."
        )


def build_client(provider: str, *, settings: Any = None) -> LLMClient:
    """Construct an LLMClient for a provider.

    When `settings` is supplied (a config.Settings), the API key / host come
    from it so runtime changes via the Settings panel take effect. Falls back
    to environment variables otherwise (keeps tests and direct callers working).
    """
    provider = (provider or "openai").strip().lower()
    api_key = getattr(settings, "openai_api_key", None)
    host = getattr(settings, "ollama_host", None)
    if provider == "openai":
        return OpenAIClient(api_key=api_key)
    if provider == "ollama":
        return OllamaClient(host=host)
    raise ValueError(f"Unknown LLM provider: {provider!r} (set LLM_PROVIDER=openai|ollama)")
