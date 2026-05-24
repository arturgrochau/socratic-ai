"""
LLMClient abstraction: thin shim over OpenAI today, Ollama tomorrow.

Two implementations:
  * OpenAIClient — wraps the existing openai SDK.
  * OllamaClient — talks to a local Ollama daemon over HTTP.

Provider selection happens in config.get_llm_client() via app.models_registry.
Existing call sites can keep using openai_client directly; new code should go
through this module. To migrate a call site, replace
    openai_client.chat.completions.create(model=M, ...)
with
    get_chat_client_for("interaction").chat_json(...)
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

    # Make the result quack like the OpenAI completion object so log_api_usage
    # keeps working without changes.
    def __getattr__(self, name: str) -> Any:
        if name == "usage":
            return self.usage
        raise AttributeError(name)


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


class OllamaClient:
    """Minimal Ollama adapter. Works for chat + embeddings.

    Transcription is not supported and raises NotImplementedError — keep
    WHISPER_PROVIDER=openai when using Ollama for everything else.
    """
    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests
        response = requests.post(f"{self.host}{path}", json=payload, timeout=300)
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
    ) -> ChatResult:
        if stream:
            raise NotImplementedError("Streaming via OllamaClient not wired yet.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
            "stream": False,
        }
        if json_schema is not None:
            payload["format"] = "json"

        last_error: Exception | None = None
        for attempt in range(2):
            data = self._post("/api/chat", payload)
            content = (data.get("message", {}) or {}).get("content", "")
            if json_schema is None:
                return ChatResult(
                    content=content,
                    raw=data,
                    usage=ChatUsage(
                        prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"),
                        total_tokens=(
                            (data.get("prompt_eval_count") or 0)
                            + (data.get("eval_count") or 0)
                        ) or None,
                    ),
                )
            try:
                parse_json_object(content, stage_name="ollama")
                return ChatResult(
                    content=content,
                    raw=data,
                    usage=ChatUsage(
                        prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"),
                    ),
                )
            except ValueError as exc:
                last_error = exc
                if attempt == 0:
                    payload["messages"].append({
                        "role": "user",
                        "content": "Previous response was not valid JSON. Reply only with valid JSON.",
                    })
                    continue
        raise RuntimeError(f"Ollama returned invalid JSON twice: {last_error}")

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in inputs:
            data = self._post("/api/embeddings", {"model": model, "prompt": text})
            embeddings.append(data.get("embedding", []))
        return embeddings

    def transcribe(self, *, model: str, audio_path: str, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Ollama does not provide transcription. Set WHISPER_PROVIDER=openai."
        )


def build_client(provider: str) -> LLMClient:
    provider = (provider or "openai").strip().lower()
    if provider == "openai":
        return OpenAIClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM provider: {provider!r} (set LLM_PROVIDER=openai|ollama)")
