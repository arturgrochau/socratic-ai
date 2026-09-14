"""
LLMClient abstraction over a hosted OpenAI-compatible API and a local Ollama daemon.

Two implementations behind one protocol:
  * OpenAIClient — wraps the openai SDK; JSON enforced via response_format.
    A base_url override points it at any OpenAI-compatible server (OpenRouter,
    Groq, LM Studio, llama.cpp, DeepSeek...). Servers that reject
    `json_schema` fall back to `json_object` + our own validation.
  * OllamaClient — talks to the local daemon over HTTP (httpx); JSON enforced via
    Ollama structured outputs (the JSON schema is compiled to a decoding
    grammar server-side, so malformed JSON is mechanically impossible).

Both expose chat_stream() for the Socratic chat: plain text deltas, no schema.

Provider selection happens in config.get_llm_client(). All pipeline call
sites go through chat_json/embed; transcription has its own provider switch
in app/ingestion.py (local Whisper does not involve Ollama).
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

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


@dataclass
class StreamState:
    """Filled in by chat_stream() as the stream runs; read it after draining."""

    usage: ChatUsage = field(default_factory=ChatUsage)
    model: str | None = None


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
        large_context: bool = False,
    ) -> ChatResult: ...

    def chat_stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        large_context: bool = False,
        state: StreamState | None = None,
    ) -> Iterator[str]: ...

    def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]: ...

    def transcribe(self, *, model: str, audio_path: str, **kwargs: Any) -> Any: ...


class OpenAIClient:
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI  # local import keeps non-OpenAI runs lighter

        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=self.base_url)
        # Some OpenAI-compatible servers only know the older json_object mode;
        # remembered per client so the fallback costs one failed call, not one per stage.
        self._schema_unsupported = False

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
            if self._schema_unsupported:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if json_schema is None or not self._looks_like_schema_rejection(exc):
                raise
            logger.info("%s rejected json_schema; falling back to json_object", self.base_url or "openai")
            self._schema_unsupported = True
            kwargs["response_format"] = {"type": "json_object"}
            completion = self._client.chat.completions.create(**kwargs)

        content = completion.choices[0].message.content or ""
        if json_schema is not None and self._schema_unsupported:
            parse_json_object(content, stage_name="openai-compatible")  # raises on garbage
        return ChatResult(content=content, raw=completion, usage=_usage_from(completion))

    @staticmethod
    def _looks_like_schema_rejection(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        return status in (400, 422) and ("response_format" in text or "json_schema" in text)

    def chat_stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        large_context: bool = False,
        state: StreamState | None = None,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }
        if not self.base_url:
            # Only the real OpenAI API is known to honour this; compatible
            # servers may 400 on unknown params.
            kwargs["stream_options"] = {"include_usage": True}
        for chunk in self._client.chat.completions.create(**kwargs):
            if state is not None and getattr(chunk, "usage", None):
                state.usage = _usage_from(chunk)
                state.model = getattr(chunk, "model", None) or model
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text

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


def _usage_from(completion: Any) -> ChatUsage:
    usage_obj = getattr(completion, "usage", None)
    if not usage_obj:
        return ChatUsage()
    return ChatUsage(
        prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
        completion_tokens=getattr(usage_obj, "completion_tokens", None),
        total_tokens=getattr(usage_obj, "total_tokens", None),
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

    def __init__(self, host: str | None = None, client: httpx.Client | None = None) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        # One pooled connection per client; tests inject an httpx.MockTransport client.
        self._http = client or httpx.Client()

    def _timeout(self) -> httpx.Timeout:
        import config

        return httpx.Timeout(config.OLLAMA_TIMEOUT_SECONDS, connect=10)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(f"{self.host}{path}", json=payload, timeout=self._timeout())
        response.raise_for_status()
        return response.json()

    def _stream_post(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield one parsed object per NDJSON line from a streaming endpoint."""
        with self._http.stream("POST", f"{self.host}{path}", json=payload, timeout=self._timeout()) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _options(self, temperature: float, large_context: bool) -> dict[str, Any]:
        import config

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": config.ollama_num_ctx(large=large_context),
        }
        if temperature > 0:
            options["top_p"] = 0.8  # qwen3-instruct guidance for sampled decoding
        return options

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        large_context: bool = False,
    ) -> ChatResult:
        import config

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": self._options(temperature, large_context),
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

    def chat_stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        large_context: bool = False,
        state: StreamState | None = None,
    ) -> Iterator[str]:
        import config

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": self._options(temperature, large_context),
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "stream": True,
        }
        for event in self._stream_post("/api/chat", payload):
            text = (event.get("message") or {}).get("content", "")
            if text:
                yield text
            if event.get("done"):
                if state is not None:
                    prompt_tokens = event.get("prompt_eval_count")
                    completion_tokens = event.get("eval_count")
                    state.usage = ChatUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=((prompt_tokens or 0) + (completion_tokens or 0)) or None,
                    )
                    state.model = event.get("model") or model
                break

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
    base_url = getattr(settings, "openai_base_url", None)
    host = getattr(settings, "ollama_host", None)
    if provider == "openai":
        return OpenAIClient(api_key=api_key, base_url=base_url)
    if provider == "ollama":
        return OllamaClient(host=host)
    raise ValueError(f"Unknown LLM provider: {provider!r} (set LLM_PROVIDER=openai|ollama)")
