# ADR 0006 — Pluggable LLM providers (OpenAI / Ollama)

Status: accepted

## Context

The app should run against the OpenAI API by default but also support a fully-local/offline-ish setup
for privacy-sensitive material or cost control, without rewriting call sites. Different roles (chat,
embeddings, transcription, aux extraction) may reasonably use different providers.

## Decision

Define one `LLMClient` protocol (`app/llm_client.py`) with `chat_json` / `embed` / `transcribe`, and two
implementations: `OpenAIClient` and `OllamaClient`. `config.get_llm_client(provider)` returns a cached
client; `build_client` injects the API key / host from the current `Settings`. Provider selection is
**per role** via settings: `llm_provider`, `embedding_provider`, `whisper_provider`, plus the aux group
(see [ADR-0003](0003-dual-model-aux-routing.md)). `chat_json` normalizes both providers to a common
`ChatResult` (with token usage), so logging and parsing are provider-agnostic. The result object quacks
like the OpenAI completion so existing `log_api_usage` keeps working.

## Consequences

- Switching providers is a settings change at runtime ([ADR-0001] flow, `apply_settings` clears the
  client cache); no code change, no restart.
- Roles can be mixed (e.g. OpenAI embeddings + transcription, local Llama for generation).
- **Constraint:** `OllamaClient.transcribe` raises `NotImplementedError` — Whisper transcription always
  needs OpenAI (`whisper_provider=openai`). This is documented in the README and the client docstring.
- `OllamaClient.chat_json` enforces JSON with a one-shot "reply only with valid JSON" retry, since local
  models honor `format=json` less reliably than OpenAI's `json_schema` response format.
- Some call sites still use the bare `config.openai_client` directly (ingestion/transcription, test
  cassette); `apply_settings` keeps it coherent. New code should prefer `get_llm_client`.

## Alternatives considered

- **Hard-code OpenAI** — rejected: no local/offline or cost-control story.
- **A heavyweight gateway (LiteLLM / LangChain)** — rejected: the project deliberately avoids heavy
  frameworks; a ~40-line adapter per provider covers the two providers it actually uses.
