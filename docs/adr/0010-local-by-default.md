# ADR-0010: Local by default, one-click API mode, one resident local model

**Status:** accepted (v2.3)

## Context

v2.2 was OpenAI-first: the app refused to boot without an API key whenever any provider was
`openai` (and `whisper_provider` defaulted to `openai` unconditionally, so `LLM_PROVIDER=ollama`
alone could not boot). The Ollama path was second-class: schemas were downgraded to bare
`format:"json"`, no `num_ctx`/`keep_alive` were set (the daemon's 4096 default silently truncated
consolidation prompts head-first), embeddings went one HTTP call per string through a deprecated
endpoint, transcription bypassed the provider abstraction entirely, and the single Chroma
collection corrupted retrieval on any embedding-model switch (1536-d vs 768-d vectors in one
space, or worse, same-dim silent geometry mixing).

Meanwhile the target machine (Apple-Silicon, 48 GB) measured `qwen3:30b-a3b-instruct-2507-q4_K_M`
at ~1,230 tok/s prefill and 64–71 tok/s decode — *faster at decode than the dense 4B* (3 B active
MoE params), with first-try schema-valid JSON under Ollama structured outputs.

## Decision

1. **Local is the default.** `Settings` defaults: `llm_provider=ollama`, `embedding_provider=ollama`,
   `whisper_provider=local`. No import-time key check — a missing key is a Settings-page concern,
   never a boot failure. API-mode selection is validated at `PUT /settings` time instead.
2. **Two model bundles, one toggle.** `Settings` carries `openai_*` and `local_*` model fields; the
   role accessors (`generation_model()` …) resolve through the role's provider. `PUT /settings/mode`
   flips all providers at once; bundles persist, so toggling never loses model choices.
3. **One resident local model for every role.** Generation, chat, and the aux extraction passes all
   default to the 30b-a3b MoE: it is decode-competitive with a 4B, and a single resident model
   eliminates load thrash (observed ~6 s reload per 18 GB swap). The SLM/LLM split remains for API
   mode (`openai_aux_model`) and as a manual override (`local_aux_model`).
4. **First-class OllamaClient.** Structured outputs (`format` = the bare JSON schema — grammar-
   constrained decoding, malformed JSON mechanically impossible), per-tier `num_ctx`
   (16 k default / 32 k for consolidation-class prompts), `keep_alive=30m`, `num_predict` cap under
   grammar constraint, batched `/api/embed`, and a JSON retry that includes the model's own failed
   output and nudges temperature off the deterministic path.
5. **Provider-aware concurrency.** `config.llm_call_slots(provider)`: 8-wide hosted, 2-wide local
   (`OLLAMA_NUM_PARALLEL` defaults to 1 — wide submission only queues into the HTTP timeout).
6. **Embedding-model-keyed retrieval.** One Chroma collection per (provider, model), cosine space,
   task prefixes for nomic/embeddinggemma, lazy re-embed per collection. Switching modes cannot
   read the other mode's vectors.
7. **Provider-routed transcription.** `local` → parakeet-mlx (fallback mlx-whisper, optional
   dependency group, Apple-Silicon-only) with no request-size chunking; `openai` → the previous
   Whisper path resolving `config.openai_client` at call time; `none` → clean error.

## Consequences

- Keyless first-run works end-to-end; privacy ("nothing leaves this machine") is the default
  product stance, matching the ADR-cited market gap (no mainstream study tool processes locally).
- Cached artifacts are not keyed by provider/model — after a mode switch, previously generated
  packs are served from cache regardless of which mode produced them (cheap, but be aware when
  A/B-ing local vs hosted quality; bump `GENERATION_SCHEMA_VERSION` to force regeneration).
- The legacy `interaction_retrieval_concepts` collection is orphaned; sources re-embed lazily into
  model-keyed collections on first use (one-time, cheap locally, ~$0.001/source hosted).
- The e2e cassette pins the recorded OpenAI models via `install_cassette` so runtime defaults
  can't invalidate request hashes.
