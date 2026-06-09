# ADR 0003 — Dual-model (generation + aux) routing

Status: accepted (with a known caveat — see Consequences)

## Context

The pipeline makes two very different kinds of LLM call:
- **High-volume, mechanical** calls — ledger extraction (one per window), arc audits, depth
  classification. These are structured-output tasks where a smaller/cheaper/local model is adequate.
- **Low-volume, quality-critical** calls — consolidation and cross-source synthesis, which shape the
  prose the user reads.

## Decision

Split the model used per call. `config.get_aux_client()` / `get_aux_model()` serve the mechanical calls;
`config.generation_model()` / `config.chat_model()` serve the quality-critical and chat calls.
`_run_structured_generation_step(use_aux=True)` selects the aux path. When `aux_use_local` is set and a
local Ollama daemon is reachable (cached probe `config._local_aux_reachable`), aux routes to Ollama
(`aux_local_model`, default `llama3.1:8b`); otherwise it falls back to the hosted aux model.

## Consequences

- The seam exists and works: enabling `aux_use_local` offloads the high-volume calls to a local model,
  cutting hosted cost and enabling offline-ish runs (Whisper still needs OpenAI).
- **Known caveat:** by default `aux_model` resolves to `generation_model` (both `gpt-4o-mini`), so out of
  the box the split buys *nothing* — every call hits the same model. The abstraction's cost (extra config
  surface, two client paths) is paid whether or not anyone benefits. This is acceptable as an opt-in lever
  but is easy to misread as "we're already saving money on aux calls." We are not, unless `aux_use_local`
  (or a distinct `AUX_MODEL`) is configured.
- Pragmatic options if the caveat bites: (a) set a genuinely cheaper hosted aux default, or (b) collapse
  the abstraction until local routing is actually used. Tracked in
  [ARCHITECTURE.md → Known issues](../../ARCHITECTURE.md#7-known-issues--backlog).

## Alternatives considered

- **One model everywhere** — simpler, but no path to cheap/local high-volume calls.
- **Per-stage model table** — more flexible, but more config than this app warrants today.
