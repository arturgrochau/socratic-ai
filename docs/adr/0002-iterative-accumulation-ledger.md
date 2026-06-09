# ADR 0002 — Iterative-accumulation knowledge ledger

Status: accepted (v2.0, supersedes the novelty-ledger / claim-extraction / progressive-chunking design).
**Partially superseded by [ADR-0008](0008-map-reduce-ledger.md):** the ledger model, types, dedup, and
consolidation below all stand, but extraction is now parallel map-reduce, not sequential accumulation.

## Context

Long sources don't fit in one context window, and naive per-chunk summarization produces repetitive,
overlapping output (every chunk re-explains the same core ideas). The earlier design tried to fix this
with separate claim-extraction, a novelty ledger, and progressive chunking machinery — ~4,200 lines
that were hard to reason about.

## Decision

Represent a source's knowledge as a **ledger of typed, source-grounded atomic claims** (`KnowledgeUnit`
in `app/ledger.py`: `id`, `source_id`, `type` ∈ foundational/mechanism/tradeoff/assumption/boundary/
example/open_question, `claim`, `evidence`). Build it by **iterative accumulation**
(`generation.py:_extract_source_ledger`): walk chunk windows in order, showing each extraction call the
*full prior ledger* and instructing it to emit only genuinely new units. `append_units` dedups on a
0.72 token-overlap ratio. A single **consolidation** call (`_consolidate_ledger`) then maps the ledger
onto the cognitive-arc sections defined declaratively in `prompts/sections.py`.

This gives three properties by construction: every unit is **typed**, every unit is **grounded** (carries
a source id + verbatim evidence), and each claim is recorded **once** (dedup on append).

## Consequences

- Replaced ~4,200 lines with ~880 in the generation core; ~10× cheaper per run; no `gpt-4o` critic.
- Sections consume slices of the ledger by role and are told to *reference, not restate*, which (with the
  audit pass, see arc audit in `pipeline.md`) keeps overlap down.
- The ledger is cached separately from the consolidated output (`source_ledger_units` vs
  `source_learning_sections`) — see [ADR-0004](0004-cache-via-schema-version.md). This is a deliberate
  two-level cache, not redundancy: bumping the consolidation schema re-runs only consolidation and
  reuses the (expensive) extracted ledger.
- Known trade-off: the dedup threshold (0.72) and window sizing (3–4 chunks) are tuned constants in
  `app/ledger.py` / `generation.py`, not adaptive. They work well empirically but are worth revisiting
  with `generation_stage_events` telemetry if source styles change.

## Alternatives considered

- **One-shot whole-source prompt** — rejected: doesn't fit long sources; degraded quality near limits.
- **Per-chunk summarize + post-hoc dedup** — rejected: produces overlapping prose that dedup can't fully
  repair; loses the typed topology.
- **Embedding-clustering for novelty** — rejected: heavier and less interpretable than ordered
  accumulation with a visible ledger.
