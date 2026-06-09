# ADR 0008 — Map-reduce ledger extraction (parallel, context-free windows)

Status: accepted (supersedes the sequential-accumulation part of [ADR-0002](0002-iterative-accumulation-ledger.md))

## Context

The v2.0 design extracted knowledge units by walking chunk windows **sequentially**, each LLM call
seeing the *full prior ledger* and being told not to restate it. That bought novelty-awareness at two
compounding costs: (1) **quadratic token growth** — window *k* re-sends all units from windows
1..k-1, so a 50-page PDF (~12 windows) wastes 10-20k tokens re-sending the same ledger; and (2)
**serial wall-clock** — 12 windows = 12 round-trips one after another. This is the classic
map-reduce-vs-refine trade-off from summarization pipelines (LangChain `MapReduce` vs `Refine`
chains): refine is context-aware but serial and token-hungry; map is parallel and cheap but needs a
dedup/merge step.

The decisive observation: **this codebase already has the merge step as pure code.**
`ledger.py:append_units` dedups claims at a 0.72 token-overlap threshold — the exact mechanism the
sequential prompt was trying to get the model to do.

## Decision

Switch `generation.py:_extract_source_ledger` to **map-reduce**:
- **Map**: every window is extracted concurrently (`_run_parallel`, ThreadPoolExecutor; global LLM
  concurrency capped at `MAX_PARALLEL_LLM_CALLS=8` via a semaphore at the call site, since source-level
  and window-level pools nest). Each call sees ONLY its window — `LEDGER_EXTRACTION_SYSTEM_PROMPT` no
  longer mentions a prior ledger.
- **Reduce**: payloads are merged **in window order** through `parse_units` + `append_units` — no LLM
  reduce call. Window-order merging keeps unit ids deterministic and makes the earliest occurrence of
  a duplicated claim the canonical one.
- Per-source builds are themselves parallelized across sources in `generate_tailored_learning`.
- Worker threads run under `contextvars.copy_context()` so the session logger (a ContextVar) keeps
  recording from inside pools.
- `LEDGER_SCHEMA_VERSION` 1 → 2: cached ledgers regenerate under the new extraction.

## Consequences

- A 12-window source goes from 12 serial round-trips to ~2 parallel batches; prompt tokens drop by
  the entire re-sent-ledger overhead (the dominant token cost for long sources).
- Trade-off accepted: context-free extraction can emit near-duplicate units across windows that
  phrase a claim differently enough to pass the 0.72 overlap dedup. Mitigations: the consolidation
  stage is explicitly an organize-and-compress step over the ledger, and the code validator
  (ADR-0009) flags cross-section restatement in the final output. Observed quality on the test
  fixtures is equivalent.
- Window-order determinism matters for tests and caching: same windows → same ledger.

## Alternatives considered

- **Keep sequential refine** — correct but quadratic and serial; rejected for anything beyond a few
  windows.
- **LLM reduce call** (merge units with a model) — unnecessary; the overlap dedup already exists in
  code and is deterministic.
- **RAPTOR-style hierarchical clustering** — right for 100+ page books; overkill for this app's
  typical sources. Revisit if multi-hour-video / textbook ingestion becomes a goal.
