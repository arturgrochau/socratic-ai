# ADR 0004 — Caching via schema-version constants

Status: accepted

## Context

Generation is the expensive part (~11 LLM calls for a 2-source run). Reopening a session, or
regenerating after only a prompt/format tweak, should not re-pay that cost. But when the *shape* of the
output changes (new field, changed contract), cached rows become invalid and must be regenerated.

## Decision

Cache generated artifacts in SQLite keyed by `(user_id, source_id)` (per-source) and
`(user_id, video_source_id, document_source_ids_json)` (cross-source), each row stamped with a
**schema-version integer**. On load, a row whose stored version ≠ the current constant is treated as a
cache miss and regenerated. Two independent versions:
- `GENERATION_SCHEMA_VERSION` (`generation.py`) — the consolidated/synthesis output shape.
- `LEDGER_SCHEMA_VERSION` (`generation.py`/`ledger.py`) — the extracted ledger shape.

Bumping a constant is the cache-bust mechanism (documented in the README's Versioning section).

## Consequences

- Two-level invalidation is a feature: bumping `GENERATION_SCHEMA_VERSION` re-runs only consolidation
  /synthesis and **reuses** the (expensive) extracted ledger, because `LEDGER_SCHEMA_VERSION` is
  unchanged. You only pay for re-extraction when the ledger shape itself changes.
- `CACHE_PROCESSED_SOURCES` (env, default on) gates the whole thing for testing/forced regeneration.
- Cache validity also depends on content sanity (e.g. a cached per-source row with an empty summary or
  no key terms is rejected), so a corrupt cache self-heals.
- **Known issue:** `GENERATION_SCHEMA_VERSION` is currently `15`, implying ~14 prior breaking changes,
  but there is no changelog mapping a version to what changed. That makes it hard to know whether an old
  cache is safe or why a bump happened. Fix: keep a version→change map (here or in a CHANGELOG). Low
  priority. See [ARCHITECTURE.md → Known issues](../../ARCHITECTURE.md#7-known-issues--backlog).
- No data migration: a bump throws away old cached rows rather than upgrading them. Fine for a
  regenerable local cache; would not be acceptable for user-authored data.

## Alternatives considered

- **Content-hash cache keys** (hash of inputs + prompt + code) — more precise invalidation, but more
  machinery; the integer version is trivial to reason about for a single maintainer.
- **No cache** — rejected: every reopen would re-pay full generation cost and latency.
