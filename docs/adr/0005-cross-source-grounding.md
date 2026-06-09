# ADR 0005 — Cross-source intersection grounding

Status: accepted (with a known correctness gap — see Consequences)

## Context

The most valuable—and most hallucination-prone—output is the cross-source synthesis: claims that two
or more sources "reinforce, contradict, or complete each other." A model will happily invent a
connection that isn't in the material, or attribute a point to a source that never made it.

## Decision

Defend grounding in three layers (`generation.py:_generate_synthesis`):
1. **Prompt** — synthesis is fed *only* the merged knowledge ledger, a source legend (refer to sources
   by name, never raw id), and an explicit rule: every intersection must cite claims from ≥2 different
   sources via `attributed_sentences`; do not introduce topics absent from the ledger.
2. **Audit** — `_audit_arc(CROSS_SOURCE_SECTIONS, …)` checks the `grounding` rule and triggers one
   re-generation on violation.
3. **Post-filter** — `_ground_intersections` first drops any attributed sentence citing a `source_id`
   that wasn't loaded for the run (a fabricated id), then keeps only intersections whose surviving
   citations span ≥2 distinct *real* sources.

## Consequences

- Ungrounded, single-source, or fabricated-source "intersections" are filtered out rather than shown,
  and a fabricated attribution never reaches the user (the offending sentence is pruned).
- **Update:** an earlier version of the post-filter checked only the *count* of distinct cited ids
  (`>= 2`), not whether those ids existed — a hallucinated `source_id=999` could pass. This is now
  fixed: `_ground_intersections` takes the run's real `valid_source_ids` and intersects against it
  before accepting. Covered by `tests/test_generation_quality.py::GroundIntersectionsTests`.

## Alternatives considered

- **Trust the prompt alone** — rejected: synthesis is exactly where models confabulate.
- **Verify each attributed sentence against source text (string/semantic match)** — stronger, but
  meaningfully more compute and latency; the ledger-only input + id check gets most of the benefit for
  far less cost. Worth revisiting if grounding errors are observed.
