# ADR 0007 — Single-source enrichment (a lone document is a full artifact)

Status: accepted (supersedes the "synthesis only for ≥2 sources" behavior)

## Context

Generation originally hard-gated the enrichment stage on `total_sources >= 2`
(`generate_tailored_learning`). A single document therefore returned **empty** insights and quiz — no
takeaways, no apply-it scenarios, no quiz, no "bigger picture." The per-source sections also used tight
budgets. Net effect: uploading one document produced a thin result, which users (correctly) read as a
regression versus earlier versions. The cross-source synthesis design is sound (see
[ADR-0005](0005-cross-source-grounding.md)), but it was the *only* path to a quiz.

## Decision

Always produce an enrichment artifact, branching on source count:
- **≥2 sources:** cross-source synthesis as before (intersections grounded in ≥2 sources).
- **exactly 1 source:** a new **single-source deepening** stage
  (`generation.py:_generate_single_source_deepening`) that reuses `SYNTHESIS_JSON_SCHEMA` and a
  dedicated `SINGLE_SOURCE_SYNTHESIS_SYSTEM_PROMPT`. It produces `key_takeaways`, a short
  bigger-picture `synthesis_text`, `application_scenarios`, and a `questions` quiz — all grounded in the
  one source's ledger. `intersections` is always empty (nothing to compare), so no cross-source
  grounding is fabricated.

Supporting changes:
- Per-source budgets in `prompts/sections.py` were loosened (deeper, still selective — the
  anti-restatement/"cut filler" rules in `CONSOLIDATION_SYSTEM_PROMPT` are unchanged).
- The enrichment is cached like the cross-source one. The combined cache key uses a **sentinel
  `video_source_id = 0`** when there is no video, and the cache-validity check accepts a row that has a
  quiz/takeaways/synthesis (not only intersections). This also fixes a pre-existing gap where docs-only
  multi-source synthesis was never cached, and a bug where `key_takeaways` were never persisted
  (a new `key_takeaways_json` column was added).
- `GENERATION_SCHEMA_VERSION` 15 → 16 to regenerate stale cached output under the richer prompts.
- The UI now renders the enrichment for any source count and finally renders `application_scenarios`
  (previously generated but never displayed). The tab is labeled "Quiz & Applications" for one source,
  "Cross-Source" for many.

## Consequences

- A single document now yields takeaways + a bigger-picture reflection + apply-it scenarios + a quiz,
  at the cost of one extra LLM call per single-source run (verified end-to-end against the live model).
- The quiz handoff into Socratic chat works for single documents too (it only needed a non-empty quiz).
- Trade-off: the single-source quiz is not adversarially grounded the way cross-source intersections
  are; it relies on the prompt's "test this source's mechanisms, no rote recall" rules. Acceptable —
  there is no second source to cross-check against.

## Alternatives considered

- **Leave single-source thin, tell users to add a second source** — rejected: the common case is one
  document, and the tool should be useful there.
- **A separate `SourceQuizSection` model + per-source quiz column** — more schema churn; reusing the
  existing combined insight/quiz models and cache table was lighter and kept rendering unchanged.
