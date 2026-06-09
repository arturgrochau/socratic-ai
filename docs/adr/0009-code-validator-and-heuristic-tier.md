# ADR 0009 — Code replaces two LLM passes (section audit, chat depth tier) + focused quiz call

Status: accepted

## Context

Three places paid an LLM for work a model is not needed for, or hurt quality by over-bundling:

1. **The section audit** (`_audit_arc`, one aux call per arc) checked length budgets and
   cross-section restatement. Counting sentences is arithmetic; restatement is token-overlap — the
   ledger already ships `_overlap_ratio` for exactly this.
2. **The chat depth classifier** (`_classify_depth`, one aux call per chat turn) mapped a query to a
   4-way tier (lookup/explain/analyze/deepen) that only scales context size and answer-length budget.
   Keyword + length + turn-count heuristics route this as well as a model, and a misroute degrades
   gracefully (slightly more/less context), never correctness.
3. **The synthesis mega-prompt** asked one mini-model call for five artifacts (takeaways, synthesis,
   intersections, scenarios, AND the quiz). MCQ-generation practice consistently shows focused
   prompts beat bundled ones on mini models for distractor quality.

## Decision

- **`app/section_validator.py:validate_arc`** replaces the audit call: machine-checkable limits
  (`max_sentences`, `max_items`, `check_restatement`) now live as structured fields on `SectionSpec`
  (`prompts/sections.py`) next to the prose budgets that go into the prompt. Sentence ceilings carry a
  +2 grace margin; restatement flags >0.60 whole-section overlap against any earlier section in arc
  order. Violations feed the existing one-regen loop unchanged.
- **`interaction._heuristic_depth`** replaces the classifier call: deepen-markers (+ prior turns
  required), analyze-markers / long queries, lookup-prefixes for short queries, else `explain`.
  `chat_context_scale` consumption is unchanged. Chat is now ONE LLM call per turn.
- **The quiz is its own focused call** (`QUIZ_SYSTEM_PROMPT` / `QUIZ_JSON_SCHEMA`), run **in
  parallel** with the insights call (`INSIGHTS_*`) in both the cross-source and single-source paths —
  same wall-clock as the old single call, better per-artifact quality. List over-runs (e.g. >4
  questions) are trimmed in code, not regenerated.

## Consequences

- Per run: −1 to −2 audit calls and −1 classifier call per chat turn, at zero quality cost for the
  checks that remained; the two fuzzy audit checks ("off-contract", "low-diversity") are dropped —
  the contracts still bind where they always actually bound, in the generation prompt itself.
- Validation is now deterministic and unit-testable (`tests/test_section_validator.py`), which the
  LLM audit never was.
- Trade-off: heuristics misroute occasional chat queries (e.g. a long lookup phrased verbosely gets
  `analyze` context). Cost: a few hundred extra context tokens; acceptable against an LLM call per
  turn.
