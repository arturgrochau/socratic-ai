# scripts/

Developer-only diagnostics, validation, and regression tooling. **Not shipped at runtime** and not
imported by the app — these drive the pipeline from the outside to measure quality and catch
regressions. Run them from the repo root (`python scripts/<name>.py`).

| Script | LOC | What it does | Needs |
|---|---|---|---|
| `validate_pipeline.py` | 933 | End-to-end contract validation against a running server: uploads fixtures, runs upload→generate→ask, asserts structure/grounding/budget invariants and scores the output. | Running server (`--api-base-url`), OpenAI key (or local provider). |
| `run_diagnostic.py` | 695 | Self-contained harness: spins up the backend on a free port in a subprocess, ingests a fixture, runs the full pipeline + 5 canned chat queries, joins session JSONL with DB telemetry, runs `checks.py`, and writes a markdown report to `logs/diagnostic/<run_id>.md`. | Fixture file; OpenAI key (unless fully local). |
| `checks.py` | 781 | Failure-mode catalog: rules over session telemetry (`logs/sessions/<run_id>.jsonl` + DB tables) that return pass/warn/fail with a remediation hint. Imported by `run_diagnostic.py` and by the e2e pytest. | Telemetry from a prior run (no LLM calls of its own). |
| `regression_suite.py` | 149 | Runs `run_diagnostic.py` across the canonical fixture shapes (single PDF, long text, two texts, video+PDF, cache validation) and aggregates one report. `--skip-video` avoids Whisper cost. | Same as `run_diagnostic.py`; slow. |
| `diagnostic_compare.py` | 169 | Prints a delta report (token cost, check counts, section sizes, answer lengths) between two prior runs by run id — no new API calls. | Two prior run ids + their telemetry. |

## Notes

- These tools write under `logs/` (gitignored). Run ids look like `gen-<hex>`.
- The pre-v2.0 leftovers were cleaned up: `evaluate_redundancy.py` was deleted (it queried a table
  that no longer exists) and the dead `--baseline`/`--progressive` plumbing was stripped from
  `run_diagnostic.py`. The live pipeline is described in [`../docs/pipeline.md`](../docs/pipeline.md).
