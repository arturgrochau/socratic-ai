# Contributing

Bug reports and pull requests are welcome. For anything larger than a fix, open an issue first so we can agree on the shape.

## Set up

```bash
git clone https://github.com/arturgrochau/socratic-ai.git
cd socratic-ai
uv sync --extra dev --extra local --extra app   # drop --extra local off Apple Silicon
uv run socratic-ai                              # browser tab at http://localhost:8000
uv run socratic-ai-app                          # native window
```

## Before you push

```bash
uv run ruff check .
uv run pytest tests/ -m "not e2e"     # unit + UI smoke, mocked LLMs, a few seconds
uv run pytest tests/ -m e2e           # replays the recorded cassette, no network
```

The end-to-end test replays `tests/_cassettes/test_e2e_pipeline.json`. Do not re-record it (`SOCRATIC_RECORD=1`) unless you changed a prompt or a schema; re-recording costs real API calls and produces a large diff.

## How the code is laid out

`AGENTS.md` (also linked as `CLAUDE.md`) explains the layout, the conventions that matter (the UI talks to the backend over HTTP, LLM access goes through `config.get_llm_client()`, caches are keyed by schema version), and where the backlog lives. `ARCHITECTURE.md` has the file-by-file map. Design decisions that changed the shape of the system have an ADR in `docs/adr/`; add one if yours does.

## Style

Ruff is the only formatter of record (`uv run ruff check . --fix`). Match the file you are editing. User-facing text uses short sentences and no em dashes.
