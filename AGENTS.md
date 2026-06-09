# AGENTS.md

Operational guide for anyone — human or AI agent — working in this repo. It answers *how to work
here*. For *how the system works* see [`ARCHITECTURE.md`](ARCHITECTURE.md), the call-by-call
[`docs/pipeline.md`](docs/pipeline.md), and the decision records in [`docs/adr/`](docs/adr/).

> `CLAUDE.md` is a symlink to this file.

## What this is

Socratic AI turns videos and documents into a structured learning pack + a source-grounded Socratic
chat. It's a **single uvicorn process**: FastAPI backend with a NiceGUI UI mounted on the same app;
SQLite + ChromaDB for storage; OpenAI by default, Ollama optional. Python ≥ 3.11.

## Run / develop

```bash
uv run socratic-ai            # supported path; resolves Python + deps, opens http://localhost:8000
# or:
python run.py                 # after `pip install .` in a venv
uvicorn main:app --reload     # dev with auto-reload
```

- A `.env` with `OPENAI_API_KEY=sk-...` is the simplest start (see `.env.example`). Or run keyless
  against Ollama: set `LLM_PROVIDER=ollama` / `EMBEDDING_PROVIDER=ollama` (Whisper still needs OpenAI).
- Provider/model/key are also switchable at **runtime** via the in-app ⚙ Settings page (persists to the
  user config dir, no restart). Env vars are only the bootstrap default.
- `ffmpeg` must be on `PATH` for video ingestion; `yt-dlp` (a dependency) handles YouTube URLs.

## Test

```bash
pytest tests/                 # unit + smoke (no network; mocked LLM)
pytest tests/ -m e2e          # end-to-end; replays recorded calls from tests/_cassettes/
```

- `asyncio_mode = auto` and the NiceGUI user plugin are configured in `pyproject.toml`.
- E2E tests use a VCR-style cassette (`tests/_recording_client.py`, `tests/_cassettes/`); they run
  offline unless you re-record against a real `OPENAI_API_KEY`.
- Diagnostic/validation tooling lives in `scripts/` — see [`scripts/README.md`](scripts/README.md).
  Some scripts need a running server and/or an API key.

## Project layout

| Dir | What |
|---|---|
| `app/` | Backend logic: ingestion, generation, interaction, retrieval, ledger, llm_client, models. |
| `routes/` | Thin FastAPI endpoints (`/upload`, `/generate-tailored-learning`, `/ask`, `/settings`). |
| `frontend/` | NiceGUI UI (`ui.py`, `state.py`, `api_client.py`, `format.py`, `pages/`, `components/`). |
| `prompts/` | System prompts + JSON schemas (`__init__.py`) and the section-contract registry (`sections.py`). |
| `scripts/` | Diagnostics, validation, regression tooling (dev-only). |
| `tests/` | pytest suite (smoke, e2e cassette, quality, ui, settings, ingestion, json). |
| `docs/` | `pipeline.md`, `adr/`, `history/` (superseded docs). |
| root | `run.py`, `main.py`, `config.py`, deploy configs (`Dockerfile`, `render.yaml`, `railway.json`). |

## Conventions

- **Loopback boundary.** The UI calls the backend over HTTP (`frontend/api_client.py`), never importing
  `app.*` directly. Keep it that way so validation/auth/error-mapping stay in the routes.
- **LLM access** goes through `app/llm_client.py` via `config.get_llm_client()` / `get_aux_client()` and
  the `config.*_model()` getters — so runtime settings changes take effect. (Transcription still uses the
  bare `config.openai_client`; `apply_settings` keeps it coherent.)
- **Prompts** live in `prompts/`. Section roles/budgets/non-overlap rules are data in
  `prompts/sections.py`, not scattered prose — edit there, not inline.
- **Caching / regeneration.** Generated artifacts are cached in SQLite keyed by a schema version. To
  force regeneration after changing output shape, bump `GENERATION_SCHEMA_VERSION` (or
  `LEDGER_SCHEMA_VERSION`) in `app/generation.py`. See [ADR-0004](docs/adr/0004-cache-via-schema-version.md).
- **Tables** are created idempotently on startup (`ensure_*_tables`, `CREATE IF NOT EXISTS` + additive
  `ALTER`). New columns go in those migration dicts, never as destructive changes.
- **Runtime/local artifacts** (`app.db`, `chroma_data/`, `uploads/`, `a2/`) are gitignored — don't commit
  them.

## Known backlog

Honest, currently-unfixed issues (design smells + thin test coverage) are listed in
[ARCHITECTURE.md → Known issues](ARCHITECTURE.md#7-known-issues--backlog), with a "Recently resolved"
section right below it and rationale captured in the relevant ADRs. Remaining items are lower priority
(aux-model default clarity, cache-version changelog, an end-to-end provider-switch test).
