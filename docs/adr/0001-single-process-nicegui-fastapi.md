# ADR 0001 — Single process: NiceGUI mounted on FastAPI

Status: accepted (v2.1, supersedes the earlier Streamlit-on-a-second-port design)

## Context

The app needs a UI and a JSON API. The original design (see `docs/history/`) ran a Streamlit frontend
as a separate process on its own port, talking to a FastAPI backend. That meant two processes to
launch, two ports, two failure modes, and a frontend framework whose rerun-everything model fought the
multi-step upload→generate→chat flow.

## Decision

Run **one** uvicorn process. Build the UI with [NiceGUI](https://nicegui.io) and mount it onto the same
FastAPI app via `ui.run_with(app, …)` (`main.py`). The UI still talks to the backend over **loopback
HTTP** (`frontend/api_client.py`) rather than importing `app.*` directly, so the routes' validation,
auth header, and error mapping apply uniformly and the API stays independently usable.

## Consequences

- One thing to start (`uv run socratic-ai`), one port (8000), one log stream.
- Per-tab UI state lives in NiceGUI storage (`app.storage.tab`/`.user`, `frontend/state.py`) instead of
  Streamlit `session_state` — explicit and copy-on-init so tabs never share mutable defaults.
- The loopback hop costs a serialization round-trip, but generation is dominated by LLM latency, so the
  overhead is negligible and the clean boundary is worth it.
- Trade-off: the UI and API share a process, so a hard crash takes both down. Acceptable for a local
  single-user app.

## Alternatives considered

- **Keep Streamlit + FastAPI (two processes)** — rejected: operational overhead and rerun model.
- **Direct in-process function calls (no HTTP)** — rejected: would duplicate validation/auth and couple
  the UI to internal signatures.
- **A JS SPA (React/Next) + API** — rejected: far more build/tooling weight than a local tool needs.
