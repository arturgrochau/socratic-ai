# Deployment

## Environment Variables
Set these variables in your deployment platform:

- `OPENAI_API_KEY` (required)
- `DATABASE_URL` (default: `sqlite:///./app.db`)
- `CHROMA_PERSIST_DIR` (default: `./chroma_data`)
- `GENERATION_MODEL` (default: `gpt-4o-mini`)
- `CHAT_MODEL` (default: `gpt-4o-mini`)
- `RETRIEVAL_MODEL` (default: `text-embedding-3-small`)
- `LLM_PROVIDER` / `EMBEDDING_PROVIDER` / `WHISPER_PROVIDER` (`openai` or `ollama`)
- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `USER_ID_HEADER` (default: `X-User-ID`)
- `ENABLE_COST_LOGGING` (`true` or `false`)
- `CACHE_PROCESSED_SOURCES` (`true` or `false`)
- `DEPLOY_ENV` (default: `production`)

> These env vars are bootstrap defaults. At runtime, provider/model/key settings
> can also be changed from the in-app **Settings** page (persisted to the user
> config dir). See `.env.example` for the full list.

## Runtime Dependencies

- `ffmpeg` must be present at runtime (already installed in the Dockerfile).
- `yt-dlp` is installed as a project dependency (`pyproject.toml`) and is required for YouTube URL ingestion.
- YouTube mode downloads one media source and then reuses the same Whisper pipeline used for uploaded videos.

## Local Docker Run

```bash
docker build -t socratic-ai .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e USER_ID_HEADER="X-User-ID" \
  socratic-ai
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Local One-Command Run (API + UI, one process)

From the repository root:

```bash
uv run socratic-ai
```

This starts the unified app (FastAPI API + NiceGUI UI) at `http://127.0.0.1:8000`
and opens your browser. Set `SOCRATIC_OPEN_BROWSER=false` to skip the browser, or
`PORT` to change the port. Use `Ctrl+C` to stop.

## Render

Use the included `render.yaml` Blueprint.

1. Create a new Render Blueprint service from this repo.
2. Set `OPENAI_API_KEY` as a secret environment variable.
3. Deploy and verify `GET /health`.

## Railway

Use the included `railway.json`.

1. Create a new project from this repo.
2. Set `OPENAI_API_KEY` in project variables.
3. Deploy and verify `GET /health`.

## Client Header Requirement

All API calls must include a non-empty user header.

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-User-ID: demo-user" \
  -d '{"session_id":"demo-session","source_ids":[1,2],"query":"What are the key ideas?","top_k":8}'

## Upload Modes

`POST /upload` supports all three source combinations:

- Video only (multipart `video` file OR form `video_url`)
- Documents only (multipart `documents`)
- Video + documents

Constraint: provide at most one video source per request (`video` XOR `video_url`).
```

## Validation Script

Run end-to-end validation after deployment:

```bash
python scripts/validate_pipeline.py --api-base-url http://127.0.0.1:8000
```
