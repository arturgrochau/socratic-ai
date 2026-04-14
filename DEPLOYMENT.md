# Deployment

## Environment Variables
Set these variables in your deployment platform:

- `OPENAI_API_KEY` (required)
- `DATABASE_URL` (default: `sqlite:///./app.db`)
- `CHROMA_PERSIST_DIR` (default: `./chroma_data`)
- `PROCESSING_MODEL` (default: `gpt-4o-mini`)
- `LINKING_MODEL` (default: `gpt-4o-mini`)
- `RETRIEVAL_MODEL` (default: `text-embedding-3-small`)
- `INTERACTION_MODEL` (default: `gpt-4o-mini`)
- `USER_ID_HEADER` (default: `X-User-ID`)
- `ENABLE_COST_LOGGING` (`true` or `false`)
- `CACHE_PROCESSED_SOURCES` (`true` or `false`)
- `DEPLOY_ENV` (default: `production`)

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

## Local One-Command Dev Run (API + Streamlit)

From the repository root:

```bash
./socratic-ai
```

This starts:

- API at `http://127.0.0.1:8000`
- Streamlit at `http://127.0.0.1:8501`

Use `Ctrl+C` once to stop both together.

Optional flags:

```bash
./socratic-ai --no-open
./socratic-ai --no-reload
```

Alias launcher is also available:

```bash
./socratic_ai
```

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
```

## Validation Script

Run end-to-end validation after deployment:

```bash
python scripts/validate_pipeline.py --api-base-url http://127.0.0.1:8000
```
