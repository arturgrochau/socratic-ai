# Pipeline: exact runtime trace

This is the call-by-call walk through every stage, with `file:function` references. It complements
the high-level map in [`/ARCHITECTURE.md`](../ARCHITECTURE.md). Line numbers drift — trust the
function names.

Models per stage: every role resolves through `config` accessors against the active mode's bundle
([ADR-0010](adr/0010-local-by-default.md)). **Local mode (default):** all chat-model roles →
`qwen3:30b-a3b-instruct-2507-q4_K_M` (one resident model; the aux split is a no-op locally by
design), embeddings → `nomic-embed-text`, transcription → parakeet-mlx/mlx-whisper on-device.
**API mode:** ledger extraction → `config.get_aux_model()` (`gpt-4o-mini`); consolidation +
insights + quiz → `config.generation_model()`; chat → `config.chat_model()`; embeddings →
`text-embedding-3-small`; transcription → `whisper-1`. Section audit and chat depth-tier are
**pure code** (no model): `app/section_validator.py` and `interaction._heuristic_depth`.
In-flight calls are capped per provider by `config.llm_call_slots` (8 hosted / 2 local); Ollama
calls request structured outputs (`format` = bare schema), `num_ctx` 16k (32k for
consolidation/insights/quiz via `large_context=True`), and `keep_alive=30m`.

---

## 0. Startup

1. `run.py:main` — reads `HOST`/`PORT`, optionally schedules a browser open, calls
   `uvicorn.run("main:app")`. (`uv run socratic-ai` resolves to `run:main` via `pyproject.toml`.)
2. `main.py` (module import) — creates `FastAPI`, includes the four routers
   (`upload`, `interaction`, `workflow`, `settings`).
3. `main.py:on_startup` (`@app.on_event("startup")`) — `run_startup_checks()` (SQLite `SELECT 1` +
   Chroma list) then the table initializers: `ensure_cost_logging_tables`, `ensure_ingestion_tables`,
   `ensure_interaction_tables`, `ensure_generation_tables` (which also calls `ensure_ledger_table`).
   All use `CREATE TABLE IF NOT EXISTS` + additive `ALTER` migrations — safe to run every boot.
4. `main.py` (after app defined) — `frontend.ui.init_ui()` registers the `/` and `/settings` NiceGUI
   pages, then `ui.run_with(app, …)` mounts the UI on the same app.

`config.py` import-time side effects: load `.env`, resolve `Settings` (env overlaid with the persisted
user config), warn if OpenAI is selected but no key is set, create the SQLite engine + Chroma client.

---

## 1. Ingest — `POST /upload`

Uploads happen **on add**, one item at a time (not buffered until Generate): adding a document/video in
`frontend/pages/build.py` calls `frontend/api_client.py:upload_document` / `upload_video_file` /
`upload_video_url` (multipart loopback POST) → `routes/upload.py:upload_content` →
`app/ingestion.py:ingest_upload_bundle`, and only the returned `{name, source_id}` ref is kept in tab
state. Document parsing and DB writes run via `asyncio.to_thread` so a large PDF doesn't block the loop.

`ingest_upload_bundle(user_id, video_file, video_url, document_files)`:
1. **Validate** — exactly one of {video file, YouTube URL} (not both); at least one source overall.
   `validate_video_file`, `validate_youtube_url`, `validate_document_files`.
2. **Acquire media** — file: `save_upload_file`; YouTube URL: `download_youtube_video` (yt-dlp Python
   API, falling back to the `yt-dlp` CLI) → bestaudio.
3. **Video → text** (wrapped in `asyncio.wait_for(..., INGESTION_VIDEO_STEP_TIMEOUT_SECONDS)`):
   `extract_audio_from_video` (ffmpeg → 16 kHz mono wav) → `transcribe_audio_with_whisper`. Files over
   `WHISPER_MAX_REQUEST_BYTES` (24 MB) are split into `WHISPER_CHUNK_SECONDS` segments and transcribed
   with running time offsets. Each Whisper call (`_transcribe_whisper_file`) passes
   `timeout=WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS` and retries up to `WHISPER_TRANSCRIPTION_MAX_RETRIES`.
4. **Docs → text** — `parse_document_text`: PDF via PyMuPDF (`extract_pdf_text_with_pages`, page
   numbers preserved), `.txt`/`.md` via `extract_note_text`.
5. **Chunk + store** — `store_video_transcript` / `store_document_pages` insert the `sources` row and
   the overlapping chunks (`build_transcript_chunks` / `build_document_chunks` →
   `_store_source_chunks`, `RAW_CHUNK_TARGET_CHARS=2400`, `RAW_CHUNK_OVERLAP_CHARS=320`). Chunks carry
   timestamps/page numbers; the old write-only `transcripts`/`document_pages` tables are gone.
6. **Embed in the background** — `_schedule_background_embedding` fires
   `upsert_retrieval_embeddings` as a fire-and-forget task so the first chat turn retrieves instantly.
7. **Return** `IngestionResponse` with `source_id`s — the UI stashes them for the generate call.

No LLM calls except Whisper. Errors map to HTTP 400 (`ValueError`) / 422 (`RuntimeError`).

---

## 2. Generate — `POST /generate-tailored-learning`

UI → `frontend/api_client.py:generate` → `routes/workflow.py:generate_tailored_learning_endpoint` →
`app/generation.py:generate_tailored_learning(video_source_id, document_source_ids, user_id)`.

Wrapped in `session_log(run_id)`. Orchestration is **parallel by default**: independent LLM calls run
through `_run_parallel` (ThreadPoolExecutor + `contextvars` propagation), with a global semaphore
capping concurrency at `MAX_PARALLEL_LLM_CALLS=8`. Steps:

1. `_normalize_ids` — drop non-positive ids, dedup docs, require ≥1 source.
2. **All sources in parallel** → `_build_source_learning_section(source_id, user_id, run_id)` per
   source (video + every doc concurrently):
   1. **Cache check** — if `CACHE_PROCESSED_SOURCES`, `_load_cached_source_learning_section`; a cache
      hit (matching `GENERATION_SCHEMA_VERSION`, non-empty summary + key_terms) returns immediately.
   2. **Load chunks** — `_load_source_chunks` (ordered by `chunk_index`).
   3. **Ledger (map-reduce, [ADR-0008](adr/0008-map-reduce-ledger.md))** — `load_ledger` (cached,
      `LEDGER_SCHEMA_VERSION=2`); on miss: `_group_chunks_into_windows` (3 chunks if ≤10 total, else
      4) → `_extract_source_ledger`: every window extracted **in parallel**, context-free (aux model,
      `LEDGER_EXTRACTION_JSON_SCHEMA`); results merged in window order by `parse_units` +
      `append_units` (0.72 token-overlap dedup, `app/ledger.py`) — the reduce step is pure code, no
      LLM call. Result is `store_ledger`'d.
   4. **Consolidate** — `_consolidate_ledger` renders the ledger and calls
      `_run_structured_generation_step` (main model, `CONSOLIDATION_JSON_SCHEMA`). Section contracts
      come from `prompts/sections.py:PER_SOURCE_SECTIONS`.
   5. **Validate (code, not LLM — [ADR-0009](adr/0009-code-validator-and-heuristic-tier.md))** —
      `section_validator.validate_arc(PER_SOURCE_SECTIONS, …)` checks sentence/item budgets and
      cross-section restatement in microseconds. On violations, consolidation is re-run **once** with
      them appended as corrections.
   6. **Assemble + store** — build `SourceLearningSection` and `_store_source_learning_section` (upsert).
3. **Enrichment — two focused calls in parallel** (insights ∥ quiz; same wall-clock as one call,
   better mini-model quality per artifact):
   - **≥2 sources** → `_generate_synthesis`: `_merge_source_ledgers` (globally unique ids, no
     cross-source dedup) + source legend → in parallel: the **insights** call
     (`INSIGHTS_SYSTEM_PROMPT`/`INSIGHTS_JSON_SCHEMA`: takeaways, synthesis, intersections,
     scenarios) and the **quiz** call (`QUIZ_SYSTEM_PROMPT`/`QUIZ_JSON_SCHEMA`, comparison-flavored).
     `validate_arc(CROSS_SOURCE_SECTIONS, …)` → one insights re-run on violations; quiz over-runs are
     trimmed in code.
   - **One source** → `_generate_single_source_deepening`: the same parallel pair with
     `SINGLE_SOURCE_INSIGHTS_SYSTEM_PROMPT` (intersections always empty) and a single-source-flavored
     quiz call. See [ADR-0007](adr/0007-single-source-enrichment.md).
   - **Post-process** — `_ground_intersections(payload, valid_source_ids)` prunes citations to
     sources actually loaded and keeps only intersections spanning ≥2 distinct real sources
     ([ADR-0005](adr/0005-cross-source-grounding.md)); validate `QuizQuestion`s/`ApplicationScenario`s;
     cap `key_takeaways` at 3. Both paths cache via `_store_combined_learning_sections` (sentinel
     `video_source_id=0` when there's no video).
4. **Return** `GenerateTailoredLearningResponse` (video section, document sections, insights, quiz).
   A stage that fails all `GENERATION_MAX_STAGE_ATTEMPTS=3` attempts raises `GenerationStageError`
   → HTTP 422 with `{run_id, stage, attempt, reason}`.

Every LLM call goes through `_run_structured_generation_step`, which retries up to 3×, logs usage
(`log_api_usage`) and per-stage timing (`log_generation_stage_event` + `session_logger`).

**Call/latency budget (fresh 1-doc run, ~5 windows):** 5 parallel ledger calls + 1 consolidation +
2 parallel enrichment calls = 8 calls in ~4 serial round-trips (vs 9 calls in 9 serial round-trips
before the redesign — and no quadratic prior-ledger re-send).

---

## 3. Chat — `POST /ask`

UI (`frontend/pages/chat.py`) → `frontend/api_client.py:ask` → `routes/interaction.py:ask_endpoint`
→ `app/interaction.py:handle_user_query(query, source_ids, session_id, user_id, top_k=8)`:

1. Normalize inputs; load or create the `InteractionSessionState` (`interaction_sessions`); reject a
   source-set mismatch against an existing session. Keep the last `MAX_PROMPT_TURNS=4` turns.
2. **Depth tier (heuristic, zero LLM calls — [ADR-0009](adr/0009-code-validator-and-heuristic-tier.md))**
   — `_heuristic_depth` routes lookup/explain/analyze/deepen from keywords, query length, and prior
   turns. `chat_context_scale(tier)` scales both the generated-context budget and `top_k`. A chat turn
   is now exactly **one** LLM call.
3. **Generated context** — `_load_generated_learning_context` pulls summary/deep-dive/key-terms from
   `source_learning_sections` + latest cross-source synthesis, truncated to the scaled budget.
4. **Retrieval** — `retrieve_context` (`app/retrieval.py`): `load_raw_chunks_for_sources` →
   `upsert_retrieval_embeddings` (normally a no-op: chunks are embedded in the background at ingest
   by `ingestion._schedule_background_embedding`; this lazy upsert covers any chunks the background
   task missed, embedding them into the Chroma
   collection) → embed the query → `collection.query` (over-fetch `top_k * QUERY_MULTIPLIER`, filter
   by source_id/user_id) → score floor (`MIN_RAW_HIT_SCORE`) + per-chunk dedup → top-k.
   `build_context_text` formats hits with provenance (source #, timestamp/page, chunk, score).
5. **Answer** — `_run_chat_completion` (main chat model, `CHAT_JSON_SCHEMA`, `temperature=0.3`) with
   query + generated context + retrieved context + recent turns. Empty answers get a graceful
   fallback message.
6. Append the turn (cap at `MAX_STORED_TURNS=20`), persist, return `AskResponse` (answer + model_name).

**Quiz handoff:** `frontend/components/quiz.py` seeds `pending_chat_prompt` in tab state with what the
learner missed; `chat.py` auto-submits it as the first chat query.

---

## 4. Settings — `GET/PUT /settings`, `GET /settings/test`

`frontend/pages/settings.py` → `frontend/api_client.py` → `routes/settings.py`:
- `GET /settings` → `read_settings` (`SettingsView`, key always masked via `_mask_key`).
- `PUT /settings` → `update_settings` → `config.apply_settings(new)`: under `_settings_lock`, swap
  `_settings`, clear the LLM client cache, reset the Ollama reachability probe, rebuild the
  bare `openai_client`, refresh the backwards-compat module constants, and persist to the user config
  dir. Takes effect immediately for subsequent requests — no restart. (A blank key preserves the
  stored one.)
- `GET /settings/test?provider=` → `test_connection`: pings the Ollama daemon's `/api/tags` or lists
  OpenAI models, for the panel's "Test" button.
