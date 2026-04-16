# Socratic AI Study Assistant

Socratic AI turns your sources into a grounded learning workflow with generation plus chat follow-up. It supports three intake modes: video only, documents only, or video plus documents.

## Learning Goal

The product goal is structured, engaged learning rather than passive summarization. It is designed to help you build concepts on top of each other, connect intertwined ideas across sources, and move from first-principles understanding to practical transfer. Deep dives, cross-source synthesis, and Socratic chat all push toward mechanism-level clarity without repetitive restatement.

## Design Choice (Short Version)

The app is intentionally retrieval-first and stage-based instead of one-shot chat. Ingestion stores transcript and document chunks, processing builds source concepts and summaries, and generation produces structured learning artifacts. This keeps answers tied to your material, makes failures observable by stage, and allows selective skipping of expensive comparative calls when a run has only one source type.

## What It Supports

- Upload one video file.
- Paste one YouTube URL (downloaded with yt-dlp, then transcribed through the same Whisper path).
- Upload one or more documents.
- Run with:
  - video only
  - documents only
  - video + documents
- Unified Socratic chat over generated and retrieved context.

## Implementation Flow

1. Ingestion
- `POST /upload` accepts optional `video`, optional `video_url`, and optional `documents`.
- Constraint: at most one video source per request (`video` XOR `video_url`).
- YouTube URLs are downloaded via yt-dlp and converted to WAV with ffmpeg.
- Whisper transcription produces timestamped segments for video sources.
- Documents are parsed (`pdf`, `txt`, `md`) and chunked.

2. Processing and linking
- `/process` now runs with any non-empty source set.
- For mixed mode (video + documents): process all sources and link video concepts to document concepts.
- For single-source mode: process only, skip linking.

3. Tailored generation
- `/generate-tailored-learning` always returns source learning sections and quiz/insight payloads.
- Mixed mode uses full comparative stages (intersections, quiz, comparative analysis, scenarios).
- Non-comparative mode uses deterministic, grounded synthesis objects and skips comparative model calls.

### Depth and Non-Redundancy Strategy

- Source-level under-surface generation now runs in two passes:
  - pass 1: structured under-surface generation (diagnostic checklist + key term breakdown)
  - pass 2: non-redundant expansion pass that explicitly avoids repeating summary/deep-dive text while adding mechanism-level explanation
- Under-surface text is normalized to continuous prose (no markdown heading/list artifacts) before storage and rendering.
- Cross-source intersections and synthesis are now refined for depth and novelty:
  - intersection explanations get non-redundant expansion against source summaries/deep dives
  - synthesis and comparative analysis get additional expansion/refinement passes with overlap suppression
- Cross-source prompt schemas were widened so responses can carry more useful detail without truncating important context.

4. Interaction
- `/ask` uses source-scoped retrieval and generated context.
- Streamlit quick actions can auto-navigate to chat and auto-submit the generated query.

## Streamlit UX Notes

- Step 1 lets users choose one video source mode: upload or YouTube link.
- After a source is saved, the input is replaced by a selected-state card plus replace button.
- Step 1 and Step 2 each provide an explicit skip button.
- Step 3 enables generation only when at least one source is selected.
- Source and cross-source sections include “Elaborate further” actions that auto-jump to the chatbox and auto-send context-rich prompts.
- New assistant responses in chat auto-scroll into view and get a brief highlight flash for visibility.
- Cross-source grounded evidence is rendered as mechanism-grounded evidence cards with source attribution and concept signals.

## Cost Expectations (gpt-4o-mini + current defaults)

Assumptions:
- Prompt/response sizes are based on current prompts and average medium-length material.
- Whisper is treated as $0.00 in this project estimator.
- Cost logging is enabled and includes generation, linking, processing, retrieval, interaction.

Average ranges per run:

| Scenario                               | Typical model work                                                 |  Estimated cost |
| -------------------------------------- | ------------------------------------------------------------------ | --------------: |
| Video only                             | Transcription + per-source generation + chat-ready retrieval setup |  $0.12 to $0.35 |
| Documents only (1-3 docs)              | Per-source processing/generation, no video linking                 |  $0.10 to $0.45 |
| Video + documents (1 video + 1-3 docs) | Full processing + linking + comparative generation                 |  $0.55 to $1.80 |
| One chat ask                           | 1 embedding + 1 interaction completion                             | $0.001 to $0.01 |

Why mixed mode costs more:
- Concept linking plus comparative generation stages dominate token usage.
- Single-source runs skip those stages by design.

## Quick Start

1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment

```env
OPENAI_API_KEY=your_key_here
```

Optional:

```env
FRONTEND_REQUEST_TIMEOUT=600
```

3. Run API

```bash
uvicorn main:app --reload
```

4. Run Streamlit

```bash
streamlit run frontend/app.py
```

## Validation and Tests

Run unit tests:

```bash
python -m pytest tests/test_json_reliability.py tests/test_optional_source_modes.py -v
```

Run end-to-end validator:

```bash
python scripts/validate_pipeline.py --api-base-url http://127.0.0.1:8000
```

## Key Files

- `frontend/app.py` : wizard UX, skip controls, auto-jump and auto-submit behavior.
- `routes/upload.py` : optional upload contract with `video_url` support.
- `app/ingestion.py` : yt-dlp download, WAV conversion, Whisper ingestion.
- `app/workflow.py` : optional-source processing/linking orchestration.
- `app/generation.py` : mixed-mode comparative path and non-comparative cost-saving path.
- `app/cost_logging.py` : token usage logging per stage/model.
- `scripts/validate_pipeline.py` : integration and usage summary checks.

## Operational Notes

- All API calls are user-scoped via `X-User-ID`.
- ffmpeg is required for audio extraction.
- yt-dlp is required for YouTube URL ingestion.
- Cached processing and generation can reduce repeated-call cost.
