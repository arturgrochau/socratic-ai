# Socratic AI Study Assistant

Socratic AI turns your sources into a grounded learning workflow with generation plus chat follow-up. It supports three intake modes: video only, documents only, or video plus documents.

## Learning Goal

The product goal is structured, engaged learning rather than passive summarization. It is designed to help you build concepts on top of each other, connect intertwined ideas across sources, and move from first-principles understanding to practical transfer. Deep dives, cross-source synthesis, and Socratic chat all push toward mechanism-level clarity without repetitive restatement.

## Design Choice (Short Version)

The app is intentionally retrieval-first and stage-based instead of one-shot chat. Ingestion stores transcript and document chunks, processing builds source concepts and summaries, and generation produces structured learning artifacts. This keeps answers tied to your material, makes failures observable by stage, and allows selective skipping of expensive comparative calls when a run has only one source type.

## In-Depth Pipeline Overview

Socratic AI is designed to break down dense educational material into a highly readable, first-principles learning journey. It does this by processing each source individually before synthesizing them together using a powerful language model (GPT-4o).

### Per-Source Analysis (The 4-Section Framework)
For each document or video, the pipeline generates a structured breakdown:
1. **Summary:** A concise overview of the core arguments and narrative.
2. **Deep Dive:** A detailed exploration of the mechanics and evidence behind the source's claims.
3. **Key Concepts:** A clean, bulleted list of essential terminology defined in layman's terms. This section includes a "Read more" expander containing a **first principles synthesis**—a cohesive teaching paragraph that seamlessly interrelates all the key terms to explain *how* the system works from the ground up.
4. **Reflection:** A set of Socratic reflection questions designed to test comprehension, complete with detailed explanations that reveal the nuances of the answers.

### Cross-Source Synthesis & Consolidation
When multiple sources are uploaded (e.g., a video lecture and a textbook chapter), the pipeline uses a single, high-density prompt (`synthesis_consolidator.py`) passed to GPT-4o. This step ingests the summaries and deep dives from all sources at once, producing:
- **Flowing Synthesis Prose:** 3-5 cohesive paragraphs that contrast, compare, and build upon the sources. It eliminates the need for rigid headers or repetitive mapping, reading naturally like a personalized tutor.
- **Integrated Reflection Quizzes:** Carefully crafted multiple-choice questions focusing on the *intersection* of the sources. The explanations are written to be informative and natural, avoiding robotic "Distractor A is wrong because..." phrasing.

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

- **First Principles Synthesis:** Instead of listing disconnected definitions, the pipeline generates a single, cohesive paragraph that teaches the foundational mechanism behind the concepts.
- **Unified Cross-Source Pass:** Replaced the previous rigid, multi-stage heuristic loops (which were prone to generating bloated, repetitive "traps") with a single, highly optimized GPT-4o call. This ensures natural text flow and maximum conceptual density while drastically reducing API calls.
- **Density Controls:** Prompts include strict length boundaries (e.g., exactly 3-5 paragraphs of 4-6 sentences) to guarantee consistently high signal-to-noise ratios.

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

# Repeated-run redundancy audit (cross-source overlap telemetry)
python scripts/evaluate_redundancy.py \
  --api-base-url http://127.0.0.1:8000 \
  --video /path/to/video.mp4 \
  --document /path/to/document.pdf \
  --iterations 10
```

## Key Files

- `frontend/app.py` : wizard UX, skip controls, auto-jump and auto-submit behavior.
- `routes/upload.py` : optional upload contract with `video_url` support.
- `app/ingestion.py` : yt-dlp download, WAV conversion, Whisper ingestion.
- `app/workflow.py` : optional-source processing/linking orchestration.
- `app/generation.py` : mixed-mode comparative path and non-comparative cost-saving path.
- `app/cost_logging.py` : token usage logging per stage/model.
- `scripts/validate_pipeline.py` : integration and usage summary checks.
- `scripts/evaluate_redundancy.py` : repeated-run overlap and claim-overlap audit with aggregate stats.

## Operational Notes

- All API calls are user-scoped via `X-User-ID`.
- ffmpeg is required for audio extraction.
- yt-dlp is required for YouTube URL ingestion.
- Cached processing and generation can reduce repeated-call cost.
