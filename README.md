# Socratic AI Study Assistant

Socratic AI turns one video lecture plus supporting documents into a guided, source-grounded learning system.

The short version:
- It helps you study one concept deeply, not just skim answers.
- It links ideas across sources, then challenges your thinking.
- It keeps everything in one interface so exploration is focused and continuous.

## Why This Exists (Learning Philosophy)

Most AI tools optimize for speed of answers.
This project optimizes for depth of understanding.

Its philosophy is close to "sapere aude" (dare to know):
- think for yourself,
- test assumptions,
- connect ideas across contexts,
- ask one better question after each answer.

It also follows first-principles learning:
- break ideas into mechanisms,
- identify assumptions and failure modes,
- rebuild understanding from fundamentals,
- then apply it in concrete scenarios.

That is why the app is designed as a full learning loop (summaries, intersections, diagnostics, quiz, Socratic follow-up), not a single chatbot text box.

## What Makes This Different From Typical AI Use

Typical usage: ask one question, get one answer, move on.

Socratic AI usage:
1. Ingest real study material.
2. Build structured concept representations.
3. Link concepts between sources.
4. Generate cross-source synthesis and challenge questions.
5. Continue exploration in a grounded Socratic chat.

This creates a "single-concept deep dive" experience:
- you can stay focused on one thing you want to learn,
- inspect it from multiple sources,
- test your understanding,
- and keep drilling deeper in the same UI.

## System Overview

### Frontend
- Streamlit dashboard for upload, generation, synthesis, quiz, and chat.
- Guided sections:
  - Video
  - Documents
  - Cross-Source Synthesis & Assessment
  - Socratic Chatbox

### Backend
- FastAPI routes for upload, processing/linking, generation, and ask.
- SQLite + Chroma persistence.
- User scoping via `X-User-ID`.

### Core pipeline
1. **Upload and ingestion**
   - Save files and extract text/audio.
   - Transcribe video audio via Whisper.
2. **Processing**
   - Chunk source text.
   - Extract concepts and key ideas.
   - Create source summaries.
3. **Linking**
   - Build candidate concept pairs by token overlap.
   - Classify relationship per pair (reinforces, overlaps, etc.).
4. **Tailored generation**
   - Per-source: title, deep dive, reflection, under-surface explanation.
   - Cross-source: intersections, synthesis, comparative analysis, scenarios, quiz.
5. **Socratic interaction**
   - Retrieval over stored chunks/concepts when needed.
   - Direct answer + grounded support + follow-up reflection.

## Prompt and Design Choices (Why It Is Built This Way)

The prompts are intentionally not just "summarize this":
- They require grounded outputs tied to provided material.
- They enforce mechanism-level explanations (not shallow paraphrase).
- They include reflection and diagnostics to promote active learning.
- They ask for practical transfer (application scenarios), not only theory.

Design choice rationale:
- **Cross-source intersections** improve transfer learning.
- **Diagnostic checklist + key-term breakdown** improves self-assessment.
- **Harder quiz with distractors** improves discrimination, not memorization.
- **Socratic follow-up** keeps exploration open-ended and learner-driven.

## Reliability and Efficiency Model

Generation now includes reliability controls for long runs:
- Structured JSON parsing safeguards.
- Bounded automatic retries per generation stage.
- Stage-level telemetry with `run_id`, stage name, attempt count, status, duration, and error detail.
- Safe fallback decoding for cached JSON rows.

Why this matters:
- Prevents a single malformed model response from wasting the whole run.
- Makes failures actionable (`run_id` + failing stage).
- Improves observability for cost/performance tuning.

## Average Model Calls (What to Expect)

Actual call volume depends on number of documents, concept density, and cache hits.

For one video + `D` documents:

- **Ingestion**
  - Whisper transcription: `1` call (video).
- **Processing**
  - Concept extraction + source summary: `2 x (1 + D)` calls.
- **Per-source generation**
  - Title + deep dive + reflection + under-surface: `4 x (1 + D)` calls.
- **Cross-source generation**
  - Insights + quiz + comparative analysis + scenarios: `4` calls.
- **Linking**
  - Up to `30` concept-comparison calls per video-document pair (bounded by candidate cap), often fewer with cache reuse.

A practical cold-start estimate for one video + two documents (`D = 2`):
- Base calls (without linking comparisons): `23`
- Plus linking comparisons: variable (typically much lower than max, highest cost driver when concept sets are large)

Ask/chat calls are separate from generation and typically add:
- 1 embedding call (retrieval path), and
- 1 chat completion (sometimes 2 in fallback/retry branches).

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Create a `.env` file with at least:

```env
OPENAI_API_KEY=your_key_here
```

Optional runtime settings are defined in `config.py`.
For long frontend waits, you can also set:

```env
FRONTEND_REQUEST_TIMEOUT=600
```

### 3) Run backend API

```bash
uvicorn main:app --reload
```

Backend: `http://127.0.0.1:8000`

### 4) Run frontend

```bash
streamlit run frontend/app.py
```

## Typical User Flow

1. Open the Streamlit app.
2. Set API URL and user ID in the sidebar.
3. Upload one video and one or more documents.
4. Click **Generate tailored Socratic learning**.
5. Explore the dashboard sections.
6. Use Socratic chat to deepen one concept at a time.

## Validation

Run end-to-end validation with test assets:

```bash
python scripts/validate_pipeline.py
```

Validation includes generation quality checks and usage summaries.

## Notes

- Data is user-scoped using the `X-User-ID` header.
- Caching and schema versioning are built into processing and generation.
- Local persistence uses SQLite by default.
- Retrieval storage uses Chroma for concept/raw chunk embeddings.
