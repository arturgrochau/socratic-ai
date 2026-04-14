# Study Assistant Implementation Plan - OpenAI Version

This document guides an AI coding agent through iterative, section-by-section development of the study assistant using **OpenAI API** (gpt-4o, gpt-4o-mini, and whisper-1). Each phase is self-contained, includes acceptance criteria, and follows the original feasibility stages. Work strictly in order. After completing a phase, run the specified tests, verify acceptance criteria, then move to the next phase.

The project uses a minimal Python + FastAPI backend with Streamlit (or Next.js) frontend. No LangChain or heavy frameworks.

Assume a `.env` file already exists containing:

OPENAI_API_KEY=...

## Phase 1: Project Setup and Environment

Create the folder structure and core configuration files.

**Tasks:**
- Initialize a new Python project with FastAPI, Uvicorn, and required dependencies.
- Create `.env` and load it with python-dotenv.
- Set up database connection (SQLite for MVP or Supabase) and Chroma vector store (local).
- Create a basic FastAPI app with health-check endpoint.
- Add ffmpeg and PyMuPDF as dependencies for media handling.

**Files to create or edit:**
- requirements.txt (or pyproject.toml)
- main.py (FastAPI entry point)
- config.py (load API keys and DB settings)
- .env.example

**Acceptance criteria:**
- Server starts on localhost:8000 with `/health` returning 200.
- OpenAI client initializes successfully.
- All environment variables load without errors.

**Test command:** `uvicorn main:app --reload` and verify the health endpoint.

Once complete, proceed to Phase 2.

## Phase 2: Ingestion Layer

Implement video and document upload with raw text extraction.

**Tasks:**
- Create upload endpoint that accepts one video file and multiple PDFs/notes.
- Extract audio from video using ffmpeg.
- Transcribe audio using OpenAI Whisper (`whisper-1` model) with timestamps.
- Extract text from PDFs using PyMuPDF (preserve page numbers).
- Store raw transcript (with timestamps) and document texts in the database with metadata.

**Files to create:**
- app/ingestion.py (core logic)
- app/models.py (Pydantic models for uploads)
- routes/upload.py

**Acceptance criteria:**
- Upload of a sample 10-minute video + one PDF succeeds.
- Clean transcript with timestamps and document text are stored with source metadata.

**Test:** Use curl or Postman to upload sample files and query the database to confirm storage.

Once complete, proceed to Phase 3.

## Phase 3: Processing Layer (Chunking + Concept Extraction)

Add structured knowledge extraction using OpenAI.

**Tasks:**
- Split transcript and documents into manageable chunks.
- Define a single system prompt that forces structured JSON output for concepts, definitions, and key ideas.
- Call OpenAI (`gpt-4o` or `gpt-4o-mini` with `response_format={"type": "json_schema"}`) once per source.
- Store the resulting JSON in the database.

**Files to create:**
- app/processing.py
- prompts/concept_extraction.py (store prompt templates and JSON schema)

**Acceptance criteria:**
- Extraction produces consistent, repeatable JSON for any small transcript or PDF section.
- Concepts include term, definition, and key_ideas arrays.
- Feasibility stage 2 (knowledge extraction) passes.

**Test:** Run processing on a short sample transcript + PDF section and inspect stored JSON.

Once complete, proceed to Phase 4.

## Phase 4: Linking Layer (Cross-Referencing)

Build the connection layer between video and document concepts.

**Tasks:**
- Retrieve extracted concepts from both sources.
- For each pair (or small batch), call OpenAI with a targeted comparison prompt.
- Output JSON: relation type (“reinforces”, “new_info”, “contradiction”, “partial_overlap”), explanation, confidence.
- Store relationships as simple edges in SQLite or a JSON field.

**Files to create:**
- app/linking.py
- prompts/cross_reference.py

**Acceptance criteria:**
- Model reliably detects and labels relationships.
- Feasibility stage 3 (cross-referencing) passes.
- Relationships are queryable by source concept ID.

**Test:** Manually pick one video concept and one document concept, run linking, and verify stored relation.

Once complete, proceed to Phase 5.

## Phase 5: Interaction Layer (Summaries, Q&A, Socratic Quiz)

Implement the user-facing intelligence.

**Tasks:**
- Retrieval: Use Chroma vector store with OpenAI embeddings (`text-embedding-3-small` or `large`) or simple metadata filter.
- Summarize/Explain: OpenAI call with relevant chunks and user goal.
- Q&A: Same retrieval plus OpenAI chat completion.
- Socratic mode: Dedicated prompt that maintains conversation state, asks one guiding question, evaluates answer, and progresses.

**Files to create:**
- app/retrieval.py
- app/interaction.py
- prompts/socratic.py
- routes/interaction.py

**Acceptance criteria:**
- All three interaction modes return coherent responses using linked sources.
- Feasibility stages 4 (retrieval) and 5 (Socratic interaction) pass.
- Conversation state is preserved across messages in quiz mode.

**Test:** Process sample content, then test “summarize”, “explain X”, and “quiz me” endpoints.

Once complete, proceed to Phase 6.

## Phase 6: Frontend, UI, and One-Click Workflow

Add simple user interface.

**Tasks:**
- Build upload/process page with “Process” button that triggers the full pipeline asynchronously.
- Add dashboard with buttons: Summarize, Explain, Quiz me.
- Use Streamlit (fastest for MVP) or basic Next.js.

**Files to create:**
- frontend/ (Streamlit app.py recommended)
- background tasks (FastAPI BackgroundTasks or Celery)

**Acceptance criteria:**
- End-to-end flow from upload to interaction works in one click.
- UI hides all backend complexity.

**Test:** Full user journey with sample lecture + documents.

## Phase 7: Final Validation and Deployment

**Tasks:**
- Run all five feasibility stages end-to-end on a complete lecture + documents.
- Add cost logging and caching (use gpt-4o-mini for extraction steps).
- Deploy to Render.com or Railway (include Dockerfile or build config).
- Add basic auth and per-user storage isolation.

**Final acceptance criteria:**
- The system transforms raw inputs into a connected knowledge graph and active Socratic tutor.
- No redundant features; only the novel linking + teaching behavior is implemented.
- Project runs entirely on OpenAI API plus minimal helpers.

Implement each phase sequentially. After every phase, commit changes and note any deviations. The agent may ask for clarification only on the current phase.