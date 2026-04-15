# Socratic AI Study Assistant

Socratic AI turns one video lecture and supporting documents into a guided learning experience.

It helps you:
- extract and link concepts across sources,
- read a cross-source synthesis with source-grounded evidence,
- test understanding with harder multi-source quiz questions,
- ask follow-up questions in a Socratic chat flow.

## What the app does

1. Uploads one video and one or more documents.
2. Transcribes and parses source content.
3. Extracts key concepts and cross-source relationships.
4. Generates:
   - source summaries,
   - cross-source intersections,
   - comparative deepening,
   - application scenarios,
   - advanced assessment questions.
5. Serves everything in a Streamlit dashboard backed by FastAPI.

## Quick start

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

### 3) Run backend API

```bash
uvicorn main:app --reload
```

Backend will run at `http://127.0.0.1:8000`.

### 4) Run frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

## Typical user flow

1. Open the Streamlit app.
2. Set API URL and user ID in the sidebar.
3. Upload video and documents.
4. Click Generate tailored Socratic learning.
5. Use the Learning Dashboard sections:
   - Video
   - Documents
   - Cross-Source Synthesis & Assessment
   - Socratic Chatbox

## Validation script

You can run an end-to-end validation with test assets:

```bash
python scripts/validate_pipeline.py
```

## Notes

- Data is user-scoped using the `X-User-ID` header.
- Caching and schema versioning are built into generation workflows.
- Local persistence uses SQLite by default.
