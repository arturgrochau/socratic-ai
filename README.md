# Socratic AI

> *Sapere aude* — Dare to know. — Immanuel Kant

Turn your videos, PDFs, and lectures into a structured thinking workout — not a summary to scroll past.

---

## Quick start

Socratic AI runs as a single local app (one process, one port). Install once with
[`uv`](https://docs.astral.sh/uv/) — it handles Python and dependencies for you.

```bash
# 1. Get uv (macOS/Linux). On Windows see https://docs.astral.sh/uv/getting-started/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and configure
git clone https://github.com/arturgrochau/socratic_ai.git
cd socratic_ai
echo "OPENAI_API_KEY=sk-..." > .env      # or run fully local with Ollama — see below

# 3. Run
uv run socratic-ai
```

The app opens at **http://localhost:8000** and your browser launches automatically.
You also need `ffmpeg` on your `PATH` if you ingest videos (`brew install ffmpeg`,
`apt install ffmpeg`, or `choco install ffmpeg`).

> No API key? Open the **⚙ Settings** page in the app and point it at a local
> [Ollama](https://ollama.com) model instead — see [Switching the brain](#switching-the-brain).

---

## What it does, in one minute

Drop in a video (YouTube link or `.mp4`) and any documents (PDFs). Socratic AI runs them through a three-stage pipeline and gives you back a learning pack: a deep dive of each source, the hidden assumptions most explanations skip, the connections *between* your sources, and a Socratic chat grounded in your actual material.

It is built to do the opposite of what most AI study tools do. It does not summarize your sources so you can stop reading them. It uses them to generate the questions that find the exact point where your understanding runs out — because that is where learning starts.

```
 ┌──────────┐    ┌───────────────────┐    ┌────────┐
 │ Ingest   │───▶│ Generate          │───▶│ Chat   │
 │ Whisper  │    │ accumulate +      │    │ on top │
 │ + PDFs   │    │ consolidate + sx  │    │ of all │
 └──────────┘    └───────────────────┘    └────────┘
   per source     ~5 calls per source       1 call /Q
                  + 1 cross-source
```

A 2-source session is ~11 LLM calls (all on `gpt-4o-mini`) and costs roughly **~$0.01** per run. Results are cached, so reopening a session is free.

### What's new in v2.1

- **One process, one port** — the UI is now built with [NiceGUI](https://nicegui.io) and mounted directly on the FastAPI backend. No more separate Streamlit server; everything serves from `http://localhost:8000`.
- **In-app Settings panel** — switch between the OpenAI API and a local Ollama model, change models, and set your key from a ⚙ page in the app. Applies at runtime, no restart, no `.env` editing. Saved to your user config dir. Point generation (or the cheap auxiliary passes) at a small/local model for faster, offline runs.
- **One-screen source input** — add a video and/or documents on a single screen with live, removable lists; Generate unlocks as soon as you have one source.
- **Interactive, hardball quiz** — click to answer (locks on first pick): right turns green, wrong turns red with a correction tied to *that* misconception, then a button hands you into the Socratic chat seeded with what you missed.
- **Leaner generation** — section audits run as one pass per arc (instead of one call per section) and ledger windows are larger, cutting LLM calls per run with no change to the accumulation design.
- **uv-based install** — `uv run socratic-ai` replaces the old PyInstaller binaries and the venv-bootstrapping launcher.

### What's new in v2.0

- **Iterative accumulation pipeline** — each LLM call sees the full accumulated analysis so far and is told not to repeat. Replaces the old novelty-ledger / claim-extraction / progressive-chunking machinery with one simple loop.
- **Layman + technical key term definitions** — every key concept now ships with both a plain-English explanation and a precise technical definition, collapsible in the UI.
- **Cleaner chat** — answers are grounded and direct. No more dangling "Socratic next question" tacked onto every reply.
- **~10× cheaper, ~4× less code** — generation pipeline dropped from ~4,200 lines to ~880 lines. No more `gpt-4o` critic fallback.
- **Pluggable LLM backend** — swap between OpenAI and Ollama per-role (chat, embedding, transcription). Mix providers freely.

---

## Run from source

See [Quick start](#quick-start) above — `uv run socratic-ai` is the supported path
(it resolves Python 3.11+ and all dependencies from `pyproject.toml`/`uv.lock`).

Prefer a manual virtualenv? It still works:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install .
python run.py             # serves the app at http://localhost:8000
```

For development with auto-reload: `uvicorn main:app --reload`.

---

## How the pipeline works

<details>
<summary><strong>Why this exists</strong> (click to expand the manifesto)</summary>

We are living through a strange inversion. For the first time in history, the bottleneck to understanding is not access to information — it is the willingness to do the cognitive work of understanding it. AI makes this worse, not better, if you use it wrong. It is trivially easy to paste a lecture into a chatbot, read the three-paragraph summary it spits back, and feel like you learned something. You didn't. You consumed output. That is not the same thing.

The Enlightenment philosophers had a name for the courage to think for yourself: *sapere aude*. Dare to know. Not dare to read a summary. Not dare to let someone else chew your food and hand it back to you. Dare to actually engage with ideas, wrestle with them, find where they break, and build your own understanding from the ground up.

Socratic AI is built on that premise. It does not answer your questions — it generates better questions. It does not explain what the source says — it forces you to figure out whether you can explain it yourself. The reflection questions are not comprehension checks; they are designed to find the exact point where your understanding runs out.

</details>

**Per-source analysis (iterative accumulation)** — the source text is split into chunks. The pipeline groups them into windows (2–3 chunks each) and walks through them. Each LLM call sees the entire accumulated analysis so far plus the next window, and is explicitly told not to repeat anything already covered. A final consolidation call structures the accumulated prose into:

| Section | What it covers |
|---|---|
| Summary | Core arguments and narrative arc |
| Deep Dive | Mechanisms, constraints, failure modes, edge cases |
| Key Concepts | Each term with a layman-friendly explanation **and** a precise technical definition (collapsible in the UI) |
| Under the Surface | Hidden assumptions and the things most explanations skip |
| Reflection Points | Socratic questions tuned to find where your understanding runs out |

**Cross-source synthesis** — when you have more than one source, a single LLM call weaves them together and produces:
- **Synthesis** — connected reasoning across the sources, not a side-by-side comparison
- **Intersections** — concrete places where the sources reinforce, contradict, or complete each other
- **Application scenarios** — transfer steps and the pitfall to watch for
- **Assessment quiz** — multiple-choice questions with grounded explanations

**Socratic chat** — questions are answered against retrieved chunks from a local vector store (ChromaDB) over your sources. The chat grounds every claim in your material and returns a direct answer (no follow-up question dangling at the end). Quick actions let you elaborate on any answer or enter **quiz mode** for graded Q&A.

### Architecture

```
FastAPI + NiceGUI  (one uvicorn process, port 8000)
        │
        ├── NiceGUI UI       (mounted on the same app via ui.run_with)
        ├── REST API         (/upload, /generate-tailored-learning, /ask, /settings)
        ├── SQLite           (sessions, sources, generated sections — cached)
        ├── ChromaDB         (vector embeddings for retrieval)
        └── LLMClient        (OpenAI by default; Ollama optional, switchable in Settings)
```

---

## Switching the brain

The default backend is OpenAI's `gpt-4o-mini`. The easiest way to switch is the
**⚙ Settings** page in the app: pick `ollama` as the provider, set the model names
and Ollama host, hit **Test Ollama**, then **Apply**. Changes take effect immediately
(the cached LLM clients are rebuilt) and persist to your user config dir — no restart.

```bash
# install ollama and pull a model first
ollama pull llama3.2
```

You can also set the same choices via environment variables in `.env` as a bootstrap
default (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `WHISPER_PROVIDER`, `OLLAMA_HOST`,
`GENERATION_MODEL`, `CHAT_MODEL`, `RETRIEVAL_MODEL` — see `.env.example`). Provider
selection is per-role, so you can mix — e.g. OpenAI for transcription and embeddings,
Llama locally for the rest. Whisper transcription still requires OpenAI. See
[`app/llm_client.py`](app/llm_client.py) for the available roles and how to add a provider.

Quality on a 7B local model will be visibly worse than `gpt-4o-mini` for the generation stages. It works; it just produces less crisp output. Use it for offline runs or privacy-sensitive material.

---

## What this is not

- It is **not a summarizer.** If you wanted a TL;DR, ChatGPT does that in one prompt. This tool is for the times you actually want to learn something.
- It is **not a tutor.** It does not have a curriculum. It works against the source material you give it.
- It does **not replace doing the work.** The reflection questions are useless if you skim them. The whole point is the friction.

---

## Versioning

Releases follow [semantic versioning](https://semver.org/). Cache invalidation is controlled by the `GENERATION_SCHEMA_VERSION` constant in `app/generation.py` — bump it to force regeneration of any cached stage on the next run.

---

## License

MIT © Artur Grochau
