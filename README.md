# Socratic AI

> *Sapere aude* — Dare to know. — Immanuel Kant

Turn your videos, PDFs, and lectures into a structured thinking workout — not a summary to scroll past.
**Private by default:** everything — models, transcription, embeddings, your material — runs on your
machine. One click in Settings switches to the OpenAI API when you want hosted models instead.

---

## Quick start

Socratic AI runs as a single local app (one process, one port). Install once with
[`uv`](https://docs.astral.sh/uv/) — it handles Python and dependencies for you.

```bash
# 1. Get uv (macOS/Linux). On Windows see https://docs.astral.sh/uv/getting-started/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get Ollama and the default local models (skip if you'll use the OpenAI API)
brew install ollama                       # or https://ollama.com/download
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
ollama pull nomic-embed-text

# 3. Clone and run — no API key needed
git clone https://github.com/arturgrochau/socratic_ai.git
cd socratic_ai
uv run socratic-ai                        # add --extra local for on-device video transcription

# on-device transcription (Apple Silicon): install once, then videos work offline
uv sync --extra local
```

The app opens at **http://localhost:8000** and your browser launches automatically.
You also need `ffmpeg` on your `PATH` if you ingest videos (`brew install ffmpeg`,
`apt install ffmpeg`, or `choco install ffmpeg`).

> Prefer hosted models? Put `OPENAI_API_KEY=sk-...` in `.env` (or paste it in
> **⚙ Settings**) and flip the mode toggle to **OpenAI API** — see
> [Switching the brain](#switching-the-brain).

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
   per source    parallel map-reduce        1 call /Q
                  + 1 cross-source
```

A 2-source session is ~12 LLM calls that run in ~4 parallel waves; chat is exactly one call per
question. Local mode costs nothing per run; API mode (`gpt-4o-mini`) is roughly **~$0.01** per run.
Results are cached, so reopening a session is free.

### What's new in v2.3

- **Local by default, one-click API switch** — generation, chat, embeddings, and transcription all
  run on-device out of the box (Ollama + MLX). A single **Local ↔ API** toggle in Settings flips
  every role at once; each mode keeps its own model bundle, so switching never loses choices. No
  API key required to boot, ever.
- **Real structured outputs on Ollama** — the JSON schemas are now compiled into decoding grammars
  server-side (`format: <schema>`), so local models mechanically cannot emit malformed JSON. Context
  windows are sized explicitly per stage (the daemon's 4096 default silently truncated long
  consolidation prompts head-first), models stay resident between stages (`keep_alive`), and
  embeddings batch through `/api/embed` instead of one HTTP call per chunk.
- **On-device transcription** — `parakeet-mlx` (≈1 min for a 1-hour lecture, lower WER than
  whisper-large-v3) with `mlx-whisper` fallback. The OpenAI 24 MB chunking machinery is bypassed
  entirely in local mode.
- **Provider-safe retrieval** — vector collections are keyed by embedding model (switching modes
  can no longer query stale vectors from the other model's geometry), with the task prefixes
  `nomic-embed-text`/`embeddinggemma` are trained on, cosine space, and an embedded-state cache
  that removes a per-chat-turn Chroma sweep.
- **Fixes that matter locally** — chat no longer injects the newest cross-source synthesis from a
  *different* source set; Ollama token telemetry is recorded (was all-NULL); concurrency is
  provider-aware (8-wide hosted, 2-wide local — a single daemon queues, it doesn't parallelize);
  deterministic failures (bad model name/key) fail fast instead of burning full retry budgets;
  SQLite runs in WAL mode so telemetry writers can't steal an LLM retry attempt; generation shows
  live per-stage progress in the UI and the Generate button guards against double-submit.

### What's new in v2.2

- **A single document is now a full learning artifact** — one source produces key takeaways, a
  bigger-picture synthesis, practical "apply it" scenarios, and an interactive quiz (not just a few
  short sections). Per-source sections are deeper too, while staying selective.
- **Several times faster generation** — ledger extraction runs as parallel map-reduce (window calls
  are concurrent and context-free, merged by code dedup — no more quadratic re-sending of the
  accumulated ledger), all sources build concurrently, and the insights + quiz are two focused
  parallel calls. A fresh run collapses from N serial round-trips to ~4.
- **Fewer, smarter mini-model calls** — the section audit and the chat depth classifier are now pure
  code (counting sentences and keyword-routing a tier are not a model's job); chat is exactly one LLM
  call per turn; chunks embed in the background at ingest so the first chat answer retrieves instantly.
- **Reliable uploads** — add a video (step 1) then documents (step 2); each file uploads the moment you
  add it and only a lightweight reference is kept in the browser tab. Adding a document just works, and
  alt-tabbing away no longer resets the page or drops the connection (generation also runs off the UI
  thread).

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

**Per-source analysis (iterative accumulation)** — the source text is split into chunks. The pipeline groups them into windows (3–4 chunks each) and walks through them. Each LLM call sees the entire accumulated analysis so far plus the next window, and is explicitly told not to repeat anything already covered. A final consolidation call structures the accumulated prose into:

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
        └── LLMClient        (Ollama by default; OpenAI API via the Settings toggle)
```

### Documentation

- [`AGENTS.md`](AGENTS.md) — how to run, test, and work in this repo (also linked as `CLAUDE.md`).
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — file-by-file map, data model, and the known-issues backlog.
- [`docs/pipeline.md`](docs/pipeline.md) — the exact call-by-call runtime trace.
- [`docs/adr/`](docs/adr/) — Architecture Decision Records (why each major choice was made).

---

## Switching the brain

The default is **Local mode**: `qwen3:30b-a3b-instruct-2507` for generation and chat,
`nomic-embed-text` for retrieval, and on-device transcription. The **⚙ Settings** page has a
one-click **Local ↔ API** mode toggle; each mode remembers its own model choices, applies
immediately (cached clients are rebuilt), and persists to your user config dir — no restart.
In local mode the model dropdowns list whatever your Ollama daemon has pulled.

The same choices work as `.env` bootstrap defaults (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`,
`WHISPER_PROVIDER=local|openai|none`, `GENERATION_MODEL`, … — see `.env.example`). Providers are
per-role, so hybrids work — e.g. API mode with the high-volume extraction passes routed to a local
daemon ("aux" switch under Advanced).

Why these local defaults: the 30b-a3b MoE decodes as fast as a dense 4B on Apple Silicon
(~65 tok/s on an M-class Pro) while giving 30B-class quality, and one resident model for every role
avoids load thrash. Tips for the daemon (set in its environment): `OLLAMA_NUM_PARALLEL=2`,
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`.

> **If another app pins its own large model** (keep-alive forever) on the same machine, two ~19GB
> models can evict each other mid-run and stall the Ollama scheduler with no error. Unload the other
> model for long generations (`curl localhost:11434/api/generate -d '{"model":"<name>","keep_alive":0}'`)
> or use a smaller generation model.

The 30B-class local models are close to `gpt-4o-mini` on the generation stages; smaller local
models (7B and under) work but produce visibly less crisp output. Use API mode when you want
frontier-hosted quality and don't mind the material leaving your machine.

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
