# Socratic AI

> *Sapere aude* — Dare to know. — Immanuel Kant

Turn your videos, PDFs, and lectures into a structured thinking workout — not a summary to scroll past.

---

## 📥 Download

**Get the latest release for your machine. No build required.**

| Platform | File | Notes |
|---|---|---|
| 🍎 **Mac — Apple Silicon (M1/M2/M3/M4)** &nbsp;⭐ *recommended for most Macs since 2020* | **[`SocraticAI-mac-arm64.zip`](https://github.com/arturgrochau/socratic_ai/releases/latest)** | Native arm64 build |
| 🍎 **Mac — Intel** | [`SocraticAI-mac-intel.zip`](https://github.com/arturgrochau/socratic_ai/releases/latest) | For pre-2020 Macs |
| 🪟 **Windows** | [`SocraticAI-windows.exe`](https://github.com/arturgrochau/socratic_ai/releases/latest) | Single-file binary |
| 🐧 **Linux / self-hosted** | See [Run from source](#run-from-source) | |

> **Which Mac do I have?** Click the Apple menu → *About This Mac*. If the **Chip** row says anything starting with "Apple", get the **arm64** build. If it says "Intel", get the **intel** build.

### First launch on Mac (read this once)

The app is signed with an ad-hoc signature, not an Apple Developer ID, so Gatekeeper will block the first open with **"SocraticAI cannot be opened because the developer cannot be verified"**. This is normal for indie tools. Choose either path:

**Option A — one Terminal command (easiest):**
```bash
xattr -d com.apple.quarantine ~/Downloads/SocraticAI
```
Then double-click it. Done.

**Option B — clickable path:**
1. Try to open the app once and dismiss the warning.
2. Open *System Settings* → *Privacy & Security*.
3. Scroll down. You will see a line about SocraticAI being blocked. Click **Open Anyway**.

You only have to do either of these once per download.

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

### What's new in v2.0

- **Iterative accumulation pipeline** — each LLM call sees the full accumulated analysis so far and is told not to repeat. Replaces the old novelty-ledger / claim-extraction / progressive-chunking machinery with one simple loop.
- **Layman + technical key term definitions** — every key concept now ships with both a plain-English explanation and a precise technical definition, collapsible in the UI.
- **Cleaner chat** — answers are grounded and direct. No more dangling "Socratic next question" tacked onto every reply.
- **~10× cheaper, ~4× less code** — generation pipeline dropped from ~4,200 lines to ~880 lines. No more `gpt-4o` critic fallback.
- **Dual-arch Mac builds** — separate native binaries for Apple Silicon and Intel Macs.
- **Pluggable LLM backend** — swap between OpenAI and Ollama per-role (chat, embedding, transcription). Mix providers freely.

---

## Run from source

Requirements: Python 3.11+, `ffmpeg` on `PATH`, an OpenAI API key (or a local [Ollama](https://ollama.com) — see [Switching the brain](#switching-the-brain)).

```bash
git clone https://github.com/arturgrochau/socratic_ai.git
cd socratic_ai

echo "OPENAI_API_KEY=sk-..." > .env

python launcher.py        # creates a venv + installs deps on first run
```

The app opens at `http://localhost:8501`.

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
FastAPI (backend, port 8000)  ◀──▶  Streamlit (frontend, port 8501)
        │
        ├── SQLite (sessions, sources, generated sections — cached)
        ├── ChromaDB (vector embeddings for retrieval)
        └── LLMClient (OpenAI by default; Ollama optional)
```

---

## Switching the brain

The default backend is OpenAI's `gpt-4o-mini`. To run against a local model with [Ollama](https://ollama.com):

```bash
# 1. install ollama, then pull a model
ollama pull llama3.2

# 2. set provider envs in your .env (whisper still needs OpenAI for now)
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
WHISPER_PROVIDER=openai     # only needed if you upload videos

# 3. launch
python launcher.py
```

Provider selection is per-role (chat, embedding, transcription), so you can mix — for example, OpenAI for transcription and embeddings, Llama locally for the rest. See [`app/llm_client.py`](app/llm_client.py) for the available roles and how to add another provider.

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
