# Socratic AI

> *Sapere aude* — Dare to know. — Immanuel Kant

Turn videos, PDFs, and lectures into a structured thinking workout — not a summary to scroll past.

---

## Why This Exists

We are living through a strange inversion. For the first time in history, the bottleneck to understanding is not access to information — it is the willingness to do the cognitive work of understanding it. AI makes this worse, not better, if you use it wrong. It is trivially easy to paste a lecture into a chatbot, read the three-paragraph summary it spits back, and feel like you learned something. You didn't. You consumed output. That is not the same thing.

The Enlightenment philosophers had a name for the courage to think for yourself: *sapere aude*. Dare to know. Not dare to read a summary. Not dare to let someone else chew your food and hand it back to you. Dare to actually engage with ideas, wrestle with them, find where they break, and build your own understanding from the ground up.

Socratic AI is built on that premise. It does not answer your questions — it generates better questions. It does not explain what the source says — it forces you to figure out whether you can explain it yourself. The reflection questions are not comprehension checks. They are designed to find the exact point where your understanding runs out, because that is where learning begins.

The name is not an accident. Socrates did not lecture. He asked questions until his interlocutors discovered that they did not know what they thought they knew. That is still the most effective pedagogy ever devised, and it scales to any source material you feed it.

---

## What It Does

Upload a video (YouTube link or `.mp4`) and/or documents (PDFs). Socratic AI runs a structured pipeline:

- **Ingestion** — transcribes video via Whisper, extracts text from PDFs, chunks and embeds everything into a local vector store
- **Processing** — extracts key concepts and a source summary in a single LLM call per source
- **Generation** — produces a full learning pack per source: deep dive, first-principles synthesis, under-the-surface assumptions, and Socratic reflection questions — all in one call, with explicit role contracts so sections never repeat each other
- **Cross-source linking** — finds conceptual bridges and contradictions between sources (batched, efficient)
- **Synthesis** — if you have multiple sources, generates integrated prose that contrasts, connects, and builds across them
- **Socratic chat** — ask follow-up questions grounded in your actual material, not the model's training data

The pipeline runs ~16 LLM calls for a 3-source session (down from 105 in earlier versions). It costs roughly $0.05–0.15 per full session on `gpt-4o-mini`.

---

## Download

| Platform | Link |
|---|---|
| **Mac** (Apple Silicon + Intel) | [→ GitHub Releases](https://github.com/arturgrochau/socratic_ai/releases/latest) |
| **Windows** | [→ GitHub Releases](https://github.com/arturgrochau/socratic_ai/releases/latest) |
| **Self-hosted** | See below |

Download the binary for your platform. On first launch it will:
1. Check for a `.env` file with your `OPENAI_API_KEY`
2. Create a local Python environment and install dependencies (once, ~60 seconds)
3. Start the app and open your browser automatically

---

## Quick Start (Self-Hosted)

**Requirements**: Python 3.11+, `ffmpeg` on PATH

```bash
git clone https://github.com/arturgrochau/socratic_ai.git
cd socratic_ai

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Launch (creates venv and installs deps on first run)
python launcher.py
```

The app opens at `http://localhost:8501`.

---

## How the Pipeline Works

**Per-source analysis** (one combined LLM call per source):

| Section | What it covers |
|---|---|
| Summary | Core arguments and narrative arc |
| Deep Dive | Mechanisms, tradeoffs, failure modes — how it actually works |
| First Principles | The foundational logic in 2-3 sentences, from scratch |
| Under the Surface | Hidden assumptions, practitioner warnings, what most explanations skip |
| Reflection Points | Socratic questions targeting the exact points where understanding typically breaks |

**Cross-source synthesis** (when you have more than one source):

Concept pairs are compared in batches to find reinforcements, contradictions, and gaps. A final synthesis call produces integrated prose — not a side-by-side comparison table, but actual connected reasoning across your sources.

**Architecture**: FastAPI backend + Streamlit frontend + SQLite (session storage) + ChromaDB (vector embeddings) + OpenAI (`gpt-4o-mini` for generation, Whisper for transcription).

---

## Versioning

Releases follow [semantic versioning](https://semver.org/). The `GENERATION_SCHEMA_VERSION` constant in `app/generation.py` controls cache invalidation — bumping it forces regeneration of all cached learning sections on the next run.

---

## License

MIT © Artur Grochau
