# Socratic AI, start to finish

This walks you from a fresh Mac to your first quiz. No terminal needed unless you want one. If a step does not look like the screenshot, check the troubleshooting section at the end before giving up.

## 0. What you need

- A Mac with Apple Silicon (M1 or newer) running macOS 14 or later.
- About 10 GB of free disk if you want the models on your Mac: the app itself is 700 MB, the recommended model for a 16 GB machine is 5 GB, and the transcription model is 600 MB.
- Or an API key from OpenAI (or OpenRouter, Groq, DeepSeek) if you would rather not run models locally. Then you need almost no disk and any Mac will do.
- For video and YouTube: `ffmpeg`. The setup page tells you the one command to install it. PDFs and text files work without it.

## 1. Install and first launch

Download `SocraticAI-3.0.0-macos-arm64.zip` from the [releases page](https://github.com/arturgrochau/socratic-ai/releases/latest), unzip it, and drag `Socratic AI.app` into Applications.

The first time you open it, macOS will say it cannot check the app for malicious software. That is because the app is signed by me, not by an Apple Developer account. Right-click the app, choose Open, and click Open again in the dialog. You only do this once. If the dialog has no Open button, go to System Settings, Privacy & Security, scroll down, and click Open Anyway.

The app opens to the Welcome page.

![Welcome page](media/01-welcome.png)

Pick where the models should run:

- On this Mac (private). Free, nothing leaves the machine. The page shows how much memory you have and which model it picked for you.
- Cloud API. You paste a key. Your documents are sent to that provider when you generate or chat.

You can change your mind later in Settings.

### If you picked "On this Mac"

The next step is a checklist. Each row is either green or tells you what to do:

![Local checklist](media/02-checklist.png)

1. Ollama is the program that runs the models. If it is not installed, click the download link, install it like any Mac app, and open it once. If it is installed but not running, click Start Ollama.
2. Models. Click Download. The bar shows progress; the 8b model takes a few minutes on a normal connection.
3. ffmpeg. Copy the `brew install ffmpeg` line into Terminal if you plan to use video. Skip it if you only have PDFs.

The page re-checks every few seconds, so rows turn green on their own. Click Finish when Ollama is running and the models are downloaded.

### If you picked "Cloud API"

Paste your key. Leave the URL empty for OpenAI. For OpenRouter, Groq, DeepSeek or LM Studio, paste their OpenAI-compatible URL (Settings has presets that fill this in). Click Finish; the app runs a quick connection test and tells you if the key does not work.

## 2. Your first session

The main screen has two steps.

Step 1 is the video. Paste a YouTube link, or switch to Upload video and drop in a file. Or click Skip video if you only have documents.

![Step 1: add a video](media/03-step1.png)

Step 2 is documents. Add PDFs, `.txt` or `.md` files. Each file uploads the moment you add it. Then click Generate Socratic learning.

![Step 2: add documents](media/04-step2.png)

Generation takes a few minutes on a local model (the first run also loads the model into memory) and under a minute on the Cloud API. The label under the button tells you which stage it is on.

## 3. Reading the pack

When it finishes you land on the dashboard. There is one tab per source, plus a tab for what ties them together, plus the chat.

![Dashboard](media/05-dashboard.png)

A suggestion for how to use it, since the order matters:

1. Read Reflection points first, before the Summary. Try to answer each one in your head. The ones you cannot answer are the reason you are here.
2. Then read Under the surface. These are the assumptions the source makes without saying so. If one surprises you, that is a gap.
3. Then Deep dive and Key concepts. Each concept has a plain explanation and a precise one; click to expand.
4. Read the Summary last, as a check on whether you got the shape of the argument right.

With two or more sources, the Cross-Source tab adds a synthesis, the places the sources agree or contradict each other, and scenarios for applying the ideas.

## 4. The quiz

The quiz is under the Cross-Source tab (or Quiz & Applications for a single source). Click an answer. It locks on your first pick: green if right, red if wrong, with an explanation of the specific misunderstanding behind the wrong option.

![Quiz](media/06-quiz.png)

After a wrong answer there is a button that takes the question into the chat, so you can argue about it.

## 5. The chat

The Socratic Chat tab answers questions from your material only. Answers stream in word by word. If the sources do not cover something, it says so instead of making it up.

![Chat](media/07-chat.png)

Under each answer there are two buttons: Elaborate further, and Quiz me on this. Quiz mode asks you a question, grades your answer, and offers the next one.

## 6. Export and where things live

The buttons at the top of the dashboard export the pack as Markdown or PDF.

Everything the app stores is in `~/Library/Application Support/socratic-ai/`: the database, the search index, uploaded files, and logs. Settings, including your API key, are in `~/Library/Application Support/socratic-ai/settings.json`. Delete the folder to start over.

## 7. Settings

The gear in the top right opens Settings.

![Settings](media/08-settings.png)

- The Mode toggle switches between On this Mac and Cloud API. Each mode remembers its own models.
- In local mode, the dropdowns list the models Ollama has downloaded. A bigger model gives sharper packs and takes longer; `qwen3:30b-a3b` is the best of the defaults if you have 32 GB.
- In API mode, Base URL points the app at OpenRouter, Groq, DeepSeek or LM Studio. Pick a preset, then adjust the model names.
- Test Ollama / Test API checks the connection before you commit to it.
- Run first-time setup again opens the Welcome page.

## 8. Troubleshooting

Ollama is not running. Open the Ollama app (it lives in the menu bar), or run `ollama serve` in Terminal. The checklist row turns green within a few seconds.

The model is too big for my Mac. Open Settings and pick a smaller one from the dropdown: `qwen3:4b` runs on 8 GB. Or switch to the Cloud API.

ffmpeg missing, or videos fail. Install Homebrew from https://brew.sh, then run `brew install ffmpeg`. Restart the app so it sees the new command.

macOS refuses to open the app. Right-click, Open, Open. Or in Terminal: `xattr -dr com.apple.quarantine "/Applications/Socratic AI.app"`.

Generation seems stuck for many minutes with no error. Another app may be holding a very large model in memory (Ollama serves one request at a time when memory is tight). Quit the other app, or run `ollama ps` in Terminal to see what is loaded and `ollama stop <name>` to free it.

Port 8000 is in use. The native app picks another free port automatically. If you run `uv run socratic-ai` from a terminal, set `PORT=8010` first.

Nothing appears when I open a second window. The app is one window. Use the Reset button in the header to start a new session.

## 9. Common questions

Is my material private? In local mode, yes: the model, the search index and the transcription all run on your Mac, and the app has no telemetry. See [PRIVACY.md](../PRIVACY.md). In API mode your text is sent to the provider you chose.

What does it cost? Local mode is free. On the Cloud API with gpt-4o-mini, a pack from two sources costs around one cent.

Intel Mac, Linux or Windows? The `.app` is Apple Silicon only. Everything else runs from a terminal: see the README's install section. Local transcription is Apple Silicon only; elsewhere use the Whisper API or skip video.

Can I use it without Ollama and without a key? No. It needs a model somewhere.

Where do I report a problem? https://github.com/arturgrochau/socratic-ai/issues. Include which mode you were in and, if it is the app, anything from Console.app filtered by "Socratic".
