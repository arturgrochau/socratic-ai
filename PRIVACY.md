# Privacy

Socratic AI has no telemetry, no analytics, no accounts and no server of its own. Nothing about you or your material is collected by the author.

## Local mode (the default)

Everything runs on your computer: the language model (through Ollama), the embeddings, the transcription (MLX), and the database. Your files, transcripts and chat history stay in your user data folder:

- macOS: `~/Library/Application Support/socratic-ai/`
- Linux: `~/.local/share/socratic-ai/`

The app makes two kinds of network requests in local mode, both to software you installed:

- `http://localhost:11434`, your own Ollama daemon, for model calls and model downloads (downloads come from ollama.com).
- Hugging Face, once, to fetch the transcription model weights the first time you add a video.

If you paste a YouTube link, the video is downloaded from YouTube by `yt-dlp` on your machine.

## Cloud API mode

If you switch to Cloud API mode in Settings, the text of your sources, your questions and the generated material are sent to the provider you configured (OpenAI by default, or whatever base URL you entered). That provider's privacy policy then applies to that data. Your API key is stored with file permissions `600` in `~/Library/Application Support/socratic-ai/settings.json` (or `~/.config/socratic-ai/` on Linux) and is never sent anywhere except to that provider.

## Deleting your data

Delete the data folder above. That removes the database, the vector store, uploaded files and logs. The settings file lives in the config folder listed above.

## Questions

Open an issue at https://github.com/arturgrochau/socratic-ai/issues.
