"""Single-process launcher for Socratic AI.

The FastAPI API and the NiceGUI frontend are mounted on the same app (see
main.py via ui.run_with), so this just starts one uvicorn server and opens the
browser. No venv bootstrap, no second process, no port 8501.

Run it with uv:

    uv run socratic-ai

or directly:

    python run.py
"""
from __future__ import annotations

import os
import threading
import webbrowser


def _open_browser(url: str) -> None:
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    if os.getenv("SOCRATIC_OPEN_BROWSER", "true").strip().lower() == "true":
        # Browser-facing host (0.0.0.0 isn't navigable).
        browser_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
        _open_browser(f"http://{browser_host}:{port}")

    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
