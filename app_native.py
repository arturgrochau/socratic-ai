"""Native macOS/desktop window for Socratic AI.

Same FastAPI + NiceGUI app as `run.py`, shown in a pywebview window instead of
a browser tab. `ui.run_with()` has no `native=` switch, so this does by hand
the three things `ui.run(native=True)` does: spawn the window process, register
the window proxy, and shut the server down when the window closes.

    uv run socratic-ai-app          # from a checkout
    Socratic AI.app                 # the PyInstaller bundle runs this module

Keep the module top free of heavy imports: pywebview runs in a spawned child
process that re-imports the entry module, and PyInstaller's bootstrap relies on
`multiprocessing.freeze_support()` being the first thing that runs.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

WINDOW_TITLE = "Socratic AI"
WINDOW_SIZE = (1200, 820)
MIN_WINDOW_SIZE = (900, 640)

# Finder launches apps with PATH=/usr/bin:/bin:/usr/sbin:/sbin; ffmpeg and
# ollama live in Homebrew's prefix. Also honour an explicit ffmpeg dir.
_EXTRA_PATH_DIRS = (
    os.getenv("SOCRATIC_FFMPEG_DIR"),
    "/opt/homebrew/bin",
    "/usr/local/bin",
    str(Path.home() / ".local" / "bin"),
)


def fix_path(env: dict[str, str] | None = None) -> str:
    """Prepend the usual tool locations to PATH once; idempotent."""
    env = os.environ if env is None else env
    current = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    extra = [d for d in _EXTRA_PATH_DIRS if d and d not in current and Path(d).is_dir()]
    env["PATH"] = os.pathsep.join(extra + current)
    return env["PATH"]


def _prepare_environment(port: int) -> None:
    os.environ["HOST"] = "127.0.0.1"
    os.environ["PORT"] = str(port)
    os.environ["SOCRATIC_OPEN_BROWSER"] = "false"
    os.environ["SOCRATIC_NATIVE"] = "1"
    fix_path()
    import config

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # numba (via librosa/parakeet) writes a JIT cache next to the code by
    # default, which is read-only inside a bundle.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(config.DATA_DIR / "numba_cache"))


def _hard_exit_after(seconds: float) -> None:
    """Closing the window mid-generation leaves worker threads alive; uvicorn's
    graceful shutdown waits on them. Give it a grace period, then leave."""
    def _exit() -> None:
        time.sleep(seconds)
        os._exit(0)

    threading.Thread(target=_exit, daemon=True).start()


def run_selftest(port: int) -> int:
    """Headless smoke test used by CI on the frozen bundle: import everything,
    serve on a free port, hit /health, exit 0."""
    import httpx
    import uvicorn

    from main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    status = None
    while time.monotonic() < deadline:
        try:
            status = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code
            break
        except Exception:
            time.sleep(0.2)
    server.should_exit = True
    thread.join(timeout=10)
    print(f"selftest: /health -> {status}")
    return 0 if status == 200 else 1


def main() -> None:
    multiprocessing.freeze_support()
    from nicegui import core
    from nicegui.native import native_mode
    from nicegui.native.native import WindowProxy

    port = native_mode.find_open_port(8000, 8999)
    _prepare_environment(port)

    if "--selftest" in sys.argv:
        sys.exit(run_selftest(port))

    core.app.native.window_args = {
        "min_size": MIN_WINDOW_SIZE,
        "text_select": True,
        "zoomable": True,
    }
    core.app.native.settings = {
        "OPEN_EXTERNAL_LINKS_IN_BROWSER": True,
        "ALLOW_DOWNLOADS": True,
    }
    width, height = WINDOW_SIZE
    native_mode.activate("http", "127.0.0.1", port, WINDOW_TITLE, width, height, False, False)
    # What Server.run() does in ui.run(native=True): lets app.native.main_window
    # proxy calls into the window process.
    core.app.native.main_window = WindowProxy()

    import uvicorn

    from main import app

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", ws="wsproto")
    except KeyboardInterrupt:
        # native_mode's check_shutdown thread interrupts the main thread once
        # the window is gone and the server has stopped.
        pass
    finally:
        _hard_exit_after(5.0)


if __name__ == "__main__":
    main()
