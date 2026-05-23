"""
Bootstrap launcher for Socratic AI.

First run: creates a virtualenv and installs requirements.txt.
Subsequent runs: starts backend + frontend and opens the browser.
Close this terminal (or press Ctrl+C) to stop.
"""
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
VENV_DIR = APP_DIR / ".venv"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501


def _venv_python() -> str:
    candidate_unix = VENV_DIR / "bin" / "python"
    candidate_win = VENV_DIR / "Scripts" / "python.exe"
    if candidate_unix.exists():
        return str(candidate_unix)
    if candidate_win.exists():
        return str(candidate_win)
    return sys.executable


def _ensure_env() -> None:
    env_file = APP_DIR / ".env"
    if not env_file.exists():
        print("ERROR: .env file not found.")
        print(f"Create {env_file} with your API key:")
        print("  OPENAI_API_KEY=sk-...")
        sys.exit(1)


def _setup_venv() -> None:
    if VENV_DIR.exists():
        return
    print("First run — setting up environment (this may take a minute)...")
    subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        check=True,
    )
    subprocess.run(
        [
            _venv_python(),
            "-m", "pip", "install", "-q", "--upgrade", "pip",
        ],
        check=True,
    )
    subprocess.run(
        [
            _venv_python(),
            "-m", "pip", "install", "-q",
            "-r", str(APP_DIR / "requirements.txt"),
        ],
        check=True,
    )
    print("Environment ready.")


def main() -> None:
    _ensure_env()
    _setup_venv()

    py = _venv_python()

    backend = subprocess.Popen(
        [
            py, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", str(BACKEND_PORT),
        ],
        cwd=APP_DIR,
    )
    frontend = subprocess.Popen(
        [
            py, "-m", "streamlit", "run",
            str(APP_DIR / "frontend" / "app.py"),
            "--server.port", str(FRONTEND_PORT),
            "--server.headless", "true",
        ],
        cwd=APP_DIR,
    )

    time.sleep(3)
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    print(f"\nSocratic AI is running → http://localhost:{FRONTEND_PORT}")
    print("Press Ctrl+C to stop.\n")

    def _shutdown(_sig: int, _frame: object) -> None:
        print("\nShutting down...")
        frontend.terminate()
        backend.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)  # type: ignore[arg-type]
    signal.signal(signal.SIGTERM, _shutdown)  # type: ignore[arg-type]

    backend.wait()


if __name__ == "__main__":
    main()
