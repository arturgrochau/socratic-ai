"""Pytest config — register the `e2e` marker."""
from __future__ import annotations

import os
import tempfile

import pytest

# Keep every test run out of the real per-user data dir (SQLite, Chroma,
# uploads, logs). Individual tests still override DATABASE_URL etc. as needed.
os.environ.setdefault("SOCRATIC_DATA_DIR", tempfile.mkdtemp(prefix="socratic-test-data-"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end pipeline tests (slow; require cassettes or OPENAI_API_KEY).",
    )
