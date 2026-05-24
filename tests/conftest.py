"""Pytest config — register the `e2e` marker."""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end pipeline tests (slow; require cassettes or OPENAI_API_KEY).",
    )
