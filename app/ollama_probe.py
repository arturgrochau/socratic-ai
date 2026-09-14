"""Small, dependency-light probes against a local Ollama daemon.

Everything that only needs to *ask the daemon a question* (is it up, what is
pulled, please unload) lives here so the settings routes, the setup wizard and
the shutdown hook share one implementation instead of three ad-hoc requests.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def _base(host: str) -> str:
    return host.rstrip("/")


def list_models(host: str, *, timeout: float = 2.0) -> list[str]:
    """Names of the models the daemon has pulled. Raises on any failure so the
    caller can distinguish "down" from "up with nothing pulled"."""
    response = httpx.get(f"{_base(host)}/api/tags", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]


def is_reachable(host: str, *, timeout: float = 1.5) -> bool:
    try:
        list_models(host, timeout=timeout)
        return True
    except Exception:
        return False


def running_models(host: str, *, timeout: float = 2.0) -> list[str]:
    """Models currently resident in memory (`ollama ps`)."""
    response = httpx.get(f"{_base(host)}/api/ps", timeout=timeout)
    response.raise_for_status()
    return [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]


def unload_models(host: str, models: list[str], *, timeout: float = 2.0) -> None:
    """Ask the daemon to evict the given models now (keep_alive=0).

    Best-effort: used on app quit so a 19 GB model does not stay resident
    for the rest of its keep_alive window. Errors are logged, never raised."""
    for name in dict.fromkeys(m for m in models if m):
        try:
            httpx.post(
                f"{_base(host)}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=timeout,
            )
        except Exception as exc:  # daemon already gone, model not loaded, ...
            logger.debug("unload %s failed: %s", name, exc)
