"""
Structured per-session logging.

Writes one JSONL file per session under SESSION_LOG_DIR with token counts,
cache hit/miss, redundancy scores, and end-to-end timing. Designed to be
grep-friendly and dashboard-friendly without pulling in a heavier observability
stack.

Two usage modes:

    # Explicit (preferred for tests / scripts):
    with SessionLogger("session-abc") as log:
        log.record("processing", {"source_id": 7, "cache_hit": True})

    # Implicit via context-var (so deep call sites don't need plumbing):
    with session_log("run-xyz", user_id="u-1") as log:
        do_pipeline_work()         # any code in here can call get_active_logger().record(...)
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator


SESSION_LOG_DIR = Path(os.getenv("SESSION_LOG_DIR", "./logs/sessions"))
ENABLE_SESSION_LOG = os.getenv("ENABLE_SESSION_LOG", "true").lower() == "true"


# A no-op fallback so call sites can write
#     get_active_logger().record(...)
# unconditionally — even when nothing set up a logger.
class _NullSessionLogger:
    session_id = "none"
    user_id = "none"

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        return None


_active_logger: ContextVar["SessionLogger | _NullSessionLogger"] = ContextVar(
    "active_session_logger", default=_NullSessionLogger()
)


class SessionLogger:
    def __init__(self, session_id: str, user_id: str | None = None) -> None:
        self.session_id = session_id.strip() or "unknown"
        self.user_id = (user_id or "unknown").strip()
        self.started_at = time.time()
        self._path = SESSION_LOG_DIR / f"{self.session_id}.jsonl"
        if ENABLE_SESSION_LOG:
            SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        if not ENABLE_SESSION_LOG:
            return
        entry = {
            "ts": time.time(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "stage": stage,
            **payload,
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")
        except OSError:
            pass

    def __enter__(self) -> "SessionLogger":
        self._token = _active_logger.set(self)
        self.record("session_start", {"started_at": self.started_at})
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.record(
            "session_end",
            {
                "duration_s": round(time.time() - self.started_at, 3),
                "ok": exc_type is None,
                "error": repr(exc) if exc else None,
            },
        )
        _active_logger.reset(self._token)


def get_active_logger() -> "SessionLogger | _NullSessionLogger":
    return _active_logger.get()


@contextmanager
def session_log(session_id: str, user_id: str | None = None) -> Iterator[SessionLogger]:
    logger = SessionLogger(session_id, user_id)
    with logger as scope:
        yield scope
