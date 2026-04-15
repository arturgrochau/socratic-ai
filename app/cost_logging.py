from __future__ import annotations

from typing import Any

from sqlalchemy import text

from config import ENABLE_COST_LOGGING, db_engine


def ensure_cost_logging_tables() -> None:
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS api_call_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    call_stage TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    request_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_api_call_usage_user_stage
                ON api_call_usage (user_id, call_stage, created_at)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS generation_stage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms INTEGER,
                    error_message TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_stage_events_run
                ON generation_stage_events (run_id, stage_name, created_at)
                """
            )
        )


def extract_usage_fields(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is None:
        return None, None, None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
        int(total_tokens) if total_tokens is not None else None,
    )


def log_api_usage(
    *,
    response: Any,
    user_id: str,
    call_stage: str,
    model_name: str,
    request_count: int = 1,
) -> None:
    if not ENABLE_COST_LOGGING:
        return

    prompt_tokens, completion_tokens, total_tokens = extract_usage_fields(response)

    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO api_call_usage (
                    user_id,
                    call_stage,
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    request_count
                ) VALUES (
                    :user_id,
                    :call_stage,
                    :model_name,
                    :prompt_tokens,
                    :completion_tokens,
                    :total_tokens,
                    :request_count
                )
                """
            ),
            {
                "user_id": user_id,
                "call_stage": call_stage,
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "request_count": max(int(request_count), 1),
            },
        )


def log_generation_stage_event(
    *,
    run_id: str,
    user_id: str,
    stage_name: str,
    attempt_number: int,
    status: str,
    duration_ms: int | None = None,
    error_message: str | None = None,
    details: str | None = None,
) -> None:
    if not ENABLE_COST_LOGGING:
        return

    with db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO generation_stage_events (
                    run_id,
                    user_id,
                    stage_name,
                    attempt_number,
                    status,
                    duration_ms,
                    error_message,
                    details
                ) VALUES (
                    :run_id,
                    :user_id,
                    :stage_name,
                    :attempt_number,
                    :status,
                    :duration_ms,
                    :error_message,
                    :details
                )
                """
            ),
            {
                "run_id": str(run_id).strip(),
                "user_id": user_id,
                "stage_name": stage_name,
                "attempt_number": max(int(attempt_number), 1),
                "status": str(status).strip() or "unknown",
                "duration_ms": int(duration_ms) if duration_ms is not None else None,
                "error_message": (str(error_message)[:1200] if error_message else None),
                "details": (str(details)[:2000] if details else None),
            },
        )
