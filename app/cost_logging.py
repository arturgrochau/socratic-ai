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
