from __future__ import annotations

import json
from typing import Any, TypeVar, cast

T = TypeVar("T")


def parse_json_object(raw_content: str, *, stage_name: str) -> dict[str, Any]:
    content = str(raw_content or "").strip()
    if not content:
        raise ValueError(f"{stage_name}: model returned empty JSON content")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{stage_name}: malformed JSON at line {exc.lineno} column {exc.colno} (char {exc.pos})"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{stage_name}: expected JSON object payload")

    return payload


def safe_json_loads(raw_value: Any, *, default: T) -> T:
    if raw_value is None:
        return default

    if isinstance(raw_value, (dict, list)):
        parsed = raw_value
    else:
        text_value = str(raw_value).strip()
        if not text_value:
            return default

        try:
            parsed = json.loads(text_value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    if isinstance(default, list) and not isinstance(parsed, list):
        return default
    if isinstance(default, dict) and not isinstance(parsed, dict):
        return default

    return cast(T, parsed)
