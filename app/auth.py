from __future__ import annotations

from fastapi import Request

from config import USER_ID_HEADER


def get_user_id(request: Request) -> str:
    user_id = (request.headers.get(USER_ID_HEADER) or "").strip()
    if not user_id:
        raise ValueError(f"Missing required user header: {USER_ID_HEADER}")
    return user_id
