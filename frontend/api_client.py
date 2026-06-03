"""Async HTTP client to the in-process FastAPI backend.

The NiceGUI UI and the API share one uvicorn process, so these are loopback
calls. Going over HTTP (rather than importing app.* directly) keeps a clean
boundary and reuses the routes' validation, error mapping, and auth header.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from nicegui import app


def base_url() -> str:
    host = os.getenv("HOST", "127.0.0.1")
    if host in {"0.0.0.0", ""}:
        host = "127.0.0.1"
    port = os.getenv("PORT", "8000")
    return f"http://{host}:{port}"


def _headers() -> dict[str, str]:
    user_id = "local"
    try:
        user_id = str(app.storage.user.get("user_id") or "local").strip() or "local"
    except Exception:
        user_id = "local"
    return {"X-User-ID": user_id}


def _request_timeout() -> float:
    raw = str(os.getenv("FRONTEND_REQUEST_TIMEOUT", "600")).strip()
    try:
        value = float(raw)
    except ValueError:
        value = 600.0
    return max(120.0, value)


def _raise_for_payload(response: httpx.Response, label: str) -> None:
    if response.status_code < 400:
        return
    detail: Any
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else payload
    except Exception:
        detail = response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(detail, dict):
        message = str(detail.get("message") or "Request failed.").strip()
        bits = [message]
        for key in ("stage", "run_id", "reason"):
            value = str(detail.get(key) or "").strip()
            if value:
                bits.append(f"{key}={value}")
        detail = " | ".join(bits)
    raise RuntimeError(f"{label} failed: {detail}")


async def upload(
    *,
    video: dict[str, Any] | None,
    video_url: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    data: dict[str, str] = {}

    if video is not None:
        files.append(("video", (str(video["name"]), video["data"], str(video["mime_type"]))))
    elif video_url.strip():
        data["video_url"] = video_url.strip()

    for doc in documents:
        files.append(("documents", (str(doc["name"]), doc["data"], str(doc["mime_type"]))))

    async with httpx.AsyncClient(timeout=_request_timeout()) as client:
        response = await client.post(
            f"{base_url()}/upload",
            files=files or None,
            data=data or None,
            headers=_headers(),
        )
    _raise_for_payload(response, "/upload")
    return response.json()


async def generate(*, video_source_id: int | None, document_source_ids: list[int]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_request_timeout()) as client:
        response = await client.post(
            f"{base_url()}/generate-tailored-learning",
            json={"video_source_id": video_source_id, "document_source_ids": document_source_ids},
            headers=_headers(),
        )
    _raise_for_payload(response, "/generate-tailored-learning")
    return response.json()


async def ask(*, session_id: str, source_ids: list[int], query: str, top_k: int = 8) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_request_timeout()) as client:
        response = await client.post(
            f"{base_url()}/ask",
            json={"session_id": session_id, "source_ids": source_ids, "query": query, "top_k": top_k},
            headers=_headers(),
        )
    _raise_for_payload(response, "/ask")
    return response.json()


async def get_settings() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{base_url()}/settings", headers=_headers())
    _raise_for_payload(response, "GET /settings")
    return response.json()


async def put_settings(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(f"{base_url()}/settings", json=payload, headers=_headers())
    _raise_for_payload(response, "PUT /settings")
    return response.json()


async def test_connection(provider: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"{base_url()}/settings/test", params={"provider": provider}, headers=_headers()
        )
    _raise_for_payload(response, "GET /settings/test")
    return response.json()
