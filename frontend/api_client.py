"""Async HTTP client to the in-process FastAPI backend.

The NiceGUI UI and the API share one uvicorn process, so these are loopback
calls. Going over HTTP (rather than importing app.* directly) keeps a clean
boundary and reuses the routes' validation, error mapping, and auth header.
"""
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
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


async def _post_upload(
    *,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    data: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_request_timeout()) as client:
        response = await client.post(
            f"{base_url()}/upload",
            files=files or None,
            data=data or None,
            headers=_headers(),
        )
    _raise_for_payload(response, "/upload")
    return response.json()


async def upload_document(*, name: str, mime_type: str, data: bytes) -> dict[str, Any]:
    """Ingest one document immediately; return a JSON-safe {name, source_id} ref.

    Uploading on add (instead of buffering bytes in tab storage) is what makes the
    builder survive an alt-tab reconnect."""
    payload = await _post_upload(files=[("documents", (name, data, mime_type))])
    docs = payload.get("documents") or []
    if not docs or int(docs[0].get("source_id", 0) or 0) <= 0:
        raise RuntimeError("Upload did not return a document source id.")
    return {"name": name, "source_id": int(docs[0]["source_id"])}


async def upload_video_file(*, name: str, mime_type: str, data: bytes) -> dict[str, Any]:
    payload = await _post_upload(files=[("video", (name, data, mime_type))])
    source_id = int((payload.get("video") or {}).get("source_id", 0) or 0)
    if source_id <= 0:
        raise RuntimeError("Upload did not return a video source id.")
    return {"name": name, "source_id": source_id, "kind": "file"}


async def upload_video_url(url: str) -> dict[str, Any]:
    payload = await _post_upload(data={"video_url": url.strip()})
    source_id = int((payload.get("video") or {}).get("source_id", 0) or 0)
    if source_id <= 0:
        raise RuntimeError("Upload did not return a video source id.")
    return {"name": url.strip(), "source_id": source_id, "kind": "youtube"}


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


async def ask_stream(
    *, session_id: str, source_ids: list[int], query: str, top_k: int = 8
) -> AsyncIterator[dict[str, Any]]:
    """Yield the NDJSON events of POST /ask/stream: {"delta"}..., {"done"} or {"error"}."""
    async with httpx.AsyncClient(timeout=_request_timeout()) as client:
        async with client.stream(
            "POST",
            f"{base_url()}/ask/stream",
            json={"session_id": session_id, "source_ids": source_ids, "query": query, "top_k": top_k},
            headers=_headers(),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_payload(response, "/ask/stream")
            async for line in response.aiter_lines():
                line = line.strip()
                if line:
                    yield json.loads(line)


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


async def put_mode(mode: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{base_url()}/settings/mode", json={"mode": mode}, headers=_headers()
        )
    _raise_for_payload(response, "PUT /settings/mode")
    return response.json()


async def get_ollama_models() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base_url()}/settings/ollama-models", headers=_headers())
    _raise_for_payload(response, "GET /settings/ollama-models")
    return response.json()


async def generate_status() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base_url()}/generate-status", headers=_headers())
    _raise_for_payload(response, "GET /generate-status")
    return response.json()


async def get_setup_status() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{base_url()}/setup/status", headers=_headers())
    _raise_for_payload(response, "GET /setup/status")
    return response.json()


async def start_ollama() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{base_url()}/setup/ollama-start", headers=_headers())
    _raise_for_payload(response, "POST /setup/ollama-start")
    return response.json()


async def start_ollama_pull(models: list[str]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url()}/setup/ollama-pull", json={"models": models}, headers=_headers()
        )
    _raise_for_payload(response, "POST /setup/ollama-pull")
    return response.json()


async def get_ollama_pull(job_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{base_url()}/setup/ollama-pull/{job_id}", headers=_headers())
    _raise_for_payload(response, "GET /setup/ollama-pull")
    return response.json()


async def complete_setup(mode: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url()}/setup/complete", json={"mode": mode}, headers=_headers()
        )
    _raise_for_payload(response, "POST /setup/complete")
    return response.json()
