"""Per-tab and per-user UI state, backed by NiceGUI storage.

`app.storage.tab` holds the wizard / chat / generation state for one browser
tab (replaces Streamlit's st.session_state). `app.storage.user` holds the
user id across tabs. Access these only inside a connected @ui.page handler.
"""
from __future__ import annotations

import uuid
from typing import Any

from nicegui import app

# Single-user local app: a fixed id satisfies the backend X-User-ID requirement
# and scopes the local DB. No user-facing account concept.
LOCAL_USER_ID = "local"

# IMPORTANT: every value here must be JSON-serializable. NiceGUI serializes tab
# storage on reconnect, so raw upload BYTES must never live here — files are
# uploaded to the backend immediately and only their {name, source_id} reference
# is kept (see frontend/pages/build.py). Storing bytes here was the cause of the
# "alt-tab resets the page / not connected" bug.
_TAB_DEFAULTS: dict[str, Any] = {
    "step": 1,                            # 1 = video, 2 = documents
    "video_mode": "YouTube link",         # or "Upload video"
    "video_source": None,                 # {"name","source_id","kind":"file"|"youtube"} once uploaded
    "document_sources": [],               # list of {"name","source_id"} once uploaded
    "pipeline_ready": False,
    "generation_result": None,
    "video_source_id": None,
    "document_source_ids": [],
    "source_ids": [],
    "session_id": "",
    "ask_messages": [],                   # list of {"role","content"}
    "pending_chat_prompt": "",            # seed auto-submitted into chat (quiz handoff)
    "quiz_selected": {},                  # question index (str) -> chosen option index
    "quiz_mode_active": False,
    "quiz_awaiting_answer": False,
    "quiz_turn_count": 0,
}


def _fresh(value: Any) -> Any:
    """Copy mutable defaults so tabs never share a list/dict."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def tab() -> dict[str, Any]:
    """Return this tab's state dict, ensuring all defaults exist."""
    store = app.storage.tab
    for key, value in _TAB_DEFAULTS.items():
        if key not in store:
            store[key] = _fresh(value)
    if not store.get("session_id"):
        store["session_id"] = str(uuid.uuid4())
    return store


def user_id() -> str:
    return str(app.storage.user.get("user_id") or LOCAL_USER_ID).strip()


def reset_builder() -> None:
    """Clear the build wizard + dashboard + chat back to a fresh session."""
    store = app.storage.tab
    for key, value in _TAB_DEFAULTS.items():
        store[key] = _fresh(value)
    store["session_id"] = str(uuid.uuid4())
