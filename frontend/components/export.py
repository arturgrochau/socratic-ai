"""Download buttons for the generated study snapshot (.md and .json)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from nicegui import ui

from frontend import format as fmt


def render_export_buttons(generation_result: dict[str, Any], *, user_id: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    user_token = fmt.slugify_token(user_id, fallback="user")
    prefix = f"socratic_snapshot_{timestamp}_{user_token}"

    with ui.card().classes("w-full"):
        ui.label("Save this generation locally").classes("text-base font-semibold")
        ui.label("Download a readable study snapshot or the full raw payload.").classes(
            "text-sm text-gray-500"
        )
        with ui.row():
            def _download_md() -> None:
                content = fmt.build_generation_export_markdown(generation_result, user_id=user_id)
                ui.download(content.encode("utf-8"), f"{prefix}.md", media_type="text/markdown")

            def _download_json() -> None:
                content = fmt.build_generation_export_json(generation_result, user_id=user_id)
                ui.download(content.encode("utf-8"), f"{prefix}.json", media_type="application/json")

            ui.button("Download study snapshot (.md)", on_click=_download_md).props("outline")
            ui.button("Download raw payload (.json)", on_click=_download_json).props("outline")
        ui.label("Tip: open the markdown file and print it to PDF for a PDF copy.").classes(
            "text-xs text-gray-400"
        )
