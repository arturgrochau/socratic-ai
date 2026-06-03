"""Single 'Add sources' screen + generation, then the dashboard underneath.

Replaces the old 3-step wizard. One screen: an optional Video card and an
optional Documents card, each showing a live, removable list of what's added,
and one Generate button that is only enabled once at least one source exists.
"""
from __future__ import annotations

from typing import Any

from nicegui import ui

from frontend import api_client, format as fmt, state
from frontend.pages.dashboard import render_dashboard


def render_main_body() -> None:
    """Render the source-input screen, or the dashboard once generation ran."""

    @ui.refreshable
    def body() -> None:
        st = state.tab()
        if st["pipeline_ready"]:
            render_dashboard()
            return
        _sources_screen(st, body)

    body()


def _pack_upload(event: Any) -> dict[str, Any]:
    return {
        "name": event.name,
        "mime_type": getattr(event, "type", None) or "application/octet-stream",
        "data": event.content.read(),
    }


def _source_row(label: str, icon: str, on_remove: Any) -> None:
    with ui.row().classes("items-center gap-2 w-full"):
        ui.icon(icon).classes("text-primary")
        ui.label(label).classes("grow")
        ui.button(icon="close", on_click=on_remove).props("flat round dense color=grey")


def _sources_screen(st: dict[str, Any], body: Any) -> None:
    ui.label("Add your sources").classes("text-2xl font-bold")
    ui.label(
        "Add a video and/or documents, then generate. At least one source is required."
    ).classes("text-sm text-gray-500")

    # ── Video card (optional) ────────────────────────────────────────────────
    with ui.card().classes("w-full"):
        ui.label("🎬 Video (optional)").classes("text-lg font-semibold text-primary")

        saved_url = fmt.normalize_youtube_url(st["video_url"])
        if st["video_upload"]:
            _source_row(
                st["video_upload"]["name"], "movie",
                lambda: (st.update(video_upload=None), body.refresh()),
            )
        elif saved_url:
            _source_row(
                saved_url, "smart_display",
                lambda: (st.update(video_url=""), body.refresh()),
            )
        else:
            def set_mode(value: str) -> None:
                st["video_mode"] = value
                body.refresh()

            ui.toggle(
                ["YouTube link", "Upload video"],
                value=st["video_mode"],
                on_change=lambda e: set_mode(e.value),
            ).props("color=primary")

            if st["video_mode"] == "Upload video":
                def on_video_upload(e: Any) -> None:
                    if not fmt.is_supported_video_name(e.name):
                        ui.notify("Unsupported video. Use mp4, mov, m4v, avi, mkv, webm.", type="warning")
                        return
                    st["video_upload"] = _pack_upload(e)
                    st["video_url"] = ""
                    body.refresh()

                ui.upload(on_upload=on_video_upload, auto_upload=True, max_files=1).props(
                    'accept="video/*,.mkv,.webm"'
                ).classes("w-full")
            else:
                url_input = ui.input(
                    "Paste one YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
                ).classes("w-full")

                def save_link() -> None:
                    normalized = fmt.normalize_youtube_url(url_input.value or "")
                    if not normalized:
                        ui.notify("Enter a valid single-video YouTube URL.", type="negative")
                        return
                    st["video_url"] = normalized
                    st["video_upload"] = None
                    body.refresh()

                url_input.on("keydown.enter", save_link)
                ui.button("Save link", on_click=save_link).props("unelevated color=primary")

    # ── Documents card (optional) ────────────────────────────────────────────
    with ui.card().classes("w-full"):
        ui.label("📄 Documents (optional)").classes("text-lg font-semibold text-primary")

        def on_doc_upload(e: Any) -> None:
            ok, invalid = fmt.are_supported_document_names([e.name])
            if not ok:
                ui.notify(f"Unsupported: {', '.join(invalid)}. Use pdf, txt, md.", type="warning")
                return
            st["document_uploads"] = st["document_uploads"] + [_pack_upload(e)]
            body.refresh()  # re-render also resets the upload widget

        ui.upload(on_upload=on_doc_upload, auto_upload=True, multiple=True).props(
            'accept=".pdf,.txt,.md"'
        ).classes("w-full")

        for index, doc in enumerate(st["document_uploads"]):
            def _remove(i: int = index) -> None:
                docs = list(st["document_uploads"])
                del docs[i]
                st["document_uploads"] = docs
                body.refresh()

            _source_row(doc["name"], "description", _remove)

    # ── Generate (gated on >=1 source) ───────────────────────────────────────
    has_video = bool(st["video_upload"]) or bool(fmt.normalize_youtube_url(st["video_url"]))
    has_any = has_video or bool(st["document_uploads"])

    stage_label = ui.label("").classes("text-primary")
    spinner = ui.spinner(size="lg")
    spinner.visible = False

    generate_btn = ui.button("Generate Socratic learning").props(
        "unelevated size=lg color=primary"
    ).classes("w-full")
    if not has_any:
        generate_btn.disable()
        ui.label("Add a video or a document to start.").classes("text-xs text-gray-400")
    else:
        ui.label("Estimated time: about 2 to 5 minutes.").classes("text-xs text-gray-400")

    async def run_generation() -> None:
        generate_btn.disable()
        spinner.visible = True
        try:
            stage_label.text = "Stage 1/3: Uploading sources..."
            upload_payload = await api_client.upload(
                video=st["video_upload"], video_url=st["video_url"], documents=st["document_uploads"],
            )
            video_data = upload_payload.get("video") or {}
            video_source_id = int(video_data.get("source_id", 0) or 0)
            normalized_video_id = video_source_id if video_source_id > 0 else None
            document_source_ids = [int(d["source_id"]) for d in upload_payload.get("documents", [])]

            stage_label.text = "Stage 2/3: Generating tailored learning..."
            generation_payload = await api_client.generate(
                video_source_id=normalized_video_id, document_source_ids=document_source_ids,
            )

            stage_label.text = "Stage 3/3: Finalizing dashboard..."
            st["video_source_id"] = normalized_video_id
            st["document_source_ids"] = document_source_ids
            st["source_ids"] = (
                ([normalized_video_id] if normalized_video_id is not None else []) + document_source_ids
            )
            st["generation_result"] = generation_payload
            st["pipeline_ready"] = True
            st["ask_messages"] = []
            st["quiz_selected"] = {}
            ui.notify("Tailored Socratic learning generated.", type="positive")
            body.refresh()
        except Exception as exc:  # noqa: BLE001 — show the failure to the user
            stage_label.text = ""
            ui.notify(str(exc), type="negative", multi_line=True, close_button="OK")
        finally:
            spinner.visible = False
            if not st["pipeline_ready"]:
                generate_btn.enable()

    generate_btn.on_click(run_generation)
