"""Two-step source input + generation, then the dashboard.

Step 1 is the (optional) video; step 2 is documents. A document is mandatory
only when the video was skipped. Uploads save instantly and show as removable
rows; the proceed/Generate button enables as soon as a valid source exists.

Upload reliability: each on_upload handler saves the file, then calls
e.sender.reset() to clear the uploader's own chrome, and refreshes ONLY a nested
section (saved list + button) — never the whole screen mid-upload. That keeps
the uploader element alive so a single pick reliably lands in state.
"""
from __future__ import annotations

from typing import Any

from nicegui import ui

from frontend import api_client, format as fmt, state
from frontend.pages.dashboard import render_dashboard


def render_main_body() -> None:
    @ui.refreshable
    def body() -> None:
        st = state.tab()
        if st["pipeline_ready"]:
            render_dashboard()
            return
        if int(st.get("step", 1)) == 1:
            _step_video(st, body)
        else:
            _step_documents(st, body)

    body()


async def _pack_file(file: Any) -> dict[str, Any]:
    # NiceGUI 3.x: the event carries a FileUpload with .name / .content_type and
    # an async read(). (The old e.name / e.content sync API no longer exists.)
    return {
        "name": file.name,
        "mime_type": getattr(file, "content_type", None) or "application/octet-stream",
        "data": await file.read(),
    }


def _source_row(label: str, icon: str, on_remove: Any) -> None:
    with ui.row().classes("items-center gap-2 w-full"):
        ui.icon(icon).classes("text-primary")
        ui.label(label).classes("grow")
        ui.button(icon="close", on_click=on_remove).props("flat round dense color=grey").tooltip("Remove")


def _step_video(st: dict[str, Any], body: Any) -> None:
    ui.label("Step 1 of 2 · Add a video").classes("text-2xl font-bold")
    ui.label("Paste a YouTube link or upload a video. You can skip this step.").classes(
        "text-sm text-gray-500"
    )

    has_video = lambda: bool(st["video_upload"]) or bool(fmt.normalize_youtube_url(st["video_url"]))

    with ui.card().classes("w-full gap-2"):
        def set_mode(value: str) -> None:
            st["video_mode"] = value
            body.refresh()

        ui.toggle(
            ["YouTube link", "Upload video"],
            value=st["video_mode"],
            on_change=lambda e: set_mode(e.value),
        ).props("color=primary")

        # Uploader / input live OUTSIDE the refreshable controls section so an
        # in-flight upload is never destroyed by a refresh.
        if st["video_mode"] == "Upload video":
            async def on_video_upload(e: Any) -> None:
                if not fmt.is_supported_video_name(e.file.name):
                    ui.notify("Unsupported video. Use mp4, mov, m4v, avi, mkv, webm.", type="warning")
                    return
                st["video_upload"] = await _pack_file(e.file)
                st["video_url"] = ""
                e.sender.reset()
                controls.refresh()

            ui.upload(on_upload=on_video_upload, auto_upload=True, max_files=1).props(
                'accept="video/*,.mkv,.webm" flat bordered'
            ).classes("w-full")
        else:
            url_input = ui.input(
                "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            ).classes("w-full")

            def save_link() -> None:
                normalized = fmt.normalize_youtube_url(url_input.value or "")
                if not normalized:
                    ui.notify("Enter a valid single-video YouTube URL.", type="negative")
                    return
                st["video_url"] = normalized
                st["video_upload"] = None
                url_input.value = ""
                controls.refresh()

            url_input.on("keydown.enter", save_link)
            ui.button("Save link", on_click=save_link).props("unelevated color=primary")

        @ui.refreshable
        def controls() -> None:
            if st["video_upload"]:
                _source_row(
                    st["video_upload"]["name"], "movie",
                    lambda: (st.update(video_upload=None), controls.refresh()),
                )
            elif fmt.normalize_youtube_url(st["video_url"]):
                _source_row(
                    st["video_url"], "smart_display",
                    lambda: (st.update(video_url=""), controls.refresh()),
                )

            def go_next() -> None:
                st["step"] = 2
                body.refresh()

            label = "Next →" if has_video() else "Skip video →"
            ui.button(label, on_click=go_next).props("unelevated color=primary").classes("mt-2")

        controls()


def _step_documents(st: dict[str, Any], body: Any) -> None:
    has_video = bool(st["video_upload"]) or bool(fmt.normalize_youtube_url(st["video_url"]))

    ui.label("Step 2 of 2 · Add documents").classes("text-2xl font-bold")
    ui.label(
        "Upload PDFs, text, or markdown."
        + ("" if has_video else " At least one document is required (no video was added).")
    ).classes("text-sm text-gray-500")

    with ui.card().classes("w-full gap-2"):
        async def on_docs_upload(e: Any) -> None:
            # on_multi_upload fires once with all selected files (no per-file race).
            added: list[dict[str, Any]] = []
            for file in e.files:
                ok, invalid = fmt.are_supported_document_names([file.name])
                if not ok:
                    ui.notify(f"Skipped {', '.join(invalid)} (use pdf, txt, md).", type="warning")
                    continue
                added.append(await _pack_file(file))
            if added:
                st["document_uploads"] = st["document_uploads"] + added
            e.sender.reset()
            panel.refresh()

        ui.upload(on_multi_upload=on_docs_upload, auto_upload=True, multiple=True).props(
            'accept=".pdf,.txt,.md" flat bordered'
        ).classes("w-full")

        @ui.refreshable
        def panel() -> None:
            for index, doc in enumerate(st["document_uploads"]):
                def _remove(i: int = index) -> None:
                    docs = list(st["document_uploads"])
                    del docs[i]
                    st["document_uploads"] = docs
                    panel.refresh()

                _source_row(doc["name"], "description", _remove)

            has_docs = bool(st["document_uploads"])
            can_generate = has_docs or has_video

            with ui.row().classes("w-full items-center gap-2 mt-2"):
                ui.button("← Back", on_click=lambda: (st.update(step=1), body.refresh())).props("flat")
                gen = ui.button("Generate Socratic learning", on_click=run_generation).props(
                    "unelevated size=lg color=primary"
                ).classes("grow")
                if not can_generate:
                    gen.disable()

            if not can_generate:
                ui.label("Upload a document above to enable Generate.").classes("text-xs text-gray-500")
            elif has_video and not has_docs:
                ui.label(
                    "Documents are optional — generate with just the video, or add some."
                ).classes("text-xs text-gray-400")
            else:
                ui.label("Estimated time: about 2 to 5 minutes.").classes("text-xs text-gray-400")

        stage_label = ui.label("").classes("text-primary")
        spinner = ui.spinner(size="lg")
        spinner.visible = False

        async def run_generation() -> None:
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

        panel()
