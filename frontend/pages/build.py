"""Two-step source input (video, then documents) + generation, then dashboard.

Step 1 is the optional video; step 2 is documents. A document is mandatory only
when the video was skipped. The proceed/Generate button enables as soon as a
valid source exists.

Two things make uploads reliable here, learned the hard way:
  1. Each upload handler resets the uploader's own chrome and refreshes ONLY a
     nested section (the saved list + button) — never the whole screen mid-upload.
     Refreshing the whole body destroys the live uploader element and uploads are
     lost. The uploader element therefore lives OUTSIDE the nested refreshable.
  2. Files are uploaded to the backend the moment they're added, and only a
     JSON-safe {name, source_id} reference is kept in tab state — never raw bytes
     (bytes can't be serialized on a reconnect, which used to reset the page).
"""
from __future__ import annotations

from typing import Any

from nicegui import ui

from frontend import api_client, state
from frontend import format as fmt
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


async def _read_bytes(file: Any) -> tuple[str, str, bytes]:
    # NiceGUI 3.x upload event carries a FileUpload with .name / .content_type and
    # an async read(). Bytes are used immediately for the upload, never stored.
    return (
        file.name,
        getattr(file, "content_type", None) or "application/octet-stream",
        await file.read(),
    )


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

    with ui.card().classes("w-full gap-2"):
        def set_mode(value: str) -> None:
            st["video_mode"] = value
            body.refresh()

        ui.toggle(
            ["YouTube link", "Upload video"],
            value=st["video_mode"],
            on_change=lambda e: set_mode(e.value),
        ).props("color=primary")

        # Uploader / input live OUTSIDE the nested `controls` refreshable so an
        # in-flight upload is never destroyed by a refresh.
        if st["video_mode"] == "Upload video":
            async def on_video_upload(e: Any) -> None:
                name, mime_type, data = await _read_bytes(e.file)
                if not fmt.is_supported_video_name(name):
                    ui.notify("Unsupported video. Use mp4, mov, m4v, avi, mkv, webm.", type="warning")
                    return
                e.sender.reset()
                ui.notify(f"Uploading {name}…")
                try:
                    st["video_source"] = await api_client.upload_video_file(
                        name=name, mime_type=mime_type, data=data,
                    )
                    controls.refresh()
                except Exception as exc:  # noqa: BLE001
                    ui.notify(str(exc), type="negative", multi_line=True, close_button="OK")

            ui.upload(
                on_upload=on_video_upload, auto_upload=True, max_files=1,
                label="Choose a video file",
            ).props('accept="video/*,.mkv,.webm" flat bordered').classes("w-full")
        else:
            url_input = ui.input(
                "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            ).classes("w-full")

            async def save_link() -> None:
                normalized = fmt.normalize_youtube_url(url_input.value or "")
                if not normalized:
                    ui.notify("Enter a valid single-video YouTube URL.", type="negative")
                    return
                ui.notify("Fetching video…")
                try:
                    st["video_source"] = await api_client.upload_video_url(normalized)
                    url_input.value = ""
                    controls.refresh()
                except Exception as exc:  # noqa: BLE001
                    ui.notify(str(exc), type="negative", multi_line=True, close_button="OK")

            url_input.on("keydown.enter", save_link)
            ui.button("Save link", on_click=save_link).props("unelevated color=primary")

        @ui.refreshable
        def controls() -> None:
            if st["video_source"]:
                kind = str(st["video_source"].get("kind") or "")
                icon = "smart_display" if kind == "youtube" else "movie"
                _source_row(
                    st["video_source"]["name"], icon,
                    lambda: (st.update(video_source=None), controls.refresh()),
                )

            def go_next() -> None:
                st["step"] = 2
                body.refresh()

            label = "Next →" if st["video_source"] else "Skip video →"
            ui.button(label, on_click=go_next).props("unelevated color=primary").classes("mt-2")

        controls()


def _step_documents(st: dict[str, Any], body: Any) -> None:
    has_video = bool(st["video_source"])

    ui.label("Step 2 of 2 · Add documents").classes("text-2xl font-bold")
    ui.label(
        "Upload PDFs, text, or markdown."
        + ("" if has_video else " At least one document is required (no video was added).")
    ).classes("text-sm text-gray-500")

    with ui.card().classes("w-full gap-2"):
        async def on_docs_upload(e: Any) -> None:
            # on_multi_upload fires once with all selected files (no per-file race).
            sender = e.sender
            added: list[dict[str, Any]] = []
            for file in e.files:
                name, mime_type, data = await _read_bytes(file)
                ok, invalid = fmt.are_supported_document_names([name])
                if not ok:
                    ui.notify(f"Skipped {', '.join(invalid)} (use pdf, txt, md).", type="warning")
                    continue
                ui.notify(f"Uploading {name}…")
                try:
                    added.append(await api_client.upload_document(
                        name=name, mime_type=mime_type, data=data,
                    ))
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"{name}: {exc}", type="negative", multi_line=True, close_button="OK")
            if added:
                st["document_sources"] = st["document_sources"] + added
            sender.reset()
            panel.refresh()

        ui.upload(
            on_multi_upload=on_docs_upload, auto_upload=True, multiple=True,
            label="Choose documents (PDF, txt, md)",
        ).props('accept=".pdf,.txt,.md" flat bordered').classes("w-full")

        @ui.refreshable
        def panel() -> None:
            for index, doc in enumerate(st["document_sources"]):
                def _remove(i: int = index) -> None:
                    docs = list(st["document_sources"])
                    del docs[i]
                    st["document_sources"] = docs
                    panel.refresh()

                _source_row(doc["name"], "description", _remove)

            has_docs = bool(st["document_sources"])
            can_generate = (has_docs or has_video) and not st.get("generating")

            with ui.row().classes("w-full items-center gap-2 mt-2"):
                ui.button("← Back", on_click=lambda: (st.update(step=1), body.refresh())).props("flat")
                gen = ui.button("Generate Socratic learning", on_click=run_generation).props(
                    "unelevated size=lg color=primary"
                ).classes("grow")
                if not can_generate:
                    gen.disable()

            if st.get("generating"):
                ui.label("Generation in progress…").classes("text-xs text-gray-500")
            elif not (has_docs or has_video):
                ui.label("Upload a document above to enable Generate.").classes("text-xs text-gray-500")
            elif has_video and not has_docs:
                ui.label(
                    "Documents are optional — generate with just the video, or add some."
                ).classes("text-xs text-gray-400")
            else:
                ui.label(
                    "Estimated time: a few minutes (a local model's first run also loads it into memory)."
                ).classes("text-xs text-gray-400")

        stage_label = ui.label("").classes("text-primary")
        spinner = ui.spinner(size="lg")
        spinner.visible = False

        def _friendly_stage(stage: str, status: str) -> str:
            if not stage or status == "cache_hit":
                return ""
            if stage == "pipeline_start":
                return "Starting the pipeline…"
            if stage.startswith("ledger:"):
                parts = stage.split(":")
                name = parts[1] if len(parts) >= 3 else ""
                part = parts[2] if len(parts) >= 3 else ""
                suffix = f" — {name}, part {part}" if name and part else ""
                return f"Extracting knowledge{suffix}…"
            if stage.startswith("consolidation:"):
                return f"Building the deep dive — {stage.split(':', 1)[1]}…"
            if stage in ("synthesis:insights", "deepening:insights"):
                return "Synthesizing insights…"
            if stage == "quiz":
                return "Writing the quiz…"
            return ""

        async def _poll_progress() -> None:
            try:
                payload = await api_client.generate_status()
            except Exception:  # noqa: BLE001 — progress is best-effort
                return
            text = _friendly_stage(
                str(payload.get("stage_name") or ""), str(payload.get("status") or "")
            )
            if text:
                stage_label.text = text

        async def run_generation() -> None:
            if st.get("generating"):
                return
            st["generating"] = True
            spinner.visible = True
            panel.refresh()
            progress_timer = ui.timer(2.0, _poll_progress)
            try:
                video_source_id = st["video_source"]["source_id"] if st["video_source"] else None
                document_source_ids = [int(d["source_id"]) for d in st["document_sources"]]

                stage_label.text = "Generating tailored learning…"
                generation_payload = await api_client.generate(
                    video_source_id=video_source_id, document_source_ids=document_source_ids,
                )

                st["video_source_id"] = video_source_id
                st["document_source_ids"] = document_source_ids
                st["source_ids"] = (
                    ([video_source_id] if video_source_id is not None else []) + document_source_ids
                )
                st["generation_result"] = generation_payload
                st["pipeline_ready"] = True
                st["ask_messages"] = []
                st["quiz_selected"] = {}
                ui.notify("Tailored Socratic learning generated.", type="positive")
                body.refresh()
            except Exception as exc:  # noqa: BLE001 — surface the failure to the user
                stage_label.text = ""
                ui.notify(str(exc), type="negative", multi_line=True, close_button="OK")
            finally:
                progress_timer.cancel()
                spinner.visible = False
                st["generating"] = False
                if not st["pipeline_ready"]:  # panel is gone once the dashboard rendered
                    panel.refresh()

        panel()
