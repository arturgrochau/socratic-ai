"""Learning dashboard: per-source tabs, cross-source synthesis, quiz, chat."""
from __future__ import annotations

from typing import Any

from nicegui import ui

from frontend import format as fmt, state
from frontend.components import export
from frontend.components.learning_section import (
    render_cross_source,
    render_source_learning_section,
)
from frontend.pages.chat import build_chat_panel


def render_dashboard() -> None:
    st = state.tab()
    generation_result: dict[str, Any] = st.get("generation_result") or {}
    if not generation_result:
        ui.label("Generate tailored content to populate the dashboard.").classes("text-gray-500")
        return

    ui.label("Learning Dashboard").classes("text-2xl font-bold mt-4")
    export.render_export_buttons(generation_result, user_id=state.user_id())

    video_payload = generation_result.get("video") or None
    document_payloads = generation_result.get("documents", []) or []
    insights_payload = generation_result.get("insights", {}) or {}
    quiz_payload = generation_result.get("quiz", {}) or {}

    show_cross = fmt.should_show_cross_source_section(
        generation_result=generation_result,
        insights_payload=insights_payload,
        quiz_payload=quiz_payload,
    )

    with ui.tabs().classes("w-full") as tabs:
        if video_payload:
            ui.tab("Video")
        if document_payloads:
            ui.tab("Documents")
        if show_cross:
            ui.tab("Cross-Source")
        ui.tab("Socratic Chat")

    # Default to the most useful starting tab (cross-source if present, else first).
    default_tab = "Cross-Source" if show_cross else ("Video" if video_payload else (
        "Documents" if document_payloads else "Socratic Chat"))

    # Lets the quiz hand off into the chat tab, seeding it with a question.
    chat_handle: dict[str, Any] = {}

    def on_discuss(seed: str) -> None:
        st["pending_chat_prompt"] = seed
        tabs.set_value("Socratic Chat")
        refresh = chat_handle.get("refresh")
        if refresh:
            refresh()

    with ui.tab_panels(tabs, value=default_tab).classes("w-full"):
        if video_payload:
            with ui.tab_panel("Video"):
                render_source_learning_section(video_payload)

        if document_payloads:
            with ui.tab_panel("Documents"):
                if len(document_payloads) == 1:
                    render_source_learning_section(document_payloads[0])
                else:
                    with ui.tabs().classes("w-full") as doc_tabs:
                        for index, _ in enumerate(document_payloads, start=1):
                            ui.tab(f"Doc {index}")
                    with ui.tab_panels(doc_tabs, value="Doc 1").classes("w-full"):
                        for index, section in enumerate(document_payloads, start=1):
                            with ui.tab_panel(f"Doc {index}"):
                                render_source_learning_section(section)

        if show_cross:
            with ui.tab_panel("Cross-Source"):
                render_cross_source(insights_payload, quiz_payload, on_discuss=on_discuss)

        with ui.tab_panel("Socratic Chat"):
            chat_handle["refresh"] = build_chat_panel()
