"""Socratic chat panel with the quiz-mode state machine (ported 1:1)."""
from __future__ import annotations

from nicegui import ui

from frontend import api_client, state
from frontend import format as fmt


def build_chat_panel():
    """Render the chat panel for the current tab. Safe to call inside a tab panel.

    Returns the panel's refresh callable so other tabs (the quiz) can seed a
    question into `pending_chat_prompt` and trigger an auto-submit here.
    """

    # The element that receives streamed tokens for the in-flight reply. Kept
    # outside the refreshable so submit() can find it after a refresh.
    live: dict[str, ui.markdown | None] = {"md": None}

    @ui.refreshable
    def panel() -> None:
        st = state.tab()
        if not st["pipeline_ready"]:
            ui.label("Generate tailored content before asking questions.").classes("text-gray-500")
            return

        ui.label("Socratic Chatbox").classes("text-lg font-semibold")
        ui.label(
            "Ask follow-up questions to deepen understanding across your sources and insights."
        ).classes("text-sm text-gray-500")

        messages = st["ask_messages"]
        with ui.column().classes("w-full gap-2"):
            for message in messages:
                ui.chat_message(
                    message["content"],
                    sent=(message["role"] == "user"),
                    name="You" if message["role"] == "user" else "Socratic AI",
                ).classes("w-full")
            if st.get("ask_pending"):
                with ui.chat_message(name="Socratic AI").classes("w-full"):
                    with ui.row().classes("items-center gap-2") as thinking:
                        ui.spinner(size="sm")
                        ui.label("Thinking…").classes("text-gray-500")
                    live["md"] = ui.markdown("").classes("w-full")
                    live["thinking"] = thinking

        async def submit(query: str, *, display: str | None = None) -> None:
            query_text = (query or "").strip()
            if not query_text:
                return
            display_text = (display or query_text).strip() or query_text
            messages.append({"role": "user", "content": display_text})
            st["ask_pending"] = True
            panel.refresh()
            try:
                buffer = ""
                async for event in api_client.ask_stream(
                    session_id=st["session_id"],
                    source_ids=st["source_ids"],
                    query=query_text,
                ):
                    if "delta" in event:
                        if not buffer and live.get("thinking") is not None:
                            live["thinking"].set_visibility(False)
                        buffer += str(event["delta"])
                        if live["md"] is not None:
                            live["md"].set_content(buffer)
                    elif event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    elif event.get("done"):
                        buffer = str(event.get("answer") or buffer)
                answer = buffer.strip() or (
                    "I could not generate an answer this time. Please try rephrasing your question."
                )
                messages.append({"role": "assistant", "content": answer})
            except Exception as exc:  # noqa: BLE001 — surface errors inline like the old UI
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"I ran into an error while answering that request: {exc}. "
                        "Please try asking again with a more specific wording."
                    ),
                })
            st["ask_pending"] = False
            panel.refresh()

        # Quiz handoff: if another tab seeded a prompt, auto-ask it once.
        pending = str(st.get("pending_chat_prompt") or "").strip()
        if pending:
            st["pending_chat_prompt"] = ""
            ui.timer(0.1, lambda p=pending: submit(p, display="(from quiz) " + p[:80] + "…"), once=True)

        # Quiz-mode controls (mirror the Streamlit state machine).
        if st["quiz_mode_active"]:
            if st["quiz_awaiting_answer"]:
                ui.label(
                    "Quiz mode is active. Answer the current quiz question below to receive grading."
                ).classes("text-blue-700")
            else:
                ui.label(
                    f"Quiz feedback complete for {int(st['quiz_turn_count'])} question(s). "
                    "Continue for another question or exit quiz mode."
                ).classes("text-green-700")

                async def continue_quiz() -> None:
                    st["quiz_awaiting_answer"] = True
                    await submit(
                        fmt.build_quiz_continue_prompt(messages), display="Continue quiz."
                    )

                def exit_quiz() -> None:
                    st["quiz_mode_active"] = False
                    st["quiz_awaiting_answer"] = False
                    st["quiz_turn_count"] = 0
                    panel.refresh()

                with ui.row():
                    ui.button("Continue quiz", on_click=continue_quiz).props("outline")
                    ui.button("Exit quiz mode", on_click=exit_quiz).props("outline")

        # Quick actions after an assistant turn (non-quiz).
        if messages and messages[-1]["role"] == "assistant" and not st["quiz_mode_active"]:
            async def elaborate() -> None:
                await submit(fmt.build_quick_elaboration_prompt(messages))

            async def quiz_me() -> None:
                st["quiz_mode_active"] = True
                st["quiz_awaiting_answer"] = True
                st["quiz_turn_count"] = 0
                await submit(
                    fmt.build_quick_quiz_prompt(messages), display="Quiz me on this."
                )

            with ui.row():
                ui.button("Elaborate further", on_click=elaborate).props("outline")
                ui.button("Quiz me on this", on_click=quiz_me).props("outline")

        # Input row.
        placeholder = (
            "Ask a follow-up..."
            if messages
            else "Ask your first question — e.g. 'Explain the strongest intersection between these sources.'"
        )
        with ui.row().classes("w-full no-wrap items-center"):
            box = ui.input(placeholder=placeholder).classes("grow").props("outlined dense")

            async def on_send() -> None:
                value = (box.value or "").strip()
                if not value:
                    return
                box.value = ""
                # Quiz answer being graded.
                if st["quiz_mode_active"] and st["quiz_awaiting_answer"]:
                    grading = fmt.build_quiz_grading_prompt(ask_messages=messages, user_answer=value)
                    st["quiz_awaiting_answer"] = False
                    st["quiz_turn_count"] = int(st["quiz_turn_count"]) + 1
                    await submit(grading, display=value)
                    return
                # Any other submission exits quiz feedback mode.
                if st["quiz_mode_active"] and not st["quiz_awaiting_answer"]:
                    st["quiz_mode_active"] = False
                    st["quiz_turn_count"] = 0
                await submit(value)

            box.on("keydown.enter", on_send)
            ui.button(icon="send", on_click=on_send).props("round color=primary")

    panel()
    return panel.refresh
