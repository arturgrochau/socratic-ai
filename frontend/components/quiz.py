"""Interactive, pedagogical quiz.

Click an option to answer. The first pick locks the question (no retry): the
correct option turns green with a check; a wrong pick turns red and shows the
correction tied to that specific misconception. When every question is
answered, a button hands off into Socratic chat seeded with the quiz context.
"""
from __future__ import annotations

from typing import Any, Callable

from nicegui import ui

from frontend import state


_LABELS = ["A", "B", "C", "D", "E", "F"]


def _build_seed(questions: list[dict[str, Any]], selected: dict[str, int]) -> str:
    missed = [
        str(q.get("question", "")).strip()
        for qi, q in enumerate(questions)
        if selected.get(str(qi)) != int(q.get("answer_index", 0) or 0)
    ]
    missed = [m for m in missed if m]
    if missed:
        return (
            "I just finished the assessment quiz and got these questions wrong: "
            + " | ".join(missed)
            + ". Explain the underlying concepts I clearly misunderstood, in depth "
            "and from first principles, then check my understanding with one "
            "follow-up question."
        )
    return (
        "I just finished the assessment quiz and answered every question correctly. "
        "Push my understanding further with a harder angle on the same material, "
        "then ask me one challenging follow-up question."
    )


def render_quiz(quiz: dict[str, Any], *, on_discuss: Callable[[str], None]) -> None:
    questions: list[dict[str, Any]] = quiz.get("questions", []) or []
    if not questions:
        ui.label("No quiz questions available.").classes("text-sm text-gray-500")
        return

    @ui.refreshable
    def panel() -> None:
        st = state.tab()
        selected: dict[str, int] = st["quiz_selected"]

        for qi, q in enumerate(questions):
            options = q.get("options", []) or []
            answer_index = int(q.get("answer_index", 0) or 0)
            rationales = q.get("option_explanations", []) or []
            key = str(qi)
            locked = key in selected
            chosen = selected.get(key)

            with ui.card().classes("w-full"):
                ui.label(f"Q{qi + 1}. {q.get('question', '')}").classes("font-semibold")

                for oi, opt in enumerate(options):
                    prefix = _LABELS[oi] if oi < len(_LABELS) else str(oi + 1)
                    text = f"{prefix}.  {opt}"

                    if not locked:
                        def _pick(_qi: int = qi, _oi: int = oi) -> None:
                            new_selected = dict(selected)
                            new_selected[str(_qi)] = _oi
                            st["quiz_selected"] = new_selected  # reassign so storage persists
                            panel.refresh()

                        ui.button(text, on_click=_pick).props("flat no-caps align=left").classes(
                            "w-full justify-start normal-case"
                        )
                    else:
                        classes = "w-full justify-start normal-case "
                        icon = ""
                        if oi == answer_index:
                            classes += "bg-green-2 text-green-10"
                            icon = "check"
                        elif oi == chosen:
                            classes += "bg-red-2 text-red-10"
                            icon = "close"
                        else:
                            classes += "text-grey-7"
                        btn = ui.button(text).props("flat no-caps align=left").classes(classes)
                        if icon:
                            btn.props(f"icon={icon}")
                        btn.disable()

                if locked:
                    if chosen == answer_index:
                        ui.label("Correct.").classes("text-green-9 font-medium")
                    elif chosen is not None and 0 <= chosen < len(rationales):
                        ui.markdown(f"**{rationales[chosen]}**").classes("text-red-9")
                    if chosen != answer_index and 0 <= answer_index < len(rationales):
                        ui.markdown(
                            f"Correct answer ({_LABELS[answer_index]}): {rationales[answer_index]}"
                        ).classes("text-green-9")
                    explanation = str(q.get("explanation", "")).strip()
                    if explanation:
                        ui.markdown(explanation).classes("text-sm text-gray-600")

        total = len(questions)
        answered = sum(1 for qi in range(total) if str(qi) in selected)
        correct = sum(
            1
            for qi, q in enumerate(questions)
            if selected.get(str(qi)) == int(q.get("answer_index", 0) or 0)
        )

        ui.separator()
        ui.label(f"Answered {answered}/{total} · {correct} correct").classes(
            "text-sm text-primary font-medium"
        )
        if answered >= total:
            ui.button(
                "Discuss in Socratic chat",
                icon="forum",
                on_click=lambda: on_discuss(_build_seed(questions, selected)),
            ).props("unelevated color=primary")

    panel()
