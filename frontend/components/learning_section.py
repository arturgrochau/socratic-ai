"""Renders a single source's learning section and the cross-source synthesis."""
from __future__ import annotations

from typing import Any, Callable

from nicegui import ui

from frontend import format as fmt
from frontend.components.quiz import render_quiz


def render_source_learning_section(section: dict[str, Any]) -> None:
    generated_title = str(section.get("generated_title", "Untitled")).strip() or "Untitled"
    source_type = str(section.get("source_type") or "source").strip().lower()
    prefix = "Video" if source_type == "video" else "Document" if source_type == "document" else "Source"
    source_name = str(section.get("source_name") or "").strip()

    with ui.column().classes("w-full gap-3"):
        ui.label(f"{prefix}: {generated_title}").classes("text-xl font-semibold")
        if source_name and source_name.lower() != generated_title.lower():
            ui.label(source_name).classes("text-sm text-gray-500 -mt-2")

        # Summary
        summary = fmt.normalize_continuous_text_for_display(str(section.get("summary_text", "")))
        if summary:
            ui.markdown(fmt.format_long_prose_markdown(summary, max_sentences_per_paragraph=4))

        # Deep dive (merge deep_dive + under-surface, matching the old UI)
        deep_parts: list[str] = []
        deep_dive = fmt.normalize_continuous_text_for_display(str(section.get("deep_dive_text", "")))
        if deep_dive:
            deep_parts.append(deep_dive)
        under_surface = fmt.normalize_continuous_text_for_display(str(section.get("under_surface_explainer", "")))
        if under_surface:
            deep_parts.append(under_surface)
        if deep_parts:
            ui.label("Deep Dive").classes("text-lg font-semibold mt-2")
            ui.markdown(fmt.format_long_prose_markdown("\n\n".join(deep_parts), max_sentences_per_paragraph=4))
            checklist = section.get("diagnostic_checklist", []) or []
            if checklist:
                with ui.expansion("Read more").classes("w-full"):
                    for item in checklist:
                        cleaned = fmt.normalize_continuous_text_for_display(str(item))
                        if cleaned:
                            ui.markdown(f"- {cleaned}")

        # Key concepts (layman + technical collapsibles)
        key_terms = [str(t) for t in (section.get("key_terms", []) or [])]
        key_term_explanations = section.get("key_term_explanations", []) or []
        if key_terms:
            ui.label("Key Concepts").classes("text-lg font-semibold mt-2")
            if key_term_explanations:
                for entry in key_term_explanations:
                    if not isinstance(entry, dict):
                        continue
                    term = str(entry.get("term", "")).strip()
                    if not term:
                        continue
                    with ui.expansion(term).classes("w-full"):
                        layman = str(entry.get("layman", "")).strip()
                        technical = str(entry.get("technical", "")).strip()
                        if layman:
                            ui.markdown(f"**In plain terms:** {layman}")
                        if technical:
                            ui.markdown(f"**Technical:** {technical}")
                        if not layman and not technical:
                            ui.label("No definition available.").classes("text-sm text-gray-500")
            else:
                ui.markdown(" ".join(f"• **{t.strip()}**" for t in key_terms if t.strip()))

        # Reflection
        reflection_points = section.get("reflection_points", []) or []
        if reflection_points:
            ui.label("Reflection").classes("text-lg font-semibold mt-2")
            _render_reflection_points(reflection_points)


def _render_reflection_points(reflection_points: list[Any]) -> None:
    for index, point in enumerate(reflection_points, start=1):
        if isinstance(point, str):
            ui.markdown(f"{index}. {point}")
            continue
        question = str(point.get("question", "")).strip()
        explanation = str(point.get("explanation", "")).strip()
        reasoning_traps = str(point.get("reasoning_traps") or point.get("under_the_hood") or "").strip()

        ui.markdown(f"**Q{index}. {question}**")
        if explanation:
            ui.markdown(fmt.format_long_prose_markdown(explanation))
        if reasoning_traps:
            with ui.expansion("Read more").classes("w-full"):
                ui.markdown(
                    fmt.format_long_prose_markdown(
                        fmt.normalize_continuous_text_for_display(reasoning_traps),
                        max_sentences_per_paragraph=4,
                    )
                )


def render_cross_source(
    insights: dict[str, Any],
    quiz: dict[str, Any],
    *,
    on_discuss: Callable[[str], None],
) -> None:
    ui.label("Cross-Source Synthesis").classes("text-xl font-semibold")
    synthesis_text = fmt.normalize_continuous_text_for_display(str(insights.get("synthesis_text", "")))
    if synthesis_text:
        ui.markdown(fmt.format_long_prose_markdown(synthesis_text, max_sentences_per_paragraph=4))

    ui.label("Quiz").classes("text-lg font-semibold mt-3")
    ui.label("Pick an answer to lock it in — no retries. Wrong picks explain the trap.").classes(
        "text-sm text-gray-500"
    )
    render_quiz(quiz, on_discuss=on_discuss)
