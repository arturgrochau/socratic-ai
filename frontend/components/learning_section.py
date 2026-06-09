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

    def _clean(value: str) -> str:
        return fmt.normalize_continuous_text_for_display(fmt.strip_source_artifacts(value))

    with ui.column().classes("w-full gap-3"):
        ui.label(f"{prefix}: {generated_title}").classes("text-xl font-semibold")
        ui.label("Pulled from this source.").classes("text-xs text-gray-400 -mt-2")
        if source_name and source_name.lower() != generated_title.lower():
            ui.label(source_name).classes("text-sm text-gray-500")

        # Summary
        summary = _clean(str(section.get("summary_text", "")))
        if summary:
            ui.markdown(fmt.format_long_prose_markdown(summary, max_sentences_per_paragraph=4))

        # Deep dive (merge deep_dive + under-surface, matching the old UI)
        deep_parts: list[str] = []
        deep_dive = _clean(str(section.get("deep_dive_text", "")))
        if deep_dive:
            deep_parts.append(deep_dive)
        under_surface = _clean(str(section.get("under_surface_explainer", "")))
        if under_surface:
            deep_parts.append(under_surface)
        if deep_parts:
            ui.label("Deep Dive").classes("text-lg font-semibold mt-2")
            ui.markdown(fmt.format_long_prose_markdown("\n\n".join(deep_parts), max_sentences_per_paragraph=4))

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

        ui.markdown(f"**Q{index}. {question}**")
        if explanation:
            ui.markdown(fmt.format_long_prose_markdown(explanation))


def render_cross_source(
    insights: dict[str, Any],
    quiz: dict[str, Any],
    *,
    on_discuss: Callable[[str], None],
    single_source: bool = False,
) -> None:
    if single_source:
        ui.label("Quiz & Applications").classes("text-xl font-semibold")
        ui.label("Going deeper on this source: takeaways, where it applies, and a quiz.").classes(
            "text-xs text-gray-400 -mt-2"
        )
        synthesis_caption = "The bigger picture (AI inference, not a quote)."
        quiz_caption = "Built from this source. Pick an answer to lock it in — no retries; wrong picks explain the trap."
    else:
        ui.label("Cross-Source Synthesis").classes("text-xl font-semibold")
        ui.label("Synthesized across your sources (AI inference, not a quote).").classes(
            "text-xs text-gray-400 -mt-2"
        )
        synthesis_caption = ""
        quiz_caption = (
            "Study aid built from the synthesis. Pick an answer to lock it in — no retries; "
            "wrong picks explain the trap."
        )

    # Hierarchy: the 1-3 headline takeaways first.
    takeaways = [str(t).strip() for t in (insights.get("key_takeaways") or []) if str(t).strip()]
    if takeaways:
        with ui.card().classes("w-full bg-blue-1"):
            ui.label("Key takeaways").classes("text-sm font-semibold text-primary")
            for i, t in enumerate(takeaways[:3], start=1):
                ui.markdown(f"**{i}.** {fmt.strip_source_artifacts(t)}")

    synthesis_text = fmt.normalize_continuous_text_for_display(
        fmt.strip_source_artifacts(str(insights.get("synthesis_text", "")))
    )
    if synthesis_text:
        if synthesis_caption:
            ui.label(synthesis_caption).classes("text-xs text-gray-400")
        ui.markdown(fmt.format_long_prose_markdown(synthesis_text, max_sentences_per_paragraph=4))

    _render_application_scenarios(insights.get("application_scenarios") or [])

    ui.label("Quiz").classes("text-lg font-semibold mt-3")
    ui.label(quiz_caption).classes("text-sm text-gray-500")
    render_quiz(quiz, on_discuss=on_discuss)


def _render_application_scenarios(scenarios: list[Any]) -> None:
    """Render apply-it scenarios (previously generated but never displayed)."""
    scenarios = [s for s in scenarios if isinstance(s, dict)]
    if not scenarios:
        return
    ui.label("Apply it").classes("text-lg font-semibold mt-3")
    ui.label("Transfer the ideas to a concrete situation.").classes("text-sm text-gray-500")
    for scenario in scenarios:
        title = str(scenario.get("scenario_title", "")).strip()
        prompt = fmt.normalize_continuous_text_for_display(str(scenario.get("scenario_prompt", "")))
        steps = [str(s).strip() for s in (scenario.get("transfer_steps") or []) if str(s).strip()]
        pitfall = fmt.normalize_continuous_text_for_display(str(scenario.get("common_pitfall", "")))
        with ui.card().classes("w-full"):
            if title:
                ui.label(title).classes("font-semibold")
            if prompt:
                ui.markdown(prompt)
            if steps:
                ui.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1)))
            if pitfall:
                ui.markdown(f"**Watch out:** {pitfall}")
