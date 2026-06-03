"""Pure formatting / parsing / prompt-building helpers.

Carried over verbatim from the old Streamlit frontend (they never depended on
Streamlit). Shared by the NiceGUI pages, components, and export buttons.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse


_LEDGER_ID_RE = re.compile(r"\[u\d+[^\]]*\]")              # "[u4 | mechanism | source 2]"
_EVIDENCE_RE = re.compile(r"\(evidence:[^)]*\)", re.IGNORECASE)
_SOURCE_NUM_RE = re.compile(r"\(?\b[Ss]ource\s*#?\s*\d+\)?")  # "source 8", "(source 2)", "source #3"


def strip_source_artifacts(text_value: str) -> str:
    """Remove leaked ledger scaffolding from display prose: unit-id brackets,
    "(evidence: ...)" notes, and raw "source <n>" tokens. Safety net over the
    prompt rule that forbids the model from emitting them."""
    t = str(text_value or "")
    t = _LEDGER_ID_RE.sub("", t)
    t = _EVIDENCE_RE.sub("", t)
    t = _SOURCE_NUM_RE.sub("", t)
    t = re.sub(r"\s+([.,;:])", r"\1", t)   # tidy space before punctuation
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
SUPPORTED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "www.youtu.be",
}

SUMMARY_SECTION_TITLES = {
    "core thesis and scope": "Core Thesis and Scope",
    "key mechanisms and how they work": "Key Mechanisms and How They Work",
    "practical implications and limitations": "Practical Implications and Limitations",
}


# ── File / URL validation ─────────────────────────────────────────────────────
def file_extension(filename: str) -> str:
    lower = filename.lower().strip()
    if "." not in lower:
        return ""
    return "." + lower.rsplit(".", 1)[1]


def is_supported_video_name(filename: str) -> bool:
    return file_extension(filename) in SUPPORTED_VIDEO_EXTENSIONS


def normalize_youtube_url(url_value: str) -> str:
    raw = url_value.strip()
    if not raw:
        return ""

    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
    except Exception:
        return ""

    host = (parsed.netloc or "").lower()
    if host.startswith("www.") and host not in {"www.youtube.com", "www.youtube-nocookie.com"}:
        host = host[4:]

    if host not in SUPPORTED_YOUTUBE_HOSTS:
        return ""

    path = (parsed.path or "").strip()
    if host == "youtu.be":
        video_id = path.strip("/")
        return candidate if video_id else ""

    normalized_path = path.rstrip("/")
    if normalized_path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [""])[0].strip()
        return candidate if video_id else ""

    for prefix in ("/shorts/", "/embed/", "/live/"):
        if normalized_path.startswith(prefix):
            tail = normalized_path[len(prefix):].strip("/")
            return candidate if tail else ""

    return ""


def is_supported_youtube_url(url_value: str) -> bool:
    return bool(normalize_youtube_url(url_value))


def are_supported_document_names(filenames: list[str]) -> tuple[bool, list[str]]:
    invalid = [name for name in filenames if file_extension(name) not in SUPPORTED_DOCUMENT_EXTENSIONS]
    return len(invalid) == 0, invalid


# ── Text normalization / markdown shaping ─────────────────────────────────────
def slugify_token(value: str, *, fallback: str = "item") -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return token or fallback


def shorten_inline_text(text_value: str, *, max_chars: int) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def normalize_continuous_text_for_display(text_value: str) -> str:
    lines = str(text_value or "").splitlines()
    cleaned_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^#{1,6}(?=\S)", "", line).strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"^>\s+", "", line)
        line = re.sub(r"#{2,}", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    paragraphs: list[str] = []
    current: list[str] = []
    for line in cleaned_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _looks_structured_markdown(text_value: str) -> bool:
    normalized = str(text_value or "").strip()
    if not normalized:
        return False
    if "\n\n" in normalized:
        return True

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return False

    for line in lines:
        if line.startswith(("#", "- ", "* ", "> ")):
            return True
        if re.match(r"^\d+\.\s", line):
            return True
    return False


def format_long_prose_markdown(text_value: str, *, max_sentences_per_paragraph: int = 3) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if not normalized:
        return ""
    if _looks_structured_markdown(text_value):
        return str(text_value).strip()
    if len(normalized) < 360:
        return normalized

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    if len(sentences) <= max_sentences_per_paragraph:
        return normalized

    paragraph_size = max(2, max_sentences_per_paragraph)
    paragraphs = [
        " ".join(sentences[index:index + paragraph_size]).strip()
        for index in range(0, len(sentences), paragraph_size)
    ]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def parse_summary_sections(summary_text: str) -> list[tuple[str, str]]:
    lines = str(summary_text or "").splitlines()
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_body: list[str] = []

    def _normalize_heading(value: str) -> str:
        cleaned = re.sub(r"^#{1,6}\s*", "", value.strip())
        cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
        return cleaned

    def _flush() -> None:
        nonlocal current_title, current_body
        body_text = "\n".join(current_body).strip()
        if current_title and body_text:
            sections.append((current_title, body_text))
        current_title = ""
        current_body = []

    for line in lines:
        stripped = line.strip()
        normalized_heading = _normalize_heading(stripped)
        canonical = SUMMARY_SECTION_TITLES.get(normalized_heading)
        if canonical:
            _flush()
            current_title = canonical
            continue

        if current_title:
            current_body.append(line)

    _flush()
    return sections


# ── Chat / quiz prompt builders ───────────────────────────────────────────────
def strip_socratic_follow_up(text_value: str) -> str:
    marker = "**Socratic next question:**"
    cleaned = str(text_value or "")
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def extract_recent_assistant_context(ask_messages: list[dict], *, limit: int = 2) -> str:
    assistant_fragments: list[str] = []
    for message in reversed(ask_messages):
        if str(message.get("role") or "") != "assistant":
            continue
        cleaned = strip_socratic_follow_up(str(message.get("content") or ""))
        if not cleaned:
            continue
        assistant_fragments.append(cleaned)
        if len(assistant_fragments) >= limit:
            break

    if not assistant_fragments:
        return ""
    return "\n\n".join(reversed(assistant_fragments))


def build_quick_quiz_prompt(ask_messages: list[dict]) -> str:
    recent_context = extract_recent_assistant_context(ask_messages, limit=2)
    compact_context = shorten_inline_text(recent_context, max_chars=900)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Start quiz mode using grounded material and our recent conversation. "
        "Ask exactly one challenging question now and do not reveal the answer yet. "
        "Do not add extra headers, markdown bullets, or answer keys. "
        f"Recent assistant context: {compact_context}"
    ).strip()


def build_quiz_grading_prompt(*, ask_messages: list[dict], user_answer: str) -> str:
    recent_context = extract_recent_assistant_context(ask_messages, limit=2)
    compact_context = shorten_inline_text(recent_context, max_chars=900)
    compact_answer = shorten_inline_text(user_answer, max_chars=700)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Continue quiz mode. Grade my latest quiz answer now. "
        "Return a 0-10 score, then concise feedback in this order: what I got right, what I missed, one correction from grounded evidence, and one improvement tip. "
        "Do not ask another quiz question in this message. "
        "Do not use markdown headings or bullet lists. "
        f"My answer: {compact_answer}. "
        f"Recent assistant context: {compact_context}"
    ).strip()


def build_quiz_continue_prompt(ask_messages: list[dict]) -> str:
    recent_context = extract_recent_assistant_context(ask_messages, limit=2)
    compact_context = shorten_inline_text(recent_context, max_chars=850)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Continue quiz mode. Ask exactly one new, more elaborate question connected to recent material. "
        "You may connect the question to another source when grounded evidence supports it. "
        "Do not reveal the answer yet and do not include grading in this message. "
        f"Recent assistant context: {compact_context}"
    ).strip()


def build_quick_elaboration_prompt(ask_messages: list[dict]) -> str:
    recent_context = extract_recent_assistant_context(ask_messages, limit=1)
    compact_context = shorten_inline_text(recent_context, max_chars=820)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Elaborate further on your previous answer with deeper mechanism-level detail in plain language. "
        "Do not restate or paraphrase prior wording. Add net-new causal steps, assumptions, constraints, edge cases, and one practical diagnostic check. "
        "Do not use markdown headings or bullet lists. "
        f"Previous assistant answer context: {compact_context}"
    ).strip()


# ── Source labelling / cross-source visibility ────────────────────────────────
def resolve_source_label_map(video_payload: dict | None, document_payloads: list[dict]) -> dict[int, str]:
    label_map: dict[int, str] = {}

    if video_payload:
        video_source_id = int(video_payload.get("source_id", 0) or 0)
        if video_source_id > 0:
            video_title = (
                str(video_payload.get("source_name") or video_payload.get("generated_title") or "Video").strip()
                or "Video"
            )
            label_map[video_source_id] = f"Video: {video_title}"

    for index, document_payload in enumerate(document_payloads, start=1):
        source_id = int(document_payload.get("source_id", 0) or 0)
        if source_id <= 0:
            continue
        document_title = (
            str(
                document_payload.get("source_name")
                or document_payload.get("generated_title")
                or f"Document {index}"
            ).strip()
            or f"Document {index}"
        )
        label_map[source_id] = f"Document: {document_title}"

    return label_map


def should_show_cross_source_section(
    *,
    generation_result: dict,
    insights_payload: dict,
    quiz_payload: dict,
) -> bool:
    source_ids = generation_result.get("source_ids") or []
    if len(source_ids) < 2:
        return False

    return any(
        [
            bool(insights_payload.get("intersections")),
            bool(insights_payload.get("parallels")),
            bool(str(insights_payload.get("synthesis_text") or "").strip()),
            bool(str(insights_payload.get("layman_bridge") or "").strip()),
            bool(str(insights_payload.get("comparative_analysis") or "").strip()),
            bool(insights_payload.get("application_scenarios")),
            bool(quiz_payload.get("questions")),
        ]
    )


# ── Export builders ───────────────────────────────────────────────────────────
def _append_export_markdown_section(lines: list[str], heading: str, text_value: str) -> None:
    cleaned_text = normalize_continuous_text_for_display(text_value)
    if not cleaned_text:
        return
    lines.append(f"### {heading}")
    lines.append(cleaned_text)
    lines.append("")


def _build_source_export_markdown(section_payload: dict, *, fallback_title: str) -> list[str]:
    source_type = str(section_payload.get("source_type") or "source").strip().lower()
    source_prefix = "Video" if source_type == "video" else "Document" if source_type == "document" else "Source"
    generated_title = str(section_payload.get("generated_title") or fallback_title).strip() or fallback_title
    source_name = str(section_payload.get("source_name") or "").strip()
    source_id = int(section_payload.get("source_id", 0) or 0)

    lines: list[str] = [f"## {source_prefix}: {generated_title}"]
    lines.append(f"- Source ID: {source_id if source_id > 0 else 'n/a'}")
    if source_name and source_name.lower() != generated_title.lower():
        lines.append(f"- Original name: {source_name}")

    lines.append("")

    _append_export_markdown_section(lines, "Core Thesis", str(section_payload.get("summary_text") or ""))
    _append_export_markdown_section(lines, "Mechanisms & Relationships", str(section_payload.get("deep_dive_text") or ""))
    _append_export_markdown_section(
        lines,
        "Open Questions & Limitations",
        str(section_payload.get("under_surface_explainer") or ""),
    )

    key_term_explanations = section_payload.get("key_term_explanations") or []
    if key_term_explanations:
        lines.append("### Glossary")
        for entry in key_term_explanations:
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term") or "").strip()
            if not term:
                continue
            layman = normalize_continuous_text_for_display(str(entry.get("layman") or ""))
            technical = normalize_continuous_text_for_display(str(entry.get("technical") or ""))
            definition = " ".join(p for p in [layman, technical] if p)
            lines.append(f"- **{term}**: {definition}" if definition else f"- **{term}**")
        lines.append("")
    else:
        key_terms = [str(term).strip() for term in (section_payload.get("key_terms") or []) if str(term).strip()]
        if key_terms:
            lines.append("### Glossary")
            lines.append(f"- Key terms: {', '.join(key_terms)}")
            lines.append("")

    reflection_points = section_payload.get("reflection_points") or []
    if reflection_points:
        lines.append("### Retrieval Practice")
        for index, point in enumerate(reflection_points, start=1):
            if isinstance(point, dict):
                question = normalize_continuous_text_for_display(str(point.get("question") or ""))
                explanation = normalize_continuous_text_for_display(str(point.get("explanation") or ""))
                reasoning_traps = normalize_continuous_text_for_display(str(point.get("reasoning_traps") or point.get("under_the_hood") or ""))
                if question:
                    lines.append(f"{index}. {question}")
                if explanation:
                    lines.append(f"   - Why: {explanation}")
                if reasoning_traps:
                    lines.append(f"   - Reasoning traps: {reasoning_traps}")
            else:
                cleaned_point = normalize_continuous_text_for_display(str(point))
                if cleaned_point:
                    lines.append(f"{index}. {cleaned_point}")
        lines.append("")

    return lines


def build_generation_export_markdown(generation_result: dict, *, user_id: str) -> str:
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Socratic Study Snapshot",
        "",
        f"- Exported at: {exported_at}",
        f"- User: {str(user_id or '').strip() or 'anonymous'}",
        f"- Source IDs: {', '.join(str(value) for value in (generation_result.get('source_ids') or [])) or 'none'}",
        "",
    ]

    video_payload = generation_result.get("video") or None
    document_payloads = generation_result.get("documents", []) or []

    if video_payload or document_payloads:
        lines.append("# Source Learning")
        lines.append("")
        if video_payload:
            lines.extend(_build_source_export_markdown(video_payload, fallback_title="Video"))
        for index, document_payload in enumerate(document_payloads, start=1):
            lines.extend(
                _build_source_export_markdown(
                    document_payload,
                    fallback_title=f"Document {index}",
                )
            )

    insights_payload = generation_result.get("insights") or {}
    quiz_payload = generation_result.get("quiz") or {}
    intersections = insights_payload.get("intersections", []) or []

    if insights_payload or quiz_payload:
        lines.append("# Cross-Source Synthesis")
        lines.append("")

    takeaways = [str(t).strip() for t in (insights_payload.get("key_takeaways") or []) if str(t).strip()]
    if takeaways:
        lines.append("## Key Takeaways")
        for i, t in enumerate(takeaways[:3], start=1):
            lines.append(f"{i}. {strip_source_artifacts(t)}")
        lines.append("")

    synthesis_text = normalize_continuous_text_for_display(
        strip_source_artifacts(str(insights_payload.get("synthesis_text") or ""))
    )
    if synthesis_text or intersections:
        lines.append("## Comparative Tradeoffs")
        lines.append("")
        if synthesis_text:
            lines.append(synthesis_text)
            lines.append("")
        for index, intersection in enumerate(intersections, start=1):
            if not isinstance(intersection, dict):
                continue
            title = str(intersection.get("title") or f"Intersection {index}").strip()
            lines.append(f"### {index}. {title}")

            why_it_matters = normalize_continuous_text_for_display(str(intersection.get("why_it_matters") or ""))
            integrated_explanation = normalize_continuous_text_for_display(
                str(intersection.get("integrated_explanation") or "")
            )
            if why_it_matters:
                lines.append(f"- Why it matters: {why_it_matters}")
            if integrated_explanation:
                lines.append(f"- Integrated explanation: {integrated_explanation}")

            attributed_sentences = intersection.get("attributed_sentences", []) or []
            if attributed_sentences:
                lines.append("- Grounding evidence:")
                for sentence in attributed_sentences:
                    if not isinstance(sentence, dict):
                        continue
                    quote = normalize_continuous_text_for_display(str(sentence.get("text") or ""))
                    source_id = int(sentence.get("source_id", 0) or 0)
                    if quote:
                        lines.append(f"  - Source {source_id if source_id > 0 else '?'}: {quote}")
            lines.append("")

    application_scenarios = insights_payload.get("application_scenarios", []) or []
    if application_scenarios:
        lines.append("## Operational Implications")
        lines.append("")
        for index, scenario in enumerate(application_scenarios, start=1):
            if not isinstance(scenario, dict):
                continue
            title = str(scenario.get("scenario_title") or f"Scenario {index}").strip()
            lines.append(f"### {index}. {title}")
            prompt_text = normalize_continuous_text_for_display(str(scenario.get("scenario_prompt") or ""))
            if prompt_text:
                lines.append(f"- Prompt: {prompt_text}")
            transfer_steps = scenario.get("transfer_steps") or []
            if transfer_steps:
                lines.append("- Transfer steps:")
                for step_index, step_text in enumerate(transfer_steps, start=1):
                    cleaned_step = normalize_continuous_text_for_display(str(step_text))
                    if cleaned_step:
                        lines.append(f"  {step_index}. {cleaned_step}")
            pitfall_text = normalize_continuous_text_for_display(str(scenario.get("common_pitfall") or ""))
            if pitfall_text:
                lines.append(f"- Common pitfall: {pitfall_text}")
            lines.append("")

    questions = quiz_payload.get("questions", []) or []
    if questions:
        lines.append("## Retrieval Practice")
        lines.append("")
        for index, question_payload in enumerate(questions, start=1):
            if not isinstance(question_payload, dict):
                continue
            question = normalize_continuous_text_for_display(str(question_payload.get("question") or ""))
            if question:
                lines.append(f"### Q{index}. {question}")
            options = question_payload.get("options", []) or []
            option_explanations = question_payload.get("option_explanations", []) or []
            answer_index = int(question_payload.get("answer_index", 0) or 0)
            for option_index, option in enumerate(options):
                cleaned_option = normalize_continuous_text_for_display(str(option))
                if not cleaned_option:
                    continue
                marker = " (correct)" if option_index == answer_index else ""
                lines.append(f"{option_index + 1}. {cleaned_option}{marker}")
                if option_index < len(option_explanations):
                    rationale = normalize_continuous_text_for_display(str(option_explanations[option_index]))
                    if rationale:
                        lines.append(f"   - {rationale}")

            explanation = normalize_continuous_text_for_display(str(question_payload.get("explanation") or ""))
            if explanation:
                lines.append(f"- Why this is correct: {explanation}")
            lines.append("")

    study_advice = normalize_continuous_text_for_display(str(quiz_payload.get("study_advice") or ""))
    if study_advice:
        lines.append("## Study Advice")
        lines.append(study_advice)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_generation_export_json(generation_result: dict, *, user_id: str) -> str:
    export_payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "user_id": str(user_id or "").strip() or None,
        "generation_result": generation_result,
    }
    return json.dumps(export_payload, ensure_ascii=True, indent=2)
