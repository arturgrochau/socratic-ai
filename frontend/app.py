from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from urllib.parse import parse_qs, urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components


def _resolve_timeout_seconds() -> int:
    raw = str(os.getenv("FRONTEND_REQUEST_TIMEOUT", "600")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 600
    return max(120, value)


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = _resolve_timeout_seconds()
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


def _init_state() -> None:
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "source_ids" not in st.session_state:
        st.session_state.source_ids = []
    if "video_source_id" not in st.session_state:
        st.session_state.video_source_id = None
    if "document_source_ids" not in st.session_state:
        st.session_state.document_source_ids = []
    if "interaction_session_id" not in st.session_state:
        st.session_state.interaction_session_id = str(uuid.uuid4())
    if "ask_result" not in st.session_state:
        st.session_state.ask_result = None
    if "ask_query" not in st.session_state:
        st.session_state.ask_query = ""
    if "ask_messages" not in st.session_state:
        st.session_state.ask_messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo-user"
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "wizard_video_upload" not in st.session_state:
        st.session_state.wizard_video_upload = None
    if "wizard_document_uploads" not in st.session_state:
        st.session_state.wizard_document_uploads = []
    if "generation_result" not in st.session_state:
        st.session_state.generation_result = None
    if "dashboard_section" not in st.session_state:
        st.session_state.dashboard_section = "Video"
    if "ask_prefill" not in st.session_state:
        st.session_state.ask_prefill = ""
    if "socratic_chat_text" not in st.session_state:
        st.session_state.socratic_chat_text = ""
    if "pending_dashboard_section" not in st.session_state:
        st.session_state.pending_dashboard_section = ""
    if "wizard_replace_video" not in st.session_state:
        st.session_state.wizard_replace_video = False
    if "wizard_replace_documents" not in st.session_state:
        st.session_state.wizard_replace_documents = False
    if "wizard_video_source_mode" not in st.session_state:
        st.session_state.wizard_video_source_mode = "Paste YouTube link"
    if "wizard_last_video_source_mode" not in st.session_state:
        st.session_state.wizard_last_video_source_mode = "Paste YouTube link"
    if "wizard_video_url" not in st.session_state:
        st.session_state.wizard_video_url = ""
    if "wizard_video_url_input" not in st.session_state:
        st.session_state.wizard_video_url_input = ""
    if "auto_submit_query" not in st.session_state:
        st.session_state.auto_submit_query = ""
    if "flash_latest_assistant" not in st.session_state:
        st.session_state.flash_latest_assistant = False
    if "quiz_mode_active" not in st.session_state:
        st.session_state.quiz_mode_active = False
    if "quiz_awaiting_answer" not in st.session_state:
        st.session_state.quiz_awaiting_answer = False
    if "quiz_turn_count" not in st.session_state:
        st.session_state.quiz_turn_count = 0


def _request_headers() -> dict[str, str]:
    return {"X-User-ID": str(st.session_state.user_id).strip()}


def _extract_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    if not isinstance(payload, dict):
        return str(payload)

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = str(detail.get("message") or "Generation failed.").strip()
        run_id = str(detail.get("run_id") or "").strip()
        stage = str(detail.get("stage") or "").strip()
        reason = str(detail.get("reason") or "").strip()

        fragments = [message]
        if stage:
            fragments.append(f"stage={stage}")
        if run_id:
            fragments.append(f"run_id={run_id}")
        if reason:
            fragments.append(f"reason={reason}")
        return " | ".join(fragments)
    return str(payload)


def _post_json(path: str, body: dict) -> dict:
    response = requests.post(
        f"{st.session_state.api_base_url}{path}",
        json=body,
        headers=_request_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{path} failed: {_extract_error(response)}")
    return response.json()


def _pack_uploaded_file(uploaded_file) -> dict[str, str | bytes]:
    return {
        "name": uploaded_file.name,
        "mime_type": uploaded_file.type or "application/octet-stream",
        "data": uploaded_file.getvalue(),
    }


def _reset_builder() -> None:
    st.session_state.pipeline_ready = False
    st.session_state.source_ids = []
    st.session_state.video_source_id = None
    st.session_state.document_source_ids = []
    st.session_state.ask_result = None
    st.session_state.ask_query = ""
    st.session_state.ask_messages = []
    st.session_state.generation_result = None
    st.session_state.interaction_session_id = str(uuid.uuid4())
    st.session_state.wizard_step = 1
    st.session_state.wizard_video_upload = None
    st.session_state.wizard_document_uploads = []
    st.session_state.dashboard_section = "Video"
    st.session_state.ask_prefill = ""
    st.session_state.socratic_chat_text = ""
    st.session_state.pending_dashboard_section = ""
    st.session_state.wizard_replace_video = False
    st.session_state.wizard_replace_documents = False
    st.session_state.wizard_video_source_mode = "Paste YouTube link"
    st.session_state.wizard_last_video_source_mode = "Paste YouTube link"
    st.session_state.wizard_video_url = ""
    st.session_state.wizard_video_url_input = ""
    st.session_state.auto_submit_query = ""
    st.session_state.flash_latest_assistant = False


def _file_extension(filename: str) -> str:
    lower = filename.lower().strip()
    if "." not in lower:
        return ""
    return "." + lower.rsplit(".", 1)[1]


def _is_supported_video_name(filename: str) -> bool:
    return _file_extension(filename) in SUPPORTED_VIDEO_EXTENSIONS


def _is_supported_youtube_url(url_value: str) -> bool:
    return bool(_normalize_youtube_url(url_value))


def _normalize_youtube_url(url_value: str) -> str:
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


def _are_supported_document_names(filenames: list[str]) -> tuple[bool, list[str]]:
    invalid = [name for name in filenames if _file_extension(name) not in SUPPORTED_DOCUMENT_EXTENSIONS]
    return len(invalid) == 0, invalid


def _queue_chat_navigation(prefill: str, *, auto_submit: bool = False) -> None:
    normalized = prefill.strip()
    if auto_submit:
        st.session_state.auto_submit_query = normalized
        st.session_state.ask_prefill = ""
    else:
        st.session_state.ask_prefill = normalized
        st.session_state.auto_submit_query = ""
    st.session_state.pending_dashboard_section = "Socratic Chatbox"


def _render_chat_scroll_and_flash() -> None:
        components.html(
                """
                <script>
                    const parentDoc = window.parent.document;
                    const messages = parentDoc.querySelectorAll('[data-testid="stChatMessage"]');
                    if (messages.length > 0) {
                        const last = messages[messages.length - 1];
                        last.scrollIntoView({ behavior: "smooth", block: "center" });
                        const previousBoxShadow = last.style.boxShadow;
                        const previousBackground = last.style.backgroundColor;
                        last.style.transition = "box-shadow 180ms ease, background-color 380ms ease";
                        last.style.boxShadow = "0 0 0 2px rgba(250, 176, 5, 0.85)";
                        last.style.backgroundColor = "rgba(255, 243, 205, 0.55)";
                        setTimeout(() => {
                            last.style.boxShadow = previousBoxShadow;
                            last.style.backgroundColor = previousBackground;
                        }, 1300);
                    }
                </script>
                """,
                height=0,
        )


def _slugify_token(value: str, *, fallback: str = "item") -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return token or fallback


def _shorten_inline_text(text_value: str, *, max_chars: int) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _normalize_continuous_text_for_display(text_value: str) -> str:
    lines = str(text_value or "").splitlines()
    cleaned_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        line = line.replace("\u2014", ", ").replace("\u2013", ", ")
        line = re.sub(r"\s-\s", ", ", line)
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


def _parse_summary_sections(summary_text: str) -> list[tuple[str, str]]:
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


def _render_structured_summary(summary_text: str) -> None:
    parsed_sections = _parse_summary_sections(summary_text)
    if parsed_sections:
        for title, body in parsed_sections:
            cleaned_body = _normalize_continuous_text_for_display(body)
            if not cleaned_body:
                continue
            st.markdown(f"#### {title}")
            st.markdown(_format_long_prose_markdown(cleaned_body, max_sentences_per_paragraph=4))
        return

    st.markdown(_normalize_continuous_text_for_display(summary_text))


def _strip_socratic_follow_up(text_value: str) -> str:
    marker = "**Socratic next question:**"
    cleaned = str(text_value or "")
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def _extract_recent_assistant_context(ask_messages: list[dict], *, limit: int = 2) -> str:
    assistant_fragments: list[str] = []
    for message in reversed(ask_messages):
        if str(message.get("role") or "") != "assistant":
            continue
        cleaned = _strip_socratic_follow_up(str(message.get("content") or ""))
        if not cleaned:
            continue
        assistant_fragments.append(cleaned)
        if len(assistant_fragments) >= limit:
            break

    if not assistant_fragments:
        return ""
    return "\n\n".join(reversed(assistant_fragments))


def _build_quick_quiz_prompt(ask_messages: list[dict]) -> str:
    recent_context = _extract_recent_assistant_context(ask_messages, limit=2)
    compact_context = _shorten_inline_text(recent_context, max_chars=900)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Start quiz mode using grounded material and our recent conversation. "
        "Ask exactly one challenging question now and do not reveal the answer yet. "
        "Do not add extra headers, markdown bullets, or answer keys. "
        f"Recent assistant context: {compact_context}"
    ).strip()


def _build_quiz_grading_prompt(*, ask_messages: list[dict], user_answer: str) -> str:
    recent_context = _extract_recent_assistant_context(ask_messages, limit=2)
    compact_context = _shorten_inline_text(recent_context, max_chars=900)
    compact_answer = _shorten_inline_text(user_answer, max_chars=700)
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


def _build_quiz_continue_prompt(ask_messages: list[dict]) -> str:
    recent_context = _extract_recent_assistant_context(ask_messages, limit=2)
    compact_context = _shorten_inline_text(recent_context, max_chars=850)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Continue quiz mode. Ask exactly one new, more elaborate question connected to recent material. "
        "You may connect the question to another source when grounded evidence supports it. "
        "Do not reveal the answer yet and do not include grading in this message. "
        f"Recent assistant context: {compact_context}"
    ).strip()


def _build_quick_elaboration_prompt(ask_messages: list[dict]) -> str:
    recent_context = _extract_recent_assistant_context(ask_messages, limit=1)
    compact_context = _shorten_inline_text(recent_context, max_chars=820)
    if not compact_context:
        compact_context = "No prior assistant explanation in this session."

    return (
        "Elaborate further on your previous answer with deeper mechanism-level detail in plain language. "
        "Do not restate or paraphrase prior wording. Add net-new causal steps, assumptions, constraints, edge cases, and one practical diagnostic check. "
        "Do not use markdown headings or bullet lists. "
        f"Previous assistant answer context: {compact_context}"
    ).strip()


def _build_elaboration_chat_prompt(
    *,
    context_label: str,
    under_surface_text: str,
    key_terms: list[str],
    point_question: str = "",
    point_explanation: str = "",
    depth_level: str = "",
) -> str:
    compact_under_surface = _shorten_inline_text(under_surface_text, max_chars=480)
    compact_explanation = _shorten_inline_text(point_explanation, max_chars=260)
    terms_text = ", ".join(term.strip() for term in key_terms[:6] if term.strip())

    prompt_bits = [
        f"Elaborate further on: {context_label}.",
        f"Under-the-surface focus: {compact_under_surface or 'n/a'}.",
    ]
    if point_question:
        prompt_bits.append(f"Reflection question: {point_question}.")
    if compact_explanation:
        prompt_bits.append(f"Current explanation: {compact_explanation}.")
    if depth_level:
        prompt_bits.append(f"Target depth: {depth_level}.")
    if terms_text:
        prompt_bits.append(f"Key terms: {terms_text}.")

    prompt_bits.append(
        "Answer directly first, then expand the mechanism step by step in plain language, and end with one practical check I can apply. "
        "Do not repeat prior phrasing or examples; add net-new detail and avoid markdown headers or lists."
    )
    return " ".join(prompt_bits).strip()


def _compose_assistant_chat_message(
    answer_text: str,
    follow_up_question: str,
    *,
    include_follow_up: bool = True,
) -> str:
    normalized_answer = str(answer_text or "").strip()
    normalized_follow_up = str(follow_up_question or "").strip()

    if include_follow_up and normalized_follow_up:
        return f"{normalized_answer}\n\n**Socratic next question:** {normalized_follow_up}".strip()
    return normalized_answer or "I could not generate an answer this time. Please try rephrasing your question."


def _run_upload_process_and_generate(
    video_upload: dict[str, str | bytes] | None,
    video_url: str,
    document_uploads: list[dict[str, str | bytes]],
    stage_callback: Callable[[str], None] | None = None,
) -> None:
    def _set_stage(message: str) -> None:
        if stage_callback is not None:
            stage_callback(message)

    _set_stage("Stage 1/4: Uploading files...")

    files: list[tuple[str, tuple[str, bytes, str]]] = []
    data: dict[str, str] = {}

    if video_upload is not None:
        files.append(
            (
                "video",
                (
                    str(video_upload["name"]),
                    video_upload["data"],
                    str(video_upload["mime_type"]),
                ),
            )
        )
    elif video_url.strip():
        data["video_url"] = video_url.strip()

    for document_upload in document_uploads:
        files.append(
            (
                "documents",
                (
                    str(document_upload["name"]),
                    document_upload["data"],
                    str(document_upload["mime_type"]),
                ),
            )
        )

    upload_response = requests.post(
        f"{st.session_state.api_base_url}/upload",
        files=files if files else None,
        data=data if data else None,
        headers=_request_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if upload_response.status_code >= 400:
        raise RuntimeError(f"/upload failed: {_extract_error(upload_response)}")

    upload_payload = upload_response.json()
    video_payload = upload_payload.get("video") or {}
    video_source_id = int(video_payload.get("source_id", 0) or 0)
    normalized_video_source_id = video_source_id if video_source_id > 0 else None
    document_source_ids = [
        int(document_payload["source_id"])
        for document_payload in upload_payload.get("documents", [])
    ]

    process_payload = {
        "video_source_id": normalized_video_source_id,
        "document_source_ids": document_source_ids,
    }

    _set_stage("Stage 2/4: Processing sources and linking concepts...")
    _post_json("/process", process_payload)

    _set_stage("Stage 3/4: Generating tailored learning...")
    generation_payload = _post_json("/generate-tailored-learning", process_payload)

    _set_stage("Stage 4/4: Finalizing dashboard...")

    st.session_state.video_source_id = normalized_video_source_id
    st.session_state.document_source_ids = document_source_ids
    st.session_state.source_ids = (
        ([normalized_video_source_id] if normalized_video_source_id is not None else [])
        + document_source_ids
    )
    st.session_state.pipeline_ready = True
    st.session_state.interaction_session_id = str(uuid.uuid4())
    st.session_state.ask_result = None
    st.session_state.ask_messages = []
    st.session_state.ask_prefill = ""
    st.session_state.auto_submit_query = ""
    st.session_state.flash_latest_assistant = False
    st.session_state.generation_result = generation_payload
    st.session_state.wizard_step = 3
    st.session_state.dashboard_section = "Cross-Source Synthesis & Assessment"


def _resolve_source_label_map(video_payload: dict | None, document_payloads: list[dict]) -> dict[int, str]:
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


def _should_show_cross_source_section(
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


def _render_key_terms(key_terms: list[str]) -> None:
    if not key_terms:
        return
    formatted = ", ".join(f"**{term.strip()}**" for term in key_terms if term.strip())
    if formatted:
        st.markdown(f"Key terms: {formatted}")


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


def _format_long_prose_markdown(text_value: str, *, max_sentences_per_paragraph: int = 3) -> str:
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


def _text_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9']+", str(left or "").lower())
        if len(token) >= 4
    }
    right_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9']+", str(right or "").lower())
        if len(token) >= 4
    }
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(min(len(left_tokens), len(right_tokens)))


def _render_integrated_cross_source_analysis(
    *,
    layman_bridge: str,
    synthesis_text: str,
    comparative_analysis: str,
    emphasis_terms: list[str],
) -> None:
    cleaned_bridge = _normalize_continuous_text_for_display(layman_bridge)
    cleaned_synthesis = _normalize_continuous_text_for_display(synthesis_text)
    cleaned_comparative = _normalize_continuous_text_for_display(comparative_analysis)

    if not any([cleaned_bridge, cleaned_synthesis, cleaned_comparative]):
        return

    constraint_parts = [part for part in [cleaned_synthesis, cleaned_comparative] if part]
    if cleaned_bridge and not constraint_parts:
        constraint_parts.append(cleaned_bridge)

    if constraint_parts:
        st.markdown("### 2) Constraint or Breakdown")
        merged_constraint_text = " ".join(constraint_parts).strip()
        st.markdown(
            _bold_keywords_in_text(
                _format_long_prose_markdown(merged_constraint_text, max_sentences_per_paragraph=4),
                emphasis_terms,
            )
        )


def _render_reflection_points(
    reflection_points: list[dict] | list[str],
    *,
    source_label: str,
    source_key: str,
    key_terms: list[str],
) -> None:
    if not reflection_points:
        st.write("No reflection points available.")
        return

    for index, point in enumerate(reflection_points, start=1):
        if isinstance(point, str):
            st.markdown(f"{index}. {point}")
            continue

        question = str(point.get("question", "")).strip()
        explanation = str(point.get("explanation", "")).strip()
        under_the_hood = str(point.get("under_the_hood", "")).strip()
        depth_level = str(point.get("depth_level", "")).strip()

        st.markdown(f"**{index}. {question}**")
        if depth_level:
            st.caption(f"Depth: {depth_level}")
        if explanation:
            st.markdown(_format_long_prose_markdown(explanation))
        if under_the_hood:
            with st.expander(f"Why this works under the surface (point {index})"):
                cleaned_under_the_hood = _normalize_continuous_text_for_display(under_the_hood)
                st.markdown(_format_long_prose_markdown(cleaned_under_the_hood, max_sentences_per_paragraph=4))
                if st.button(
                    "Elaborate further in chat",
                    key=f"elaborate_reflection_{source_key}_{index}",
                    use_container_width=False,
                ):
                    _queue_chat_navigation(
                        _build_elaboration_chat_prompt(
                            context_label=f"{source_label} | reflection point {index}",
                            under_surface_text=under_the_hood,
                            key_terms=key_terms,
                            point_question=question,
                            point_explanation=explanation,
                            depth_level=depth_level,
                        ),
                        auto_submit=True,
                    )
                    st.rerun()


def _render_source_learning_section(section_payload: dict) -> None:
    generated_title = str(section_payload.get("generated_title", "Untitled")).strip() or "Untitled"
    source_type = str(section_payload.get("source_type") or "source").strip().lower()
    source_prefix = "Video" if source_type == "video" else "Document" if source_type == "document" else "Source"
    source_label = f"{source_prefix}: {generated_title}"
    source_key = _slugify_token(f"{source_prefix}_{section_payload.get('source_id', '0')}_{generated_title}")
    source_name = str(section_payload.get("source_name") or "").strip()

    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _clean_block(raw_text: str, title_text: str, section_kind: str) -> str:
        text_value = (raw_text or "").strip()
        if not text_value:
            return ""

        title_norm = _normalize(title_text)
        lines = text_value.splitlines()
        cleaned_lines: list[str] = []

        for line in lines:
            candidate = line.strip()
            if not candidate:
                cleaned_lines.append("")
                continue

            candidate_norm = _normalize(candidate)
            if not cleaned_lines:
                if title_norm and candidate_norm == title_norm:
                    continue
                if section_kind == "summary" and (
                    candidate_norm == "summary"
                    or candidate_norm.startswith("summary of")
                    or candidate_norm in {"video summary", "document summary"}
                ):
                    continue
                if section_kind == "deep_dive" and (
                    candidate_norm == "deep dive"
                    or candidate_norm.startswith("deep dive")
                ):
                    continue

            cleaned_lines.append(candidate)

        cleaned_text = "\n".join(cleaned_lines).strip()
        while cleaned_text.lower().startswith("summary\n"):
            cleaned_text = cleaned_text.split("\n", 1)[1].strip()

        return cleaned_text

    st.subheader(source_label)
    if source_name and _normalize(source_name) != _normalize(generated_title):
        st.caption(source_name)
    summary_text = _clean_block(
        str(section_payload.get("summary_text", "")),
        generated_title,
        "summary",
    )
    _render_structured_summary(summary_text)

    key_terms = section_payload.get("key_terms", []) or []
    normalized_terms = [str(term) for term in key_terms]
    _render_key_terms(normalized_terms)

    deep_dive_text = _clean_block(
        str(section_payload.get("deep_dive_text", "")),
        generated_title,
        "deep_dive",
    )
    if deep_dive_text:
        deep_dive_text = _normalize_continuous_text_for_display(deep_dive_text)
        st.markdown("### Deep Dive")
        st.markdown(
            _bold_keywords_in_text(
                _format_long_prose_markdown(deep_dive_text),
                normalized_terms,
            )
        )

    under_surface_text = str(section_payload.get("under_surface_explainer") or "").strip()
    if under_surface_text:
        with st.expander("Why this works under the surface"):
            normalized_under_surface = _normalize_continuous_text_for_display(under_surface_text)
            formatted_under_surface = _format_long_prose_markdown(
                normalized_under_surface,
                max_sentences_per_paragraph=4,
            )
            st.markdown(_bold_keywords_in_text(formatted_under_surface, normalized_terms))

            diagnostic_checklist = section_payload.get("diagnostic_checklist", []) or []
            if diagnostic_checklist:
                st.markdown("**Diagnostic checklist**")
                for item in diagnostic_checklist:
                    st.markdown(f"- {item}")

            if st.button(
                "Elaborate further in chat",
                key=f"elaborate_under_surface_{source_key}",
                use_container_width=False,
            ):
                _queue_chat_navigation(
                    _build_elaboration_chat_prompt(
                        context_label=source_label,
                        under_surface_text=normalized_under_surface,
                        key_terms=normalized_terms,
                    ),
                    auto_submit=True,
                )
                st.rerun()

    reflection_points = section_payload.get("reflection_points", []) or []
    st.markdown("### Socratic Reflection Points")
    _render_reflection_points(
        reflection_points,
        source_label=source_label,
        source_key=source_key,
        key_terms=normalized_terms,
    )


def _render_ask_result() -> None:
    if not st.session_state.ask_result:
        return

    st.subheader("Answer")
    st.markdown(_normalize_continuous_text_for_display(str(st.session_state.ask_result.get("answer", "")).strip()))

    follow_up_question = (st.session_state.ask_result.get("follow_up_question") or "").strip()
    if follow_up_question:
        st.subheader("Follow-up Question")
        st.markdown(_normalize_continuous_text_for_display(follow_up_question))


def _collect_emphasis_terms(
    attributed_sentences: list[dict],
    *,
    limit: int = 12,
) -> list[str]:
    keyword_pool: list[str] = []
    seen_keywords: set[str] = set()

    for item in attributed_sentences:
        if not isinstance(item, dict):
            continue

        emphasis_terms = [
            str(term).strip()
            for term in (item.get("emphasis_terms", []) or [])
            if str(term).strip()
        ]
        for term in emphasis_terms:
            lowered = term.lower()
            if lowered in seen_keywords:
                continue
            seen_keywords.add(lowered)
            keyword_pool.append(term)
            if len(keyword_pool) >= limit:
                return keyword_pool

    return keyword_pool


def _bold_keywords_in_text(text_value: str, keywords: list[str]) -> str:
    highlighted = text_value
    for keyword in sorted((value for value in keywords if len(value) >= 3), key=len, reverse=True):
        pattern = re.compile(rf"(?i)(?<![A-Za-z0-9])({re.escape(keyword)})(?![A-Za-z0-9])")
        highlighted = pattern.sub(lambda match: f"**{match.group(0)}**", highlighted)
    return highlighted


def _build_first_principles_evidence_summary(
    evidence_items: list[tuple[str, str, list[str]]],
) -> str:
    if not evidence_items:
        return ""

    quote_fragments: list[str] = []
    for _, quote_text, _ in evidence_items[:3]:
        fragment = _normalize_continuous_text_for_display(quote_text)
        if not fragment:
            continue
        quote_fragments.append(fragment)

    if not quote_fragments:
        return ""

    if len(quote_fragments) == 1:
        body = quote_fragments[0]
    elif len(quote_fragments) == 2:
        body = f"{quote_fragments[0]} This connects to: {quote_fragments[1]}"
    else:
        body = (
            f"{quote_fragments[0]} This mechanism continues as: {quote_fragments[1]} "
            f"A practical implication is: {quote_fragments[2]}"
        )

    return _format_long_prose_markdown(_normalize_continuous_text_for_display(body), max_sentences_per_paragraph=4)


def _render_grouped_evidence(
    attributed_sentences: list[dict],
    source_label_map: dict[int, str],
) -> None:
    evidence_items: list[tuple[str, str, list[str]]] = []
    seen_quotes: set[str] = set()

    for item in attributed_sentences:
        if not isinstance(item, dict):
            continue

        source_id = int(item.get("source_id", 0) or 0)
        if source_id <= 0 or source_id not in source_label_map:
            continue

        quote_text = " ".join(str(item.get("text", "")).split()).strip()
        if not quote_text:
            continue

        quote_text = re.sub(r"\[?\s*source\s*#?\d+\s*\]?:?\s*", "", quote_text, flags=re.IGNORECASE)
        quote_text = quote_text.strip()
        if not quote_text:
            continue

        normalized = quote_text.lower()
        if normalized in seen_quotes:
            continue
        seen_quotes.add(normalized)

        if len(quote_text) > 240:
            quote_text = quote_text[:237].rstrip() + "..."

        emphasis_terms = [
            str(term).strip()
            for term in (item.get("emphasis_terms", []) or [])
            if str(term).strip()
        ]
        source_label = source_label_map.get(source_id, f"Source {source_id}")
        evidence_items.append((source_label, quote_text, emphasis_terms))

    if not evidence_items:
        return

    with st.expander("Show grounded mechanism evidence", expanded=False):
        keyword_terms: list[str] = []
        seen_terms: set[str] = set()
        for _, _, terms in evidence_items:
            for term in terms:
                normalized = term.lower().strip()
                if not normalized or normalized in seen_terms:
                    continue
                seen_terms.add(normalized)
                keyword_terms.append(term.strip())
                if len(keyword_terms) >= 8:
                    break
            if len(keyword_terms) >= 8:
                break

        if keyword_terms:
            joined_keywords = ", ".join(f"**{term}**" for term in keyword_terms)
            st.markdown(f"First-principles keywords: {joined_keywords}")

        summary = _build_first_principles_evidence_summary(evidence_items)
        if summary:
            st.markdown(summary)


def _submit_chat_query(
    ask_query: str,
    *,
    suppress_follow_up: bool = False,
    display_query: str | None = None,
) -> None:
    query_text = ask_query.strip()
    if not query_text:
        return

    display_text = str(display_query or query_text).strip() or query_text

    st.session_state.ask_messages.append({"role": "user", "content": display_text})
    try:
        with st.spinner("Thinking..."):
            st.session_state.ask_result = _post_json(
                "/ask",
                {
                    "session_id": st.session_state.interaction_session_id,
                    "source_ids": st.session_state.source_ids,
                    "query": query_text,
                    "top_k": 8,
                },
            )

        answer_text = str(st.session_state.ask_result.get("answer", "")).strip()
        follow_up_question = str(
            st.session_state.ask_result.get("follow_up_question") or ""
        ).strip()
        assistant_text = _compose_assistant_chat_message(
            answer_text,
            follow_up_question,
            include_follow_up=not suppress_follow_up,
        )
        st.session_state.ask_messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.flash_latest_assistant = True
    except Exception as exc:
        error_text = str(exc)
        st.session_state.ask_messages.append(
            {
                "role": "assistant",
                "content": (
                    f"I ran into an error while answering that request: {error_text}. "
                    "Please try asking again with a more specific wording."
                ),
            }
        )
        st.session_state.flash_latest_assistant = True

    st.rerun()


def _build_intersection_chat_prefill(
    *,
    title: str,
    why_it_matters: str,
    emphasis_terms: list[str],
    attributed_sentences: list[dict],
) -> str:
    quote_fragments: list[str] = []
    seen: set[str] = set()
    for item in attributed_sentences:
        quote_text = " ".join(str(item.get("text", "")).split()).strip()
        if not quote_text:
            continue
        quote_text = re.sub(r"\[?\s*source\s*#?\d+\s*\]?:?\s*", "", quote_text, flags=re.IGNORECASE).strip()
        if not quote_text:
            continue
        normalized = quote_text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        quote_fragments.append(quote_text)
        if len(quote_fragments) >= 2:
            break

    keyword_text = ", ".join(emphasis_terms[:6])
    quote_text = " | ".join(quote_fragments)
    return (
        f"Elaborate further on intersection '{title}' by focusing on how it works under the surface: {why_it_matters}. "
        f"Key concepts: {keyword_text or 'none listed'}. "
        f"Evidence snippets: {quote_text or 'none provided'}. "
        "Please answer directly first, expand mechanism-level reasoning from first principles in plain language, and end with one concrete application check. "
        "Do not repeat prior phrasing and do not use markdown headings or list formatting."
    ).strip()

def _render_ask_tab(has_user_id: bool) -> None:
    if not st.session_state.pipeline_ready:
        st.warning("Generate tailored content before asking questions.")
        _render_ask_result()
        return

    st.markdown("### Socratic Chatbox")
    st.caption("Ask follow-up questions to deepen understanding across video, documents, and generated insights.")

    if st.session_state.ask_prefill.strip():
        st.session_state.socratic_chat_text = st.session_state.ask_prefill.strip()
        st.session_state.ask_prefill = ""

    auto_query = st.session_state.auto_submit_query.strip()
    if auto_query:
        st.session_state.auto_submit_query = ""
        _submit_chat_query(auto_query)
        return

    if not has_user_id:
        st.warning("Set a user ID in the sidebar to use the chatbox.")
        return

    for message in st.session_state.ask_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.flash_latest_assistant:
        _render_chat_scroll_and_flash()
        st.session_state.flash_latest_assistant = False

    if st.session_state.quiz_mode_active:
        if st.session_state.quiz_awaiting_answer:
            st.info("Quiz mode is active. Answer the current quiz question below to receive grading.")
        else:
            st.success(
                f"Quiz feedback complete for {int(st.session_state.quiz_turn_count)} question(s). Continue for another question or exit quiz mode."
            )
            col_continue_quiz, col_exit_quiz = st.columns(2)
            with col_continue_quiz:
                if st.button("Continue quiz", key="quick_action_quiz_continue"):
                    st.session_state.quiz_awaiting_answer = True
                    _submit_chat_query(
                        _build_quiz_continue_prompt(st.session_state.ask_messages),
                        suppress_follow_up=True,
                        display_query="Continue quiz.",
                    )
            with col_exit_quiz:
                if st.button("Exit quiz mode", key="quick_action_quiz_exit"):
                    st.session_state.quiz_mode_active = False
                    st.session_state.quiz_awaiting_answer = False
                    st.session_state.quiz_turn_count = 0
                    st.rerun()

    if (
        st.session_state.ask_messages
        and st.session_state.ask_messages[-1]["role"] == "assistant"
        and not st.session_state.quiz_mode_active
    ):
        col_elaborate, col_quiz = st.columns(2)
        with col_elaborate:
            if st.button("Elaborate further", key="quick_action_elaborate"):
                _submit_chat_query(_build_quick_elaboration_prompt(st.session_state.ask_messages))
        with col_quiz:
            if st.button("Quiz me on this", key="quick_action_quiz"):
                st.session_state.quiz_mode_active = True
                st.session_state.quiz_awaiting_answer = True
                st.session_state.quiz_turn_count = 0
                quick_quiz_prompt = _build_quick_quiz_prompt(st.session_state.ask_messages)
                _submit_chat_query(
                    quick_quiz_prompt,
                    suppress_follow_up=True,
                    display_query="Quiz me on this.",
                )

    st.caption("Type your question below, then click Send.")
    is_follow_up = bool(st.session_state.ask_messages)
    input_label = "Ask follow-up" if is_follow_up else "Ask your first question"
    input_placeholder = (
        "Ask follow-up..."
        if is_follow_up
        else "Example: Explain the strongest intersection and how I could apply it in a real project."
    )
    input_height = 84 if is_follow_up else 130

    with st.form(key="socratic_chat_form", clear_on_submit=True):
        ask_query = st.text_area(
            input_label,
            height=input_height,
            key="socratic_chat_text",
            placeholder=input_placeholder,
        )
        submitted = st.form_submit_button("Send" if is_follow_up else "Ask")

    if not submitted:
        return

    ask_query = ask_query.strip()
    if not ask_query:
        st.warning("Enter a question before sending.")
        return

    if st.session_state.quiz_mode_active and st.session_state.quiz_awaiting_answer:
        grading_prompt = _build_quiz_grading_prompt(
            ask_messages=st.session_state.ask_messages,
            user_answer=ask_query,
        )
        st.session_state.quiz_awaiting_answer = False
        st.session_state.quiz_turn_count = int(st.session_state.quiz_turn_count) + 1
        _submit_chat_query(
            grading_prompt,
            suppress_follow_up=True,
            display_query=ask_query,
        )
        return

    if st.session_state.quiz_mode_active and not st.session_state.quiz_awaiting_answer:
        # Any non-quiz submission exits quiz mode and returns to normal chat flow.
        st.session_state.quiz_mode_active = False
        st.session_state.quiz_awaiting_answer = False
        st.session_state.quiz_turn_count = 0

    _submit_chat_query(ask_query)


def _render_generation_tabs(has_user_id: bool) -> None:
    generation_result = st.session_state.generation_result or {}
    if not generation_result:
        st.info("Generate tailored content to populate tabs.")
        return

    video_payload = generation_result.get("video") or None
    document_payloads = generation_result.get("documents", []) or []
    insights_payload = generation_result.get("insights", {})
    quiz_payload = generation_result.get("quiz", {})
    source_label_map = _resolve_source_label_map(video_payload, document_payloads)

    section_options: list[str] = []
    if video_payload:
        section_options.append("Video")
    if document_payloads:
        section_options.append("Documents")

    if _should_show_cross_source_section(
        generation_result=generation_result,
        insights_payload=insights_payload,
        quiz_payload=quiz_payload,
    ):
        section_options.append("Cross-Source Synthesis & Assessment")
    section_options.append("Socratic Chatbox")

    pending_section = str(st.session_state.pending_dashboard_section or "").strip()
    if pending_section in section_options:
        st.session_state.dashboard_section = pending_section
    st.session_state.pending_dashboard_section = ""
    if st.session_state.dashboard_section not in section_options:
        st.session_state.dashboard_section = section_options[0]

    st.radio(
        "Dashboard Section",
        options=section_options,
        key="dashboard_section",
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_section = st.session_state.dashboard_section

    if selected_section == "Video" and video_payload:
        _render_source_learning_section(video_payload)

    if selected_section == "Documents":
        if not document_payloads:
            st.info("No document sections available.")
        else:
            document_labels = []
            for index, section_payload in enumerate(document_payloads, start=1):
                title = str(section_payload.get("generated_title", f"Document {index}")).strip()
                title = f"Document: {title}"
                document_labels.append(title[:40] if len(title) > 40 else title)

            document_tabs = st.tabs(document_labels)
            for section_payload, document_tab in zip(document_payloads, document_tabs):
                with document_tab:
                    _render_source_learning_section(section_payload)

    if selected_section == "Cross-Source Synthesis & Assessment":
        st.markdown("### 1) Mapping")

        intersections = insights_payload.get("intersections", []) or []
        all_cross_attributed_sentences: list[dict] = []
        if intersections:
            for index, intersection in enumerate(intersections, start=1):
                if not isinstance(intersection, dict):
                    continue

                title = str(intersection.get("intersection_title", f"Intersection {index}")).strip()
                why_it_matters = str(intersection.get("why_it_matters", "")).strip()
                integrated_explanation = str(intersection.get("integrated_explanation", "")).strip()

                with st.container(border=True):
                    st.markdown(f"#### {title or f'Intersection {index}'}")

                    attributed_sentences = [
                        item
                        for item in (intersection.get("attributed_sentences", []) or [])
                        if isinstance(item, dict)
                        and int(item.get("source_id", 0) or 0) in source_label_map
                    ]
                    all_cross_attributed_sentences.extend(attributed_sentences)
                    emphasis_terms = _collect_emphasis_terms(attributed_sentences)

                    if why_it_matters:
                        st.markdown(
                            f"**Why it matters:** {_bold_keywords_in_text(why_it_matters, emphasis_terms)}"
                        )
                    if integrated_explanation:
                        cleaned_integrated = _normalize_continuous_text_for_display(integrated_explanation)
                        st.markdown(
                            _bold_keywords_in_text(
                                _format_long_prose_markdown(cleaned_integrated, max_sentences_per_paragraph=4),
                                emphasis_terms,
                            )
                        )

                    if emphasis_terms:
                        concept_list = ", ".join(f"**{term}**" for term in emphasis_terms)
                        st.caption(f"Key concepts: {concept_list}")

                    _render_grouped_evidence(
                        attributed_sentences,
                        source_label_map=source_label_map,
                    )

                    ask_about_intersection = st.button(
                        "Elaborate further",
                        key=f"ask_intersection_{index}",
                        use_container_width=False,
                    )
                    if ask_about_intersection:
                        _queue_chat_navigation(_build_intersection_chat_prefill(
                            title=title or f"Intersection {index}",
                            why_it_matters=why_it_matters,
                            emphasis_terms=emphasis_terms,
                            attributed_sentences=attributed_sentences,
                        ), auto_submit=True)
                        st.rerun()

                    inferred_extension = str(intersection.get("inferred_extension") or "").strip()
                    inference_label = str(intersection.get("inference_label") or "").strip()
                    if inferred_extension and inference_label == "inferred_extension":
                        with st.expander("Inferred extension (conceptual deepening)"):
                            st.markdown(inferred_extension)
        else:
            parallels = insights_payload.get("parallels", []) or []
            if parallels:
                with st.container(border=True):
                    st.markdown("#### Core Cross-Source Intersection")
                    normalized_parallels = [
                        item
                        for item in parallels
                        if isinstance(item, dict)
                        and int(item.get("source_id", 0) or 0) in source_label_map
                    ]
                    all_cross_attributed_sentences.extend(normalized_parallels)
                    emphasis_terms = _collect_emphasis_terms(normalized_parallels)
                    if emphasis_terms:
                        concept_list = ", ".join(f"**{term}**" for term in emphasis_terms)
                        st.caption(f"Key concepts: {concept_list}")
                    _render_grouped_evidence(
                        normalized_parallels,
                        source_label_map=source_label_map,
                    )

                    ask_about_core = st.button(
                        "Elaborate further",
                        key="ask_core_intersection",
                        use_container_width=False,
                    )
                    if ask_about_core:
                        _queue_chat_navigation(_build_intersection_chat_prefill(
                            title="Core Cross-Source Intersection",
                            why_it_matters="Help me understand the strongest overlap across my sources.",
                            emphasis_terms=emphasis_terms,
                            attributed_sentences=normalized_parallels,
                        ), auto_submit=True)
                        st.rerun()
            else:
                st.write("No intersections available.")

        cross_emphasis_terms = _collect_emphasis_terms(all_cross_attributed_sentences)

        layman_bridge = str(insights_payload.get("layman_bridge", "")).strip()
        synthesis_text = str(insights_payload.get("synthesis_text", "")).strip()
        comparative_analysis = str(insights_payload.get("comparative_analysis") or "").strip()
        _render_integrated_cross_source_analysis(
            layman_bridge=layman_bridge,
            synthesis_text=synthesis_text,
            comparative_analysis=comparative_analysis,
            emphasis_terms=cross_emphasis_terms,
        )

        application_scenarios = insights_payload.get("application_scenarios", []) or []
        if application_scenarios:
            st.markdown("### 3) Transfer and Adaptation")
            for index, scenario in enumerate(application_scenarios, start=1):
                if not isinstance(scenario, dict):
                    continue
                scenario_title = str(scenario.get("scenario_title") or f"Scenario {index}").strip()
                scenario_prompt = str(scenario.get("scenario_prompt") or "").strip()
                transfer_steps = scenario.get("transfer_steps") or []
                common_pitfall = str(scenario.get("common_pitfall") or "").strip()

                with st.container(border=True):
                    st.markdown(f"#### {scenario_title}")
                    if scenario_prompt:
                        st.markdown(f"**Prompt:** {scenario_prompt}")
                    if transfer_steps:
                        st.markdown("**Transfer steps**")
                        for step_index, step_text in enumerate(transfer_steps, start=1):
                            st.markdown(f"{step_index}. {step_text}")
                    if common_pitfall:
                        st.warning(f"Common pitfall: {common_pitfall}")

        st.markdown("### 4) Decision and Implication")
        questions = quiz_payload.get("questions", []) or []
        if not questions:
            st.write("No quiz questions available.")
        else:
            for index, question_payload in enumerate(questions, start=1):
                difficulty_level = str(question_payload.get("difficulty_level", "")).strip()
                question_type = str(question_payload.get("question_type", "")).strip()
                st.markdown(f"**Q{index}. {question_payload.get('question', '')}**")
                if difficulty_level or question_type:
                    st.caption(f"Level: {difficulty_level or 'n/a'} | Type: {question_type or 'n/a'}")

                options = question_payload.get("options", []) or []
                for option_index, option_text in enumerate(options, start=1):
                    st.write(f"{option_index}. {option_text}")

                answer_index = int(question_payload.get("answer_index", 0))
                explanation = str(question_payload.get("explanation", "")).strip()
                under_the_hood = str(question_payload.get("under_the_hood", "")).strip()
                source_evidence = question_payload.get("source_evidence", []) or []
                with st.expander(f"Show answer for Q{index}"):
                    if options and 0 <= answer_index < len(options):
                        st.write(f"Correct answer: {options[answer_index]}")
                    if explanation:
                        st.markdown("**Why this is correct**")
                        st.markdown(explanation)
                    if under_the_hood:
                        st.markdown("**Why this works under the surface**")
                        st.markdown(under_the_hood)
                    if source_evidence:
                        st.markdown("**Source evidence**")
                        for evidence in source_evidence:
                            st.markdown(f"- {evidence}")

        study_advice = str(quiz_payload.get("study_advice", "")).strip()
        if study_advice:
            st.markdown("### Study Advice")
            st.markdown(study_advice)

    if selected_section == "Socratic Chatbox":
        _render_ask_tab(has_user_id)


def main() -> None:
    _init_state()

    st.title("Study Assistant")

    st.sidebar.header("Configuration")
    st.session_state.api_base_url = st.sidebar.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
    ).rstrip("/")
    st.session_state.user_id = st.sidebar.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Every API request includes this value in the X-User-ID header.",
    ).strip()

    show_sidebar_reset = st.session_state.wizard_step > 1 or st.session_state.pipeline_ready
    if show_sidebar_reset:
        st.sidebar.markdown("---")
        if st.sidebar.button("Reset build setup", use_container_width=True):
            _reset_builder()
            st.rerun()
        st.sidebar.caption("Start over with a new video/document set.")

    has_user_id = bool(st.session_state.user_id)
    if not st.session_state.user_id:
        st.warning("User ID is required. Set it in the sidebar before running requests.")

    st.header("Build Tailored Socratic Learning")
    st.caption(f"Step {st.session_state.wizard_step} of 3")

    if st.session_state.wizard_step == 1:
        st.markdown("### Step 1: Add one optional video source")
        st.caption("Choose exactly one video source: upload OR YouTube link. You can skip video.")

        selected_mode = st.radio(
            "Video source",
            options=["Upload video", "Paste YouTube link"],
            key="wizard_video_source_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

        if st.session_state.wizard_last_video_source_mode != selected_mode:
            st.session_state.wizard_replace_video = False
            if selected_mode == "Upload video":
                st.session_state.wizard_video_url = ""
            else:
                st.session_state.wizard_video_upload = None
            st.session_state.wizard_last_video_source_mode = selected_mode

        if selected_mode == "Upload video":
            uploaded_video = None
            if st.session_state.wizard_video_upload is not None and not st.session_state.wizard_replace_video:
                st.success(f"Video selected: {st.session_state.wizard_video_upload['name']}")
                if st.button("Replace video", key="replace_video_button"):
                    st.session_state.wizard_replace_video = True
                    st.rerun()
            else:
                uploaded_video = st.file_uploader(
                    "Upload one video",
                    accept_multiple_files=False,
                    key="wizard_video_uploader",
                    help="Supported formats: mp4, mov, m4v, avi, mkv, webm.",
                )
                if uploaded_video is not None and not _is_supported_video_name(uploaded_video.name):
                    st.warning("Unsupported video format. Use: mp4, mov, m4v, avi, mkv, or webm.")
                    uploaded_video = None
                if uploaded_video is not None:
                    st.session_state.wizard_video_upload = _pack_uploaded_file(uploaded_video)
                    st.session_state.wizard_video_url = ""
                    st.session_state.wizard_replace_video = False
                    st.success(f"Video selected: {uploaded_video.name}")
        else:
            if st.session_state.wizard_video_url.strip() and not st.session_state.wizard_replace_video:
                st.success(f"YouTube link saved: {st.session_state.wizard_video_url.strip()}")
                if st.button("Replace YouTube link", key="replace_video_link_button"):
                    st.session_state.wizard_replace_video = True
                    st.rerun()
            else:
                if (
                    not st.session_state.wizard_video_url_input.strip()
                    and st.session_state.wizard_video_url.strip()
                ):
                    st.session_state.wizard_video_url_input = st.session_state.wizard_video_url.strip()
                youtube_url_candidate = st.text_input(
                    "Paste one YouTube URL",
                    key="wizard_video_url_input",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                if st.button("Save link", key="save_video_link"):
                    normalized_youtube_url = _normalize_youtube_url(youtube_url_candidate)
                    if not normalized_youtube_url:
                        st.error("Enter a valid single-video YouTube URL.")
                    else:
                        st.session_state.wizard_video_url = normalized_youtube_url
                        st.session_state.wizard_video_upload = None
                        st.session_state.wizard_replace_video = False
                        st.rerun()

        col_skip, col_next = st.columns(2)
        with col_skip:
            if st.button("Skip video", disabled=(not has_user_id)):
                st.session_state.wizard_video_upload = None
                st.session_state.wizard_video_url = ""
                st.session_state.wizard_step = 2
                st.rerun()
        with col_next:
            if st.button("Next", disabled=(not has_user_id)):
                normalized_saved_youtube_url = _normalize_youtube_url(st.session_state.wizard_video_url)
                normalized_input_youtube_url = _normalize_youtube_url(
                    st.session_state.wizard_video_url_input
                )
                effective_youtube_url = normalized_saved_youtube_url or normalized_input_youtube_url

                if effective_youtube_url:
                    st.session_state.wizard_video_url = effective_youtube_url

                has_video_source = bool(st.session_state.wizard_video_upload) or bool(effective_youtube_url)
                if not has_video_source:
                    st.error("Add a valid video upload or YouTube link, or use Skip video.")
                else:
                    if effective_youtube_url:
                        st.session_state.wizard_video_upload = None
                    st.session_state.wizard_step = 2
                    st.rerun()

    elif st.session_state.wizard_step == 2:
        st.markdown("### Step 2: Add optional documents")
        st.caption("Upload documents for cross-source synthesis, or skip this step.")
        has_video_source = bool(st.session_state.wizard_video_upload) or bool(
            _normalize_youtube_url(st.session_state.wizard_video_url)
        )

        document_files: list = []
        if st.session_state.wizard_document_uploads and not st.session_state.wizard_replace_documents:
            existing_names = [item["name"] for item in st.session_state.wizard_document_uploads]
            st.success(f"Documents selected: {', '.join(existing_names)}")
            if st.button("Replace documents", key="replace_documents_button"):
                st.session_state.wizard_replace_documents = True
                st.rerun()
        else:
            uploaded_files = st.file_uploader(
                "Upload one or more documents",
                accept_multiple_files=True,
                key="wizard_document_uploader",
                help="Supported formats: pdf, txt, md.",
            )
            document_files = list(uploaded_files or [])
            if document_files:
                valid_docs, invalid_docs = _are_supported_document_names([file.name for file in document_files])
                if not valid_docs:
                    st.warning(
                        "Unsupported document format(s): "
                        + ", ".join(invalid_docs)
                        + ". Use: pdf, txt, md."
                    )
                    document_files = []

            if st.button("Save documents", key="save_documents_selection"):
                if not document_files:
                    st.error("Select at least one document before saving.")
                else:
                    st.session_state.wizard_document_uploads = [
                        _pack_uploaded_file(file) for file in document_files
                    ]
                    st.session_state.wizard_replace_documents = False
                    st.success("Saved document selection.")

        col_back, col_skip, col_next = st.columns(3)
        with col_back:
            if st.button("Back"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col_skip:
            if st.button("Skip documents", disabled=(not has_user_id or not has_video_source)):
                st.session_state.wizard_document_uploads = []
                st.session_state.wizard_replace_documents = False
                st.session_state.wizard_step = 3
                st.rerun()
            if not has_video_source:
                st.caption("Documents cannot be skipped when no video source is selected.")
        with col_next:
            if st.button("Next", disabled=(not has_user_id)):
                if not st.session_state.wizard_document_uploads:
                    st.error("Save documents first, or use Skip documents.")
                else:
                    st.session_state.wizard_step = 3
                    st.rerun()

    else:
        st.markdown("### Step 3: Generate tailored learning")
        st.caption("Estimated generation time: about 2 to 5 minutes.")

        selected_video = st.session_state.wizard_video_upload
        selected_video_url = st.session_state.wizard_video_url.strip()
        selected_documents = st.session_state.wizard_document_uploads
        has_video_source = selected_video is not None or bool(selected_video_url)
        has_document_source = bool(selected_documents)
        has_any_source = has_video_source or has_document_source

        summary_bits: list[str] = []
        if selected_video is not None:
            summary_bits.append("Video source selected")
        elif selected_video_url:
            summary_bits.append("YouTube source selected")

        if selected_documents:
            summary_bits.append(f"{len(selected_documents)} document(s) selected")

        if summary_bits:
            st.write(" | ".join(summary_bits))

        if has_any_source:
            with st.expander("Review selected sources", expanded=False):
                if selected_video is not None:
                    st.markdown(f"- Video: {selected_video['name']}")
                elif selected_video_url:
                    st.markdown(f"- YouTube: {selected_video_url}")

                for upload in selected_documents:
                    st.markdown(f"- Document: {upload['name']}")

        if not has_any_source:
            st.warning("Add at least one source before generating tailored learning.")

        generate_clicked = st.button(
            "Generate tailored Socratic learning",
            type="primary",
            use_container_width=True,
            disabled=(
                not has_user_id
                or not has_any_source
            ),
        )

        if generate_clicked:
            stage_placeholder = st.empty()

            def _update_stage(message: str) -> None:
                stage_placeholder.info(message)

            with st.spinner("Working..."):
                try:
                    _run_upload_process_and_generate(
                        video_upload=st.session_state.wizard_video_upload,
                        video_url=st.session_state.wizard_video_url,
                        document_uploads=st.session_state.wizard_document_uploads,
                        stage_callback=_update_stage,
                    )
                except Exception as exc:
                    stage_placeholder.empty()
                    st.error(str(exc))
                else:
                    stage_placeholder.success("Tailored Socratic learning generated successfully.")

    if st.session_state.pipeline_ready:
        st.header("Learning Dashboard")
        _render_generation_tabs(has_user_id)


if __name__ == "__main__":
    main()
