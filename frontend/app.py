from __future__ import annotations

import uuid
from collections.abc import Callable

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 300
DOCUMENT_COLORS = ["green", "blue", "orange", "violet", "gray"]
SOURCE_COLOR_DOTS = {
    "red": "🔴",
    "green": "🟢",
    "blue": "🔵",
    "orange": "🟠",
    "violet": "🟣",
    "gray": "⚪",
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


def _request_headers() -> dict[str, str]:
    return {"X-User-ID": str(st.session_state.user_id).strip()}


def _extract_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
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
    st.session_state.generation_result = None
    st.session_state.interaction_session_id = str(uuid.uuid4())
    st.session_state.wizard_step = 1
    st.session_state.wizard_video_upload = None
    st.session_state.wizard_document_uploads = []


def _run_upload_process_and_generate(
    video_upload: dict[str, str | bytes],
    document_uploads: list[dict[str, str | bytes]],
    stage_callback: Callable[[str], None] | None = None,
) -> None:
    def _set_stage(message: str) -> None:
        if stage_callback is not None:
            stage_callback(message)

    _set_stage("Stage 1/4: Uploading files...")

    files = [
        (
            "video",
            (
                str(video_upload["name"]),
                video_upload["data"],
                str(video_upload["mime_type"]),
            ),
        )
    ]

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
        files=files,
        headers=_request_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if upload_response.status_code >= 400:
        raise RuntimeError(f"/upload failed: {_extract_error(upload_response)}")

    upload_payload = upload_response.json()
    video_source_id = int(upload_payload["video"]["source_id"])
    document_source_ids = [
        int(document_payload["source_id"])
        for document_payload in upload_payload.get("documents", [])
    ]

    process_payload = {
        "video_source_id": video_source_id,
        "document_source_ids": document_source_ids,
    }

    _set_stage("Stage 2/4: Processing sources and linking concepts...")
    _post_json("/process", process_payload)

    _set_stage("Stage 3/4: Generating tailored learning...")
    generation_payload = _post_json("/generate-tailored-learning", process_payload)

    _set_stage("Stage 4/4: Finalizing dashboard...")

    st.session_state.video_source_id = video_source_id
    st.session_state.document_source_ids = document_source_ids
    st.session_state.source_ids = [video_source_id, *document_source_ids]
    st.session_state.pipeline_ready = True
    st.session_state.interaction_session_id = str(uuid.uuid4())
    st.session_state.ask_result = None
    st.session_state.generation_result = generation_payload
    st.session_state.wizard_step = 3


def _resolve_source_style_map(video_payload: dict, document_payloads: list[dict]) -> dict[int, tuple[str, str]]:
    style_map: dict[int, tuple[str, str]] = {}

    video_source_id = int(video_payload.get("source_id", 0) or 0)
    if video_source_id > 0:
        style_map[video_source_id] = ("red", "Video")

    for index, document_payload in enumerate(document_payloads, start=1):
        source_id = int(document_payload.get("source_id", 0) or 0)
        if source_id <= 0:
            continue
        color_name = DOCUMENT_COLORS[(index - 1) % len(DOCUMENT_COLORS)]
        style_map[source_id] = (color_name, f"Document {index}")

    return style_map


def _render_key_terms(key_terms: list[str]) -> None:
    if not key_terms:
        return
    formatted = ", ".join(f"**{term.strip()}**" for term in key_terms if term.strip())
    if formatted:
        st.markdown(f"Key terms: {formatted}")


def _render_reflection_points(reflection_points: list[dict] | list[str]) -> None:
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
            st.write(explanation)
        if under_the_hood:
            with st.expander(f"Under the hood for point {index}"):
                st.write(under_the_hood)


def _render_source_learning_section(section_payload: dict) -> None:
    st.subheader(section_payload.get("generated_title", "Untitled"))
    st.markdown("### Summary")
    st.markdown(str(section_payload.get("summary_text", "")).strip())

    deep_dive_text = str(section_payload.get("deep_dive_text", "")).strip()
    if deep_dive_text:
        st.markdown("### Deep Dive")
        st.markdown(deep_dive_text)

    key_terms = section_payload.get("key_terms", []) or []
    _render_key_terms([str(term) for term in key_terms])

    reflection_points = section_payload.get("reflection_points", []) or []
    st.markdown("### Socratic Reflection Points")
    _render_reflection_points(reflection_points)


def _render_ask_result() -> None:
    if not st.session_state.ask_result:
        return

    st.subheader("Answer")
    st.markdown(str(st.session_state.ask_result.get("answer", "")).strip())

    follow_up_question = (st.session_state.ask_result.get("follow_up_question") or "").strip()
    if follow_up_question:
        st.subheader("Follow-up Question")
        st.markdown(follow_up_question)


def _render_source_legend(video_payload: dict, document_payloads: list[dict]) -> None:
    st.markdown("### Source Attribution Legend")

    video_title = str(video_payload.get("generated_title", "Video")).strip() or "Video"
    st.caption(
        f"{SOURCE_COLOR_DOTS.get('red', '•')} Video (red): {video_title}"
    )

    for index, document_payload in enumerate(document_payloads, start=1):
        color_name = DOCUMENT_COLORS[(index - 1) % len(DOCUMENT_COLORS)]
        dot = SOURCE_COLOR_DOTS.get(color_name, "•")
        document_title = (
            str(document_payload.get("generated_title", f"Document {index}")).strip()
            or f"Document {index}"
        )
        st.caption(f"{dot} Document {index} ({color_name}): {document_title}")


def _render_attributed_sentence(
    item: dict,
    source_style_map: dict[int, tuple[str, str]],
) -> None:
    text_value = str(item.get("text", "")).strip()
    if not text_value:
        return

    source_id = int(item.get("source_id", 0) or 0)
    color_name, label = source_style_map.get(source_id, ("gray", "Unknown source"))
    emphasis_terms = [
        str(term).strip() for term in (item.get("emphasis_terms", []) or []) if str(term).strip()
    ]

    safe_text = text_value.replace("[", "(").replace("]", ")")
    dot = SOURCE_COLOR_DOTS.get(color_name, "•")
    st.markdown(f"{dot} :{color_name}[{label}] {safe_text}")

    if emphasis_terms:
        formatted_terms = ", ".join(f"**{term}**" for term in emphasis_terms)
        st.caption(f"Source: {label} | Keywords: {formatted_terms}")
    else:
        st.caption(f"Source: {label}")


def _render_ask_tab(has_user_id: bool) -> None:
    if not st.session_state.pipeline_ready:
        st.warning("Generate tailored content before asking questions.")
        _render_ask_result()
        return

    st.caption(
        f"Ready for user={st.session_state.user_id}. source_ids={st.session_state.source_ids} | "
        f"session_id={st.session_state.interaction_session_id}"
    )

    ask_query = st.text_input(
        "Ask anything about your material",
        value=st.session_state.ask_query,
    )
    st.session_state.ask_query = ask_query
    ask_clicked = st.button(
        "Ask",
        disabled=(not has_user_id or not ask_query.strip()),
    )

    try:
        if ask_clicked:
            st.session_state.ask_result = _post_json(
                "/ask",
                {
                    "session_id": st.session_state.interaction_session_id,
                    "source_ids": st.session_state.source_ids,
                    "query": ask_query.strip(),
                    "top_k": 8,
                },
            )
    except Exception as exc:
        st.error(str(exc))

    _render_ask_result()


def _render_generation_tabs(has_user_id: bool) -> None:
    generation_result = st.session_state.generation_result or {}
    if not generation_result:
        st.info("Generate tailored content to populate tabs.")
        return

    video_payload = generation_result.get("video", {})
    document_payloads = generation_result.get("documents", []) or []
    insights_payload = generation_result.get("insights", {})
    quiz_payload = generation_result.get("quiz", {})
    source_style_map = _resolve_source_style_map(video_payload, document_payloads)

    tab_video, tab_documents, tab_quiz_insights, tab_ask = st.tabs(
        ["Video", "Documents", "Cross-Source Synthesis & Assessment", "Ask"]
    )

    with tab_video:
        _render_source_learning_section(video_payload)

    with tab_documents:
        if not document_payloads:
            st.info("No document sections available.")
        else:
            document_labels = []
            for index, section_payload in enumerate(document_payloads, start=1):
                title = str(section_payload.get("generated_title", f"Document {index}")).strip()
                document_labels.append(title[:40] if len(title) > 40 else title)

            document_tabs = st.tabs(document_labels)
            for section_payload, document_tab in zip(document_payloads, document_tabs):
                with document_tab:
                    _render_source_learning_section(section_payload)

    with tab_quiz_insights:
        st.markdown("### Cross-Source Connections")
        _render_source_legend(video_payload, document_payloads)
        parallels = insights_payload.get("parallels", []) or []
        if parallels:
            for item in parallels:
                if isinstance(item, dict):
                    _render_attributed_sentence(item, source_style_map)
                else:
                    st.markdown(f"- {item}")
        else:
            st.write("No parallels available.")

        layman_bridge = str(insights_payload.get("layman_bridge", "")).strip()
        if layman_bridge:
            st.info(layman_bridge)

        synthesis_text = str(insights_payload.get("synthesis_text", "")).strip()
        if synthesis_text:
            st.markdown("### Synthesis")
            st.markdown(synthesis_text)

        st.markdown("### Knowledge Check")
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
                        st.markdown("**Under the hood**")
                        st.markdown(under_the_hood)
                    if source_evidence:
                        st.markdown("**Source evidence**")
                        for evidence in source_evidence:
                            st.markdown(f"- {evidence}")

        study_advice = str(quiz_payload.get("study_advice", "")).strip()
        if study_advice:
            st.markdown("### Study Advice")
            st.markdown(study_advice)

    with tab_ask:
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
    has_user_id = bool(st.session_state.user_id)
    if not st.session_state.user_id:
        st.warning("User ID is required. Set it in the sidebar before running requests.")

    st.header("Build Tailored Socratic Learning")
    st.caption(f"Step {st.session_state.wizard_step} of 3")

    if st.session_state.wizard_step == 1:
        st.markdown("### Step 1: Upload video")
        video_file = st.file_uploader(
            "Upload one video",
            type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
            accept_multiple_files=False,
            key="wizard_video_uploader",
        )

        if st.session_state.wizard_video_upload is not None:
            st.success(f"Selected: {st.session_state.wizard_video_upload['name']}")

        if st.button("Next", disabled=(not has_user_id)):
            if video_file is None and st.session_state.wizard_video_upload is None:
                st.error("Upload a video before continuing.")
            else:
                if video_file is not None:
                    st.session_state.wizard_video_upload = _pack_uploaded_file(video_file)
                st.session_state.wizard_step = 2
                st.rerun()

    elif st.session_state.wizard_step == 2:
        st.markdown("### Step 2: Upload source documents")
        document_files = st.file_uploader(
            "Upload one or more documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="wizard_document_uploader",
        )

        if st.session_state.wizard_document_uploads:
            existing_names = [item["name"] for item in st.session_state.wizard_document_uploads]
            st.success(f"Selected: {', '.join(existing_names)}")

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("Back"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col_next:
            if st.button("Next", disabled=(not has_user_id)):
                if not document_files and not st.session_state.wizard_document_uploads:
                    st.error("Upload at least one document before continuing.")
                else:
                    if document_files:
                        st.session_state.wizard_document_uploads = [
                            _pack_uploaded_file(file) for file in document_files
                        ]
                    st.session_state.wizard_step = 3
                    st.rerun()

    else:
        st.markdown("### Step 3: Generate tailored learning")
        st.caption("Estimated generation time: about 2 to 5 minutes.")

        selected_video = st.session_state.wizard_video_upload
        selected_documents = st.session_state.wizard_document_uploads

        if selected_video is None:
            st.warning("Video is missing. Go back to Step 1.")
        else:
            st.write(f"Video: {selected_video['name']}")
        if not selected_documents:
            st.warning("Documents are missing. Go back to Step 2.")
        else:
            st.write("Documents:")
            for upload in selected_documents:
                st.markdown(f"- {upload['name']}")

        col_back, col_generate = st.columns(2)
        with col_back:
            if st.button("Back to documents"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col_generate:
            generate_clicked = st.button(
                "Generate tailored Socratic learning",
                type="primary",
                disabled=(
                    not has_user_id
                    or st.session_state.wizard_video_upload is None
                    or not st.session_state.wizard_document_uploads
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
                        document_uploads=st.session_state.wizard_document_uploads,
                        stage_callback=_update_stage,
                    )
                except Exception as exc:
                    stage_placeholder.empty()
                    st.error(str(exc))
                else:
                    stage_placeholder.success("Done. Your tailored learning dashboard is ready.")
                    st.success("Tailored Socratic learning generated successfully.")

    if st.session_state.pipeline_ready:
        st.header("Learning Dashboard")
        _render_generation_tabs(has_user_id)

        if st.button("Start a new build"):
            _reset_builder()
            st.rerun()


if __name__ == "__main__":
    main()
