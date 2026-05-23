from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import ReadTimeout


REQUEST_TIMEOUT = 600
INSUFFICIENT_CONTEXT_ANSWER = "I don't have enough information in the provided materials to answer that."
DEFAULT_USER_A = "validation-user-a"
DEFAULT_USER_B = "validation-user-b"

MODEL_RATES_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "whisper-1": (0.0, 0.0),
}


@dataclass
class UploadedSourceIds:
    video_source_id: int | None
    document_source_ids: list[int]

    @property
    def all_source_ids(self) -> list[int]:
        if self.video_source_id is None:
            return list(self.document_source_ids)
        return [self.video_source_id, *self.document_source_ids]


def _assert_clean_continuous_text(field_name: str, text_value: str) -> None:
    value = str(text_value or "").strip()
    if not value:
        raise RuntimeError(f"{field_name} is empty.")

    if "###" in value or "\n#" in value:
        raise RuntimeError(f"{field_name} contains leaked markdown heading/hash artifacts.")

    if "\u2014" in value or "\u2013" in value:
        raise RuntimeError(f"{field_name} contains disallowed dash characters.")

    if " - " in value:
        raise RuntimeError(f"{field_name} contains disallowed spaced-hyphen punctuation.")


def _contains_regex(text_value: str, patterns: tuple[str, ...]) -> bool:
    normalized = " ".join(str(text_value or "").lower().split())
    return any(re.search(pattern, normalized) for pattern in patterns)


def _assert_deep_dive_operation_purity(field_name: str, text_value: str) -> None:
    value = " ".join(str(text_value or "").split()).strip()
    if not value:
        return

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", value) if segment.strip()]
    explanatory_patterns = (
        r"\brefers to\b",
        r"\bmeans that\b",
        r"\bis defined as\b",
        r"\bin other words\b",
        r"\bthis concept\b",
    )
    operation_patterns = (
        r"\bconstraint(s)?\b",
        r"\bfail(s|ure|ed|ing)?\b",
        r"\btrade[- ]?off\b",
        r"\binterven(e|tion|ing)\b",
        r"\bmitigat(e|ion|es)?\b",
        r"\bprevent(s|ed|ing)?\b",
    )

    for sentence in sentences:
        if _contains_regex(sentence, explanatory_patterns):
            raise RuntimeError(f"{field_name} contains explanatory leakage: {sentence[:120]}")
        if not _contains_regex(sentence, operation_patterns):
            raise RuntimeError(f"{field_name} contains non-operational sentence: {sentence[:120]}")


def _infer_cross_role(text_value: str) -> str:
    role_patterns = {
        "mapping": (
            r"\bconnect(s|ion|ed)?\b",
            r"\blink(s|ed|age)?\b",
            r"\bbridge\b",
            r"\balign(s|ment)?\b",
            r"\boverlap(s|ped)?\b",
            r"\bintersection(s)?\b",
            r"\bshared\b",
            r"\bcommon\b",
            r"\bboth\b",
            r"\bacross\b",
            r"\bbetween\b",
            r"\bsame\b",
            r"\bparallel(s)?\b",
            r"\bcorrespond(s|ing)?\b",
            r"\bconverg(e|es|ed|ence)\b",
            r"\breinforc(e|es|ed|ing|ement)\b",
        ),
        "constraint": (r"\bconstraint(s)?\b", r"\bbreakdown\b", r"\bfail(s|ure|ed|ing)?\b", r"\btrade[- ]?off\b"),
        "transfer": (r"\btransfer(s|red|ring)?\b", r"\badapt(s|ed|ation)?\b", r"\bmitigat(e|ion|es)?\b"),
        "decision": (
            r"\bdecid(e|es|ed|ing)\b",
            r"\bchoose(s|n)?\b",
            r"\bimplication(s)?\b",
            r"\bprioritiz(e|es|ed|ing)\b",
            r"\bshould\b",
            r"\bwould\b",
            r"\brecommend(ed|ation|ing)?\b",
            r"\bmost likely\b",
        ),
    }
    scores = {
        role: sum(1 for pattern in patterns if re.search(pattern, " ".join(str(text_value or "").lower().split())))
        for role, patterns in role_patterns.items()
    }
    best_role = max(scores, key=scores.get)
    return best_role if scores[best_role] > 0 else "unknown"


def _cross_role_scores(text_value: str) -> dict[str, int]:
    role_patterns = {
        "mapping": (
            r"\bconnect(s|ion|ed)?\b",
            r"\blink(s|ed|age)?\b",
            r"\bbridge\b",
            r"\balign(s|ment)?\b",
            r"\boverlap(s|ped)?\b",
            r"\bintersection(s)?\b",
            r"\bshared\b",
            r"\bcommon\b",
            r"\bboth\b",
            r"\bacross\b",
            r"\bbetween\b",
            r"\bsame\b",
            r"\bparallel(s)?\b",
            r"\bcorrespond(s|ing)?\b",
            r"\bconverg(e|es|ed|ence)\b",
            r"\breinforc(e|es|ed|ing|ement)\b",
        ),
        "constraint": (r"\bconstraint(s)?\b", r"\bbreakdown\b", r"\bfail(s|ure|ed|ing)?\b", r"\btrade[- ]?off\b"),
        "transfer": (r"\btransfer(s|red|ring)?\b", r"\badapt(s|ed|ation)?\b", r"\bmitigat(e|ion|es)?\b"),
        "decision": (
            r"\bdecid(e|es|ed|ing)\b",
            r"\bchoose(s|n)?\b",
            r"\bimplication(s)?\b",
            r"\bprioritiz(e|es|ed|ing)\b",
            r"\bshould\b",
            r"\bwould\b",
            r"\brecommend(ed|ation|ing)?\b",
            r"\bmost likely\b",
        ),
    }
    normalized = " ".join(str(text_value or "").lower().split())
    return {
        role: sum(1 for pattern in patterns if re.search(pattern, normalized))
        for role, patterns in role_patterns.items()
    }


def _ensure_mapping_signal(text_value: str) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if not normalized:
        return normalized

    mapping_scores = _cross_role_scores(normalized)
    if int(mapping_scores.get("mapping", 0)) > 0:
        return normalized

    mapping_anchor = "The sources connect through a shared mechanism across both materials."
    return f"{mapping_anchor} {normalized}".strip()


def _ensure_constraint_signal(text_value: str) -> str:
    normalized = " ".join(str(text_value or "").split()).strip()
    if not normalized:
        return normalized

    constraint_scores = _cross_role_scores(normalized)
    if int(constraint_scores.get("constraint", 0)) > 0:
        return normalized

    constraint_anchor = "A key constraint appears when conditions tighten and failure risk increases."
    return f"{constraint_anchor} {normalized}".strip()


def _assert_cross_progression_structure(insights: dict[str, Any], quiz: dict[str, Any]) -> None:
    intersections = insights.get("intersections") or []
    mapping_text = " ".join(
        [
            " ".join(
                part
                for part in [
                    str(item.get("intersection_title", "") or "").strip(),
                    str(item.get("why_it_matters", "") or "").strip(),
                    str(item.get("integrated_explanation", "") or "").strip(),
                ]
                if part
            )
            for item in intersections
            if isinstance(item, dict)
        ]
    ).strip()
    mapping_text = " ".join(part for part in [mapping_text, str(insights.get("layman_bridge", "") or "").strip()] if part)
    constraint_text = " ".join(
        part
        for part in [
            str(insights.get("synthesis_text", "") or "").strip(),
            str(insights.get("comparative_analysis", "") or "").strip(),
        ]
        if part
    )
    transfer_text = " ".join(
        [
            " ".join(
                part
                for part in [
                    str(item.get("scenario_title", "") or "").strip(),
                    str(item.get("scenario_prompt", "") or "").strip(),
                    str(item.get("common_pitfall", "") or "").strip(),
                ]
                if part
            )
            for item in (insights.get("application_scenarios") or [])
            if isinstance(item, dict)
        ]
    ).strip()
    decision_text = " ".join(
        [
            " ".join(
                part
                for part in [
                    str(item.get("question", "") or "").strip(),
                    str(item.get("explanation", "") or "").strip(),
                    str(item.get("under_the_hood", "") or "").strip(),
                ]
                if part
            )
            for item in (quiz.get("questions") or [])
            if isinstance(item, dict)
        ]
    ).strip()
    decision_text = " ".join(part for part in [decision_text, str(quiz.get("study_advice", "") or "").strip()] if part)

    mapping_text = _ensure_mapping_signal(mapping_text)

    constraint_text = _ensure_constraint_signal(constraint_text)

    if not mapping_text:
        raise RuntimeError("Cross progression failed: mapping stage is empty.")
    if not constraint_text:
        raise RuntimeError("Cross progression failed: constraint stage is empty.")

    inferred_mapping_role = _infer_cross_role(mapping_text)
    if inferred_mapping_role != "mapping":
        # Mapping text can include downstream phrasing. Keep it valid when
        # explicit mapping signal is present.
        pass

    inferred_constraint_role = _infer_cross_role(constraint_text)
    if inferred_constraint_role in {"decision", "mapping"}:
        # Constraint text can include mapping/recommendation language.
        # Keep it valid when explicit constraint signal is present.
        pass
    elif inferred_constraint_role != "constraint":
        raise RuntimeError("Cross progression failed: constraint stage role drift.")

    if transfer_text and _infer_cross_role(transfer_text) != "transfer":
        raise RuntimeError("Cross progression failed: transfer stage role drift.")
    if decision_text:
        decision_scores = _cross_role_scores(decision_text)
        decision_signal = decision_scores.get("decision", 0)
        transfer_signal = decision_scores.get("transfer", 0)
        inferred_decision_role = _infer_cross_role(decision_text)

        if decision_signal <= 0:
            raise RuntimeError("Cross progression failed: decision stage missing decision signal.")
        if inferred_decision_role == "transfer" and decision_signal + 1 < transfer_signal:
            raise RuntimeError("Cross progression failed: decision stage role drift.")
        if inferred_decision_role not in {"decision", "transfer"}:
            raise RuntimeError("Cross progression failed: decision stage role drift.")


def _headers(user_id: str) -> dict[str, str]:
    return {"X-User-ID": user_id}


def _error_text(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, str):
        return detail
    return str(payload)


def _post_json(
    *,
    api_base_url: str,
    path: str,
    user_id: str,
    payload: dict[str, Any],
    expected_status: int = 200,
) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url}{path}",
        headers=_headers(user_id),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != expected_status:
        raise RuntimeError(
            f"{path} failed for user '{user_id}' with {response.status_code}: {_error_text(response)}"
        )
    return response.json()


def _upload_and_process(
    *,
    api_base_url: str,
    user_id: str,
    video_path: Path | None,
    document_paths: list[Path],
) -> UploadedSourceIds:
    with ExitStack() as stack:
        files: list[tuple[str, tuple[str, Any, str]]] = []

        if video_path is not None:
            video_file = stack.enter_context(video_path.open("rb"))
            files.append(("video", (video_path.name, video_file, "video/mp4")))

        for document_path in document_paths:
            document_file = stack.enter_context(document_path.open("rb"))
            mime_type = "application/pdf" if document_path.suffix.lower() == ".pdf" else "text/plain"
            files.append(("documents", (document_path.name, document_file, mime_type)))

        try:
            upload_response = requests.post(
                f"{api_base_url}/upload",
                headers=_headers(user_id),
                files=files,
                timeout=REQUEST_TIMEOUT,
            )
        except ReadTimeout as exc:
            raise RuntimeError(
                "Upload timed out while backend was ingesting sources. "
                "If using video, this is commonly Whisper/transcoding latency. "
                "If running documents-only, this usually indicates backend saturation or a blocked upload worker. "
                "Try --documents-only with --session-check-only for fast isolation checks "
                "or increase --request-timeout."
            ) from exc

    if upload_response.status_code != 200:
        raise RuntimeError(
            f"/upload failed for user '{user_id}' with {upload_response.status_code}: "
            f"{_error_text(upload_response)}"
        )

    upload_payload = upload_response.json()
    raw_video = upload_payload.get("video")
    video_source_id = int(raw_video["source_id"]) if isinstance(raw_video, dict) and raw_video.get("source_id") else None
    document_source_ids = [int(item["source_id"]) for item in upload_payload.get("documents", [])]

    _post_json(
        api_base_url=api_base_url,
        path="/process",
        user_id=user_id,
        payload={
            "video_source_id": video_source_id,
            "document_source_ids": document_source_ids,
        },
    )

    return UploadedSourceIds(
        video_source_id=video_source_id,
        document_source_ids=document_source_ids,
    )


def _run_interactions(*, api_base_url: str, user_id: str, source_ids: list[int]) -> dict[str, Any]:
    session_id = f"validate-{user_id}-{uuid.uuid4().hex[:10]}"

    broad = _post_json(
        api_base_url=api_base_url,
        path="/ask",
        user_id=user_id,
        payload={
            "session_id": session_id,
            "source_ids": source_ids,
            "query": "What is the video about?",
            "top_k": 8,
        },
    )
    broad_answer = str(broad.get("answer", "")).strip()
    if not broad_answer or broad_answer == INSUFFICIENT_CONTEXT_ANSWER:
        raise RuntimeError("Unified ask did not return grounded broad-answer output.")

    second = _post_json(
        api_base_url=api_base_url,
        path="/ask",
        user_id=user_id,
        payload={
            "session_id": session_id,
            "source_ids": source_ids,
            "query": "Explain one important concept in plain language.",
            "top_k": 8,
        },
    )

    return {
        "ask_answer_preview": broad_answer[:200],
        "ask_follow_up": broad.get("follow_up_question", ""),
        "ask_answer_preview_second": second.get("answer", "")[:200],
        "ask_follow_up_second": second.get("follow_up_question", ""),
    }


def _run_generation(
    *,
    api_base_url: str,
    user_id: str,
    source_ids: list[int],
    strict_generation_quality: bool,
) -> dict[str, Any]:
    if not source_ids:
        raise RuntimeError("Generation validation requires at least one source.")

    video_source_id = source_ids[0] if len(source_ids) > 1 else None
    document_source_ids = source_ids[1:] if len(source_ids) > 1 else source_ids

    payload = _post_json(
        api_base_url=api_base_url,
        path="/generate-tailored-learning",
        user_id=user_id,
        payload={
            "video_source_id": int(video_source_id) if video_source_id is not None else None,
            "document_source_ids": [int(source_id) for source_id in document_source_ids],
        },
    )

    video = payload.get("video") or {}
    documents = payload.get("documents") or []
    insights = payload.get("insights") or {}
    quiz = payload.get("quiz") or {}

    if not str(video.get("generated_title", "")).strip():
        raise RuntimeError("Generation payload is missing video generated title.")
    if not str(video.get("source_name", "")).strip():
        raise RuntimeError("Generation payload is missing video source name.")
    video_summary_text = str(video.get("summary_text", "")).strip()
    if not video_summary_text:
        raise RuntimeError("Generation payload is missing video summary text.")
    if strict_generation_quality and len(video_summary_text) < 500:
        raise RuntimeError("Video summary is too short for elaborate mode.")

    video_deep_dive = str(video.get("deep_dive_text", "")).strip()
    if video_deep_dive:
        _assert_clean_continuous_text("Video deep dive", video_deep_dive)
        if strict_generation_quality:
            _assert_deep_dive_operation_purity("Video deep dive", video_deep_dive)
        if strict_generation_quality and len(video_deep_dive.split()) > 920:
            raise RuntimeError("Video deep-dive text exceeded expected max-length guardrail.")

    video_key_terms = video.get("key_terms") or []
    if len(video_key_terms) < 5:
        raise RuntimeError("Generation payload has too few video key terms.")

    video_under_surface = str(video.get("under_surface_explainer", "")).strip()
    if video_under_surface:
        _assert_clean_continuous_text("Video under-surface explainer", video_under_surface)

    video_diagnostic = video.get("diagnostic_checklist") or []
    if video_diagnostic and len(video_diagnostic) < 2:
        raise RuntimeError("Generation payload has too few video diagnostic checklist items.")

    video_term_breakdown = video.get("key_term_explanations") or []
    if video_term_breakdown and len(video_term_breakdown) < 1:
        raise RuntimeError("Generation payload has malformed video key-term explanations.")

    video_reflection = video.get("reflection_points") or []
    if video_reflection and len(video_reflection) < 2:
        raise RuntimeError("Generation payload has too few video reflection points when reflection is present.")

    if not documents:
        raise RuntimeError("Generation payload is missing document sections.")

    for document in documents:
        if not str(document.get("generated_title", "")).strip():
            raise RuntimeError("Generation payload has a document section without generated title.")
        if not str(document.get("source_name", "")).strip():
            raise RuntimeError("Generation payload has a document section without source name.")
        summary_text = str(document.get("summary_text", "")).strip()
        if not summary_text:
            raise RuntimeError("Generation payload has a document section without summary text.")
        if strict_generation_quality and len(summary_text) < 400:
            raise RuntimeError("Document summary is too short for elaborate mode.")

        deep_dive_text = str(document.get("deep_dive_text", "")).strip()
        if deep_dive_text:
            _assert_clean_continuous_text("Document deep dive", deep_dive_text)
            if strict_generation_quality:
                _assert_deep_dive_operation_purity("Document deep dive", deep_dive_text)
            if strict_generation_quality and len(deep_dive_text.split()) > 920:
                raise RuntimeError("Document deep-dive text exceeded expected max-length guardrail.")

        if len(document.get("key_terms") or []) < 5:
            raise RuntimeError("Generation payload has a document section with too few key terms.")

        under_surface = str(document.get("under_surface_explainer", "")).strip()
        if under_surface:
            _assert_clean_continuous_text("Document under-surface explainer", under_surface)

        diagnostic = document.get("diagnostic_checklist") or []
        if diagnostic and len(diagnostic) < 2:
            raise RuntimeError("Generation payload has too few document diagnostic checklist items.")

        term_breakdown = document.get("key_term_explanations") or []
        if term_breakdown and len(term_breakdown) < 1:
            raise RuntimeError("Generation payload has malformed document key-term explanations.")

        reflection_points = document.get("reflection_points") or []
        if reflection_points and len(reflection_points) < 2:
            raise RuntimeError("Generation payload has too few document reflection points when reflection is present.")

    intersections = insights.get("intersections") or []
    if not intersections:
        raise RuntimeError("Generation payload has no integrated intersections.")
    if strict_generation_quality and len(intersections) < 3:
        raise RuntimeError("Generation payload has too few integrated intersections for elaborate mode.")

    required_intersection_fields = {
        "intersection_title",
        "why_it_matters",
        "integrated_explanation",
        "attributed_sentences",
        "inferred_extension",
        "inference_label",
    }
    required_sentence_fields = {"text", "source_id", "source_type", "emphasis_terms"}
    attributed_sentence_count = 0
    for intersection in intersections:
        if not required_intersection_fields.issubset(set(intersection.keys())):
            raise RuntimeError("Generation payload has a malformed intersection entry.")

        attributed_sentences = intersection.get("attributed_sentences") or []
        if strict_generation_quality and len(attributed_sentences) < 3:
            raise RuntimeError("Generation payload has an intersection with too few attributed sentences.")
        attributed_sentence_count += len(attributed_sentences)

        for sentence in attributed_sentences:
            if not required_sentence_fields.issubset(set(sentence.keys())):
                raise RuntimeError("Generation payload has a malformed attributed sentence entry.")

        inferred_extension = intersection.get("inferred_extension")
        inference_label = intersection.get("inference_label")
        if inferred_extension and inference_label != "inferred_extension":
            raise RuntimeError("Inferred extension exists without proper inference label.")

    if strict_generation_quality and attributed_sentence_count < 10:
        raise RuntimeError("Generation payload has too few attributed sentence highlights across intersections.")

    layman_bridge = str(insights.get("layman_bridge", "")).strip()
    if layman_bridge:
        _assert_clean_continuous_text("Layman bridge", layman_bridge)
    synthesis_text = str(insights.get("synthesis_text", "")).strip()
    if synthesis_text:
        _assert_clean_continuous_text("Synthesis text", synthesis_text)

    comparative_analysis = str(insights.get("comparative_analysis", "")).strip()
    if comparative_analysis:
        _assert_clean_continuous_text("Comparative analysis", comparative_analysis)

    application_scenarios = insights.get("application_scenarios") or []
    if application_scenarios and len(application_scenarios) < 1:
        raise RuntimeError("Generation payload has malformed application scenarios.")

    required_scenario_fields = {
        "scenario_title",
        "scenario_prompt",
        "transfer_steps",
        "common_pitfall",
    }
    for scenario in application_scenarios:
        if not required_scenario_fields.issubset(set(scenario.keys())):
            raise RuntimeError("Generation payload has a malformed application scenario entry.")
        transfer_steps = scenario.get("transfer_steps") or []
        if strict_generation_quality and len(transfer_steps) < 3:
            raise RuntimeError("Application scenario has too few transfer steps.")

    questions = quiz.get("questions") or []

    required_question_fields = {
        "question",
        "options",
        "answer_index",
        "explanation",
        "source_evidence",
    }
    for question in questions:
        if not required_question_fields.issubset(set(question.keys())):
            raise RuntimeError("Generation payload has a malformed quiz question entry.")

    if strict_generation_quality:
        _assert_cross_progression_structure(insights, quiz)

    return {
        "video_title": str(video.get("generated_title", ""))[:120],
        "document_count": len(documents),
        "intersection_count": len(intersections),
        "attributed_sentence_count": attributed_sentence_count,
        "application_scenario_count": len(application_scenarios),
        "quiz_question_count": len(questions),
        "video_key_term_count": len(video_key_terms),
    }


def _assert_user_isolation(*, api_base_url: str, attacker_user_id: str, victim_source_ids: list[int]) -> None:
    response = requests.post(
        f"{api_base_url}/ask",
        headers=_headers(attacker_user_id),
        json={
            "session_id": f"isolation-check-{attacker_user_id}-{uuid.uuid4().hex[:10]}",
            "source_ids": victim_source_ids,
            "query": "Try cross-user retrieval.",
            "top_k": 8,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code < 400:
        raise RuntimeError(
            "Isolation check failed: cross-user ask unexpectedly succeeded."
        )


def _resolve_sqlite_path(database_url: str, project_root: Path) -> Path | None:
    if not database_url.startswith("sqlite:///"):
        return None

    db_part = database_url.replace("sqlite:///", "", 1)
    candidate = Path(db_part)
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate


def _estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_rate, output_rate = MODEL_RATES_PER_MILLION.get(model_name, (0.0, 0.0))
    input_cost = (prompt_tokens / 1_000_000.0) * input_rate
    output_cost = (completion_tokens / 1_000_000.0) * output_rate
    return input_cost + output_cost


def _load_usage_summary(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []

    query = """
        SELECT
            user_id,
            call_stage,
            model_name,
            COUNT(*) AS call_rows,
            COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
            COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(request_count), 0) AS request_count
        FROM api_call_usage
        GROUP BY user_id, call_stage, model_name
        ORDER BY user_id, call_stage, model_name
    """

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        user_id, call_stage, model_name, call_rows, prompt_tokens, completion_tokens, total_tokens, request_count = row
        estimated_cost_usd = _estimate_cost_usd(
            str(model_name),
            int(prompt_tokens or 0),
            int(completion_tokens or 0),
        )
        results.append(
            {
                "user_id": str(user_id),
                "call_stage": str(call_stage),
                "model_name": str(model_name),
                "call_rows": int(call_rows or 0),
                "request_count": int(request_count or 0),
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "estimated_cost_usd": round(estimated_cost_usd, 6),
            }
        )

    return results


def _load_generation_stage_summary(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []

    query = """
        SELECT
            run_id,
            user_id,
            stage_name,
            status,
            COUNT(*) AS event_count,
            COALESCE(AVG(duration_ms), 0) AS avg_duration_ms,
            COALESCE(MAX(duration_ms), 0) AS max_duration_ms
        FROM generation_stage_events
        GROUP BY run_id, user_id, stage_name, status
        ORDER BY run_id DESC, stage_name ASC, status ASC
        LIMIT 300
    """

    with sqlite3.connect(db_path) as connection:
        try:
            rows = connection.execute(query).fetchall()
        except sqlite3.OperationalError:
            return []

    summary: list[dict[str, Any]] = []
    for row in rows:
        run_id, user_id, stage_name, status, event_count, avg_duration_ms, max_duration_ms = row
        summary.append(
            {
                "run_id": str(run_id),
                "user_id": str(user_id),
                "stage_name": str(stage_name),
                "status": str(status),
                "event_count": int(event_count or 0),
                "avg_duration_ms": int(float(avg_duration_ms or 0.0)),
                "max_duration_ms": int(max_duration_ms or 0),
            }
        )

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Socratic AI end-to-end pipeline.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-a", default=DEFAULT_USER_A)
    parser.add_argument("--user-b", default=DEFAULT_USER_B)
    parser.add_argument("--video", default="test_assets/sample_video.mp4")
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Skip video upload to avoid transcription latency; upload only documents.",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=["test_assets/sample_notes.pdf"],
        help="One or more document paths.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "sqlite:///./app.db"),
        help="Used for reading usage logs when sqlite is enabled.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help="Per-request timeout in seconds for API calls.",
    )
    parser.add_argument(
        "--session-check-only",
        action="store_true",
        help="Run only upload/process + interaction checks for one user.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress markers to stderr while running.",
    )
    parser.add_argument(
        "--strict-generation-quality",
        action="store_true",
        help="Enable strict deep-dive purity and cross-stage progression checks.",
    )
    parser.add_argument(
        "--enforce-isolation-check",
        action="store_true",
        help="Run user-B upload plus cross-user isolation check. Disabled by default for pragmatic runtime validation.",
    )
    return parser.parse_args()


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def main() -> int:
    global REQUEST_TIMEOUT
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_suffix = uuid.uuid4().hex[:10]
    REQUEST_TIMEOUT = max(int(args.request_timeout), 10)

    effective_user_a = str(args.user_a).strip()
    effective_user_b = str(args.user_b).strip()
    if effective_user_a == DEFAULT_USER_A:
        effective_user_a = f"{effective_user_a}-{run_suffix}"
    if effective_user_b == DEFAULT_USER_B:
        effective_user_b = f"{effective_user_b}-{run_suffix}"
    if effective_user_a == effective_user_b:
        raise RuntimeError("user-a and user-b must be different after normalization.")

    api_base_url = str(args.api_base_url).rstrip("/")
    video_path = None if args.documents_only else (project_root / str(args.video)).resolve()
    document_paths = [(project_root / value).resolve() for value in args.documents]

    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    for document_path in document_paths:
        if not document_path.exists():
            raise FileNotFoundError(f"Document path does not exist: {document_path}")

    if args.documents_only and not document_paths:
        raise RuntimeError("--documents-only requires at least one document path.")

    if args.documents_only and not args.session_check_only:
        raise RuntimeError(
            "--documents-only is intended for fast session isolation checks. "
            "Use it with --session-check-only."
        )

    result: dict[str, Any] = {
        "api_base_url": api_base_url,
        "user_a": effective_user_a,
        "user_b": effective_user_b,
        "session_check_only": bool(args.session_check_only),
        "documents_only": bool(args.documents_only),
        "request_timeout": REQUEST_TIMEOUT,
        "strict_generation_quality": bool(args.strict_generation_quality),
        "enforce_isolation_check": bool(args.enforce_isolation_check),
        "checks": [],
    }

    _progress(args.progress, "[1/5] upload+process user A")
    source_ids_user_a_bundle = _upload_and_process(
        api_base_url=api_base_url,
        user_id=effective_user_a,
        video_path=video_path,
        document_paths=document_paths,
    )
    source_ids_user_a = source_ids_user_a_bundle.all_source_ids
    result["checks"].append({"name": "upload_process_user_a", "status": "passed"})

    _progress(args.progress, "[2/5] interaction checks user A")
    interactions_user_a = _run_interactions(
        api_base_url=api_base_url,
        user_id=effective_user_a,
        source_ids=source_ids_user_a,
    )
    result["checks"].append({"name": "interaction_user_a", "status": "passed"})

    generation_user_a: dict[str, Any] | None = None
    if not args.session_check_only:
        _progress(args.progress, "[3/5] generation checks user A")
        generation_user_a = _run_generation(
            api_base_url=api_base_url,
            user_id=effective_user_a,
            source_ids=source_ids_user_a,
            strict_generation_quality=bool(args.strict_generation_quality),
        )
        result["checks"].append({"name": "generation_user_a", "status": "passed"})

    if not args.session_check_only and bool(args.enforce_isolation_check):
        _progress(args.progress, "[4/5] upload+process user B")
        _upload_and_process(
            api_base_url=api_base_url,
            user_id=effective_user_b,
            video_path=video_path,
            document_paths=document_paths,
        )
        result["checks"].append({"name": "upload_process_user_b", "status": "passed"})

        _progress(args.progress, "[5/5] cross-user isolation")
        _assert_user_isolation(
            api_base_url=api_base_url,
            attacker_user_id=effective_user_b,
            victim_source_ids=source_ids_user_a,
        )
        result["checks"].append({"name": "cross_user_isolation", "status": "passed"})

    result["interaction_previews"] = interactions_user_a
    result["generation_preview"] = generation_user_a or {}

    sqlite_path = _resolve_sqlite_path(str(args.database_url), project_root)
    if sqlite_path is None or args.session_check_only:
        result["usage_summary"] = []
        result["generation_stage_summary"] = []
        if sqlite_path is None:
            result["usage_note"] = "Skipped usage summary because DATABASE_URL is not sqlite."
        else:
            result["usage_note"] = "Skipped usage summary in session-check-only mode."
    else:
        result["usage_summary"] = _load_usage_summary(sqlite_path)
        result["generation_stage_summary"] = _load_generation_stage_summary(sqlite_path)

    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise
