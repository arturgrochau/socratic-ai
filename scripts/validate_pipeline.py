from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


REQUEST_TIMEOUT = 600
INSUFFICIENT_CONTEXT_ANSWER = "I don't have enough information in the provided materials to answer that."

MODEL_RATES_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "whisper-1": (0.0, 0.0),
}


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
    video_path: Path,
    document_paths: list[Path],
) -> list[int]:
    with ExitStack() as stack:
        files: list[tuple[str, tuple[str, Any, str]]] = []

        video_file = stack.enter_context(video_path.open("rb"))
        files.append(("video", (video_path.name, video_file, "video/mp4")))

        for document_path in document_paths:
            document_file = stack.enter_context(document_path.open("rb"))
            mime_type = "application/pdf" if document_path.suffix.lower() == ".pdf" else "text/plain"
            files.append(("documents", (document_path.name, document_file, mime_type)))

        upload_response = requests.post(
            f"{api_base_url}/upload",
            headers=_headers(user_id),
            files=files,
            timeout=REQUEST_TIMEOUT,
        )

    if upload_response.status_code != 200:
        raise RuntimeError(
            f"/upload failed for user '{user_id}' with {upload_response.status_code}: "
            f"{_error_text(upload_response)}"
        )

    upload_payload = upload_response.json()
    video_source_id = int(upload_payload["video"]["source_id"])
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

    return [video_source_id, *document_source_ids]


def _run_interactions(*, api_base_url: str, user_id: str, source_ids: list[int]) -> dict[str, Any]:
    broad = _post_json(
        api_base_url=api_base_url,
        path="/ask",
        user_id=user_id,
        payload={
            "session_id": f"validate-{user_id}",
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
            "session_id": f"validate-{user_id}",
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


def _run_generation(*, api_base_url: str, user_id: str, source_ids: list[int]) -> dict[str, Any]:
    if len(source_ids) < 2:
        raise RuntimeError("Generation validation requires one video and at least one document source.")

    payload = _post_json(
        api_base_url=api_base_url,
        path="/generate-tailored-learning",
        user_id=user_id,
        payload={
            "video_source_id": int(source_ids[0]),
            "document_source_ids": [int(source_id) for source_id in source_ids[1:]],
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
    if len(video_summary_text) < 500:
        raise RuntimeError("Video summary is too short for elaborate mode.")

    video_deep_dive = str(video.get("deep_dive_text", "")).strip()
    if not video_deep_dive:
        raise RuntimeError("Generation payload is missing video deep-dive text.")
    if len(video_deep_dive) < 500:
        raise RuntimeError("Video deep-dive text is too short for elaborate mode.")
    _assert_clean_continuous_text("Video deep dive", video_deep_dive)
    if len(video_deep_dive.split()) > 920:
        raise RuntimeError("Video deep-dive text exceeded expected max-length guardrail.")

    video_key_terms = video.get("key_terms") or []
    if len(video_key_terms) < 5:
        raise RuntimeError("Generation payload has too few video key terms.")

    video_under_surface = str(video.get("under_surface_explainer", "")).strip()
    if not video_under_surface:
        raise RuntimeError("Generation payload is missing video under-surface explainer text.")
    if len(video_under_surface) < 600:
        raise RuntimeError("Video under-surface explainer is too short for elaborate mode.")
    _assert_clean_continuous_text("Video under-surface explainer", video_under_surface)

    video_diagnostic = video.get("diagnostic_checklist") or []
    if len(video_diagnostic) < 4:
        raise RuntimeError("Generation payload has too few video diagnostic checklist items.")

    video_term_breakdown = video.get("key_term_explanations") or []
    if len(video_term_breakdown) < 4:
        raise RuntimeError("Generation payload has too few video key-term explanations.")

    if len(video.get("reflection_points") or []) < 4:
        raise RuntimeError("Generation payload has too few video reflection points.")

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
        if len(summary_text) < 400:
            raise RuntimeError("Document summary is too short for elaborate mode.")

        deep_dive_text = str(document.get("deep_dive_text", "")).strip()
        if not deep_dive_text:
            raise RuntimeError("Generation payload has a document section without deep-dive text.")
        _assert_clean_continuous_text("Document deep dive", deep_dive_text)
        if len(deep_dive_text.split()) > 920:
            raise RuntimeError("Document deep-dive text exceeded expected max-length guardrail.")

        if len(document.get("key_terms") or []) < 5:
            raise RuntimeError("Generation payload has a document section with too few key terms.")

        under_surface = str(document.get("under_surface_explainer", "")).strip()
        if not under_surface:
            raise RuntimeError("Generation payload has a document section without under-surface explainer text.")
        if len(under_surface) < 500:
            raise RuntimeError("Document under-surface explainer is too short for elaborate mode.")
        _assert_clean_continuous_text("Document under-surface explainer", under_surface)

        diagnostic = document.get("diagnostic_checklist") or []
        if len(diagnostic) < 4:
            raise RuntimeError("Generation payload has a document section with too few diagnostic checklist items.")

        term_breakdown = document.get("key_term_explanations") or []
        if len(term_breakdown) < 4:
            raise RuntimeError("Generation payload has a document section with too few key-term explanations.")

        if len(document.get("reflection_points") or []) < 4:
            raise RuntimeError("Generation payload has a document section with too few reflection points.")

    intersections = insights.get("intersections") or []
    if not intersections:
        raise RuntimeError("Generation payload has no integrated intersections.")
    if len(intersections) < 3:
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
        if len(attributed_sentences) < 3:
            raise RuntimeError("Generation payload has an intersection with too few attributed sentences.")
        attributed_sentence_count += len(attributed_sentences)

        for sentence in attributed_sentences:
            if not required_sentence_fields.issubset(set(sentence.keys())):
                raise RuntimeError("Generation payload has a malformed attributed sentence entry.")

        inferred_extension = intersection.get("inferred_extension")
        inference_label = intersection.get("inference_label")
        if inferred_extension and inference_label != "inferred_extension":
            raise RuntimeError("Inferred extension exists without proper inference label.")

    if attributed_sentence_count < 10:
        raise RuntimeError("Generation payload has too few attributed sentence highlights across intersections.")

    if not str(insights.get("layman_bridge", "")).strip():
        raise RuntimeError("Generation payload has no layman bridge explanation.")
    _assert_clean_continuous_text("Layman bridge", str(insights.get("layman_bridge", "")).strip())
    if not str(insights.get("synthesis_text", "")).strip():
        raise RuntimeError("Generation payload has no synthesis text.")
    _assert_clean_continuous_text("Synthesis text", str(insights.get("synthesis_text", "")).strip())

    comparative_analysis = str(insights.get("comparative_analysis", "")).strip()
    if not comparative_analysis:
        raise RuntimeError("Generation payload has no comparative analysis text.")
    if len(comparative_analysis) < 280:
        raise RuntimeError("Comparative analysis text is too short for elaborate mode.")
    _assert_clean_continuous_text("Comparative analysis", comparative_analysis)

    application_scenarios = insights.get("application_scenarios") or []
    if len(application_scenarios) < 2:
        raise RuntimeError("Generation payload has too few application scenarios.")

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
        if len(transfer_steps) < 3:
            raise RuntimeError("Application scenario has too few transfer steps.")

    questions = quiz.get("questions") or []
    if len(questions) < 6:
        raise RuntimeError("Generation payload has too few quiz questions.")

    required_question_fields = {
        "question",
        "options",
        "answer_index",
        "explanation",
        "under_the_hood",
        "difficulty_level",
        "question_type",
        "source_evidence",
    }
    for question in questions:
        if not required_question_fields.issubset(set(question.keys())):
            raise RuntimeError("Generation payload has a malformed advanced quiz question entry.")

    if not str(quiz.get("study_advice", "")).strip():
        raise RuntimeError("Generation payload has no study advice.")

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
            "session_id": f"isolation-check-{attacker_user_id}",
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
    parser.add_argument("--user-a", default="validation-user-a")
    parser.add_argument("--user-b", default="validation-user-b")
    parser.add_argument("--video", default="test_assets/sample_video.mp4")
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]

    api_base_url = str(args.api_base_url).rstrip("/")
    video_path = (project_root / str(args.video)).resolve()
    document_paths = [(project_root / value).resolve() for value in args.documents]

    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    for document_path in document_paths:
        if not document_path.exists():
            raise FileNotFoundError(f"Document path does not exist: {document_path}")

    result: dict[str, Any] = {
        "api_base_url": api_base_url,
        "user_a": args.user_a,
        "user_b": args.user_b,
        "checks": [],
    }

    source_ids_user_a = _upload_and_process(
        api_base_url=api_base_url,
        user_id=args.user_a,
        video_path=video_path,
        document_paths=document_paths,
    )
    result["checks"].append({"name": "upload_process_user_a", "status": "passed"})

    interactions_user_a = _run_interactions(
        api_base_url=api_base_url,
        user_id=args.user_a,
        source_ids=source_ids_user_a,
    )
    result["checks"].append({"name": "interaction_user_a", "status": "passed"})

    generation_user_a = _run_generation(
        api_base_url=api_base_url,
        user_id=args.user_a,
        source_ids=source_ids_user_a,
    )
    result["checks"].append({"name": "generation_user_a", "status": "passed"})

    _upload_and_process(
        api_base_url=api_base_url,
        user_id=args.user_b,
        video_path=video_path,
        document_paths=document_paths,
    )
    result["checks"].append({"name": "upload_process_user_b", "status": "passed"})

    _assert_user_isolation(
        api_base_url=api_base_url,
        attacker_user_id=args.user_b,
        victim_source_ids=source_ids_user_a,
    )
    result["checks"].append({"name": "cross_user_isolation", "status": "passed"})

    result["interaction_previews"] = interactions_user_a
    result["generation_preview"] = generation_user_a

    sqlite_path = _resolve_sqlite_path(str(args.database_url), project_root)
    if sqlite_path is None:
        result["usage_summary"] = []
        result["generation_stage_summary"] = []
        result["usage_note"] = "Skipped usage summary because DATABASE_URL is not sqlite."
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
