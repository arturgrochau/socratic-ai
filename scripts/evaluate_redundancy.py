from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


REQUEST_TIMEOUT = 600


@dataclass
class IterationResult:
    user_id: str
    source_max_overlap: float
    cross_max_overlap: float
    source_max_claim_overlap: float
    cross_max_claim_overlap: float
    db_cross_max_overlap: float | None
    db_cross_max_claim_overlap: float | None


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
) -> tuple[int, list[int]]:
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

    return video_source_id, document_source_ids


def _normalize_text(text_value: str) -> str:
    return " ".join(str(text_value or "").split()).strip().lower()


def _token_overlap_ratio(left_text: str, right_text: str) -> float:
    left_tokens = {token for token in _normalize_text(left_text).split() if len(token) >= 4}
    right_tokens = {token for token in _normalize_text(right_text).split() if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / float(min(len(left_tokens), len(right_tokens)))


def _sentence_split(text_value: str) -> list[str]:
    normalized = _normalize_text(text_value)
    if not normalized:
        return []
    chunks = [chunk.strip() for chunk in normalized.replace("!", ".").replace("?", ".").split(".")]
    return [chunk for chunk in chunks if chunk]


def _claims(text_value: str) -> set[str]:
    claims: set[str] = set()
    for sentence in _sentence_split(text_value):
        if len(sentence) < 28:
            continue
        claims.add(sentence)
        if len(claims) >= 24:
            break
    return claims


def _pairwise_overlap_ratio(left_text: str, right_text: str) -> float:
    left = _normalize_text(left_text)
    right = _normalize_text(right_text)
    if not left or not right:
        return 0.0

    ratio = SequenceMatcher(None, left, right).ratio()
    token_ratio = _token_overlap_ratio(left, right)

    left_sentences = _sentence_split(left)
    right_sentences = _sentence_split(right)
    if not left_sentences or not right_sentences:
        sentence_ratio = 0.0
    else:
        overlap = 0
        for sentence in left_sentences:
            if sentence in right_sentences:
                overlap += 1
        sentence_ratio = overlap / float(len(left_sentences))

    return max(ratio, token_ratio, sentence_ratio)


def _pairwise_claim_overlap_ratio(left_text: str, right_text: str) -> float:
    left_claims = _claims(left_text)
    right_claims = _claims(right_text)
    if not left_claims or not right_claims:
        return 0.0
    shared = left_claims.intersection(right_claims)
    return len(shared) / float(min(len(left_claims), len(right_claims)))


def _max_pairwise_overlap(section_texts: dict[str, str]) -> tuple[float, float]:
    labels = [label for label, text_value in section_texts.items() if _normalize_text(text_value)]
    max_overlap = 0.0
    max_claim_overlap = 0.0

    for index, left_label in enumerate(labels):
        for right_label in labels[index + 1 :]:
            overlap = _pairwise_overlap_ratio(section_texts[left_label], section_texts[right_label])
            claim_overlap = _pairwise_claim_overlap_ratio(section_texts[left_label], section_texts[right_label])
            if overlap > max_overlap:
                max_overlap = overlap
            if claim_overlap > max_claim_overlap:
                max_claim_overlap = claim_overlap

    return max_overlap, max_claim_overlap


def _source_section_text_map(section: dict[str, Any]) -> dict[str, str]:
    reflection = " ".join(
        " ".join(
            part
            for part in [
                str(point.get("question", "") or "").strip(),
                str(point.get("explanation", "") or "").strip(),
                str(point.get("under_the_hood", "") or "").strip(),
            ]
            if part
        )
        for point in section.get("reflection_points", [])
    ).strip()

    return {
        "summary": str(section.get("summary_text", "") or "").strip(),
        "deep_dive": str(section.get("deep_dive_text", "") or "").strip(),
        "under_surface": str(section.get("under_surface_explainer", "") or "").strip(),
        "reflection": reflection,
    }


def _cross_section_text_map(insights: dict[str, Any], quiz: dict[str, Any]) -> dict[str, str]:
    intersections = " ".join(
        " ".join(
            part
            for part in [
                str(item.get("intersection_title", "") or "").strip(),
                str(item.get("why_it_matters", "") or "").strip(),
                str(item.get("integrated_explanation", "") or "").strip(),
                " ".join(
                    str(sentence.get("text", "") or "").strip()
                    for sentence in item.get("attributed_sentences", [])
                    if str(sentence.get("text", "") or "").strip()
                ),
            ]
            if part
        )
        for item in insights.get("intersections", [])
    ).strip()

    scenarios = " ".join(
        " ".join(
            part
            for part in [
                str(item.get("scenario_title", "") or "").strip(),
                str(item.get("scenario_prompt", "") or "").strip(),
                " ".join(str(step).strip() for step in item.get("transfer_steps", []) if str(step).strip()),
                str(item.get("common_pitfall", "") or "").strip(),
            ]
            if part
        )
        for item in insights.get("application_scenarios", [])
    ).strip()

    quiz_text = " ".join(
        " ".join(
            part
            for part in [
                str(question.get("question", "") or "").strip(),
                str(question.get("explanation", "") or "").strip(),
                str(question.get("under_the_hood", "") or "").strip(),
                " ".join(str(s).strip() for s in question.get("source_evidence", []) if str(s).strip()),
            ]
            if part
        )
        for question in quiz.get("questions", [])
    ).strip()

    return {
        "mapping": " ".join(
            part
            for part in [intersections, str(insights.get("layman_bridge", "") or "").strip()]
            if part
        ).strip(),
        "constraint": " ".join(
            part
            for part in [
                str(insights.get("synthesis_text", "") or "").strip(),
                str(insights.get("comparative_analysis", "") or "").strip(),
            ]
            if part
        ).strip(),
        "transfer": scenarios,
        "decision": " ".join(
            part
            for part in [quiz_text, str(quiz.get("study_advice", "") or "").strip()]
            if part
        ).strip(),
    }


def _fetch_db_cross_metrics(db_path: Path, user_id: str) -> tuple[float | None, float | None]:
    if not db_path.exists():
        return None, None

    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT MAX(overlap_ratio)
            FROM generation_quality_metrics
            WHERE user_id = ?
              AND section_type = 'cross-source'
              AND pair_key NOT LIKE '%::claims'
            """,
            (user_id,),
        )
        overlap_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT MAX(overlap_ratio)
            FROM generation_quality_metrics
            WHERE user_id = ?
              AND section_type = 'cross-source'
              AND pair_key LIKE '%::claims'
            """,
            (user_id,),
        )
        claim_row = cursor.fetchone()

    overlap_value = float(overlap_row[0]) if overlap_row and overlap_row[0] is not None else None
    claim_value = float(claim_row[0]) if claim_row and claim_row[0] is not None else None
    return overlap_value, claim_value


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": ordered[0],
        "mean": statistics.mean(ordered),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate section-role overlap across repeated generation runs.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Base URL for the API.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the video file.")
    parser.add_argument(
        "--document",
        action="append",
        dest="documents",
        default=[],
        type=Path,
        help="Path to a document file. Pass multiple times for multiple docs.",
    )
    parser.add_argument("--iterations", type=int, default=8, help="Number of end-to-end runs to execute.")
    parser.add_argument("--user-prefix", default="redundancy-eval", help="User ID prefix for test runs.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("chroma_data/chroma.sqlite3"),
        help="SQLite path for reading generation_quality_metrics.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write detailed results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        raise RuntimeError(f"Video file not found: {args.video}")
    if not args.documents:
        raise RuntimeError("At least one --document input is required for cross-source evaluation.")
    missing_docs = [path for path in args.documents if not path.exists()]
    if missing_docs:
        raise RuntimeError(f"Document files not found: {', '.join(str(path) for path in missing_docs)}")
    if args.iterations <= 0:
        raise RuntimeError("--iterations must be >= 1")

    results: list[IterationResult] = []

    for iteration in range(1, args.iterations + 1):
        user_id = f"{args.user_prefix}-{iteration:03d}"
        print(f"[{iteration}/{args.iterations}] Running generation for user: {user_id}")

        video_source_id, document_source_ids = _upload_and_process(
            api_base_url=args.api_base_url,
            user_id=user_id,
            video_path=args.video,
            document_paths=args.documents,
        )

        payload = _post_json(
            api_base_url=args.api_base_url,
            path="/generate-tailored-learning",
            user_id=user_id,
            payload={
                "video_source_id": video_source_id,
                "document_source_ids": document_source_ids,
            },
        )

        source_sections: list[dict[str, Any]] = []
        if isinstance(payload.get("video"), dict):
            source_sections.append(payload["video"])
        source_sections.extend([item for item in payload.get("documents", []) if isinstance(item, dict)])

        source_overlaps: list[float] = []
        source_claim_overlaps: list[float] = []
        for section in source_sections:
            max_overlap, max_claim_overlap = _max_pairwise_overlap(_source_section_text_map(section))
            source_overlaps.append(max_overlap)
            source_claim_overlaps.append(max_claim_overlap)

        cross_max_overlap = 0.0
        cross_max_claim_overlap = 0.0
        if len(source_sections) >= 2:
            insights = payload.get("insights") if isinstance(payload.get("insights"), dict) else {}
            quiz = payload.get("quiz") if isinstance(payload.get("quiz"), dict) else {}
            cross_max_overlap, cross_max_claim_overlap = _max_pairwise_overlap(
                _cross_section_text_map(insights, quiz)
            )

        db_cross_overlap, db_cross_claim_overlap = _fetch_db_cross_metrics(args.db_path, user_id)

        result = IterationResult(
            user_id=user_id,
            source_max_overlap=max(source_overlaps) if source_overlaps else 0.0,
            cross_max_overlap=cross_max_overlap,
            source_max_claim_overlap=max(source_claim_overlaps) if source_claim_overlaps else 0.0,
            cross_max_claim_overlap=cross_max_claim_overlap,
            db_cross_max_overlap=db_cross_overlap,
            db_cross_max_claim_overlap=db_cross_claim_overlap,
        )
        results.append(result)

        print(
            "  source_max_overlap={:.3f} source_max_claim_overlap={:.3f} "
            "cross_max_overlap={:.3f} cross_max_claim_overlap={:.3f}".format(
                result.source_max_overlap,
                result.source_max_claim_overlap,
                result.cross_max_overlap,
                result.cross_max_claim_overlap,
            )
        )

    report = {
        "config": {
            "api_base_url": args.api_base_url,
            "video": str(args.video),
            "documents": [str(path) for path in args.documents],
            "iterations": args.iterations,
        },
        "aggregate": {
            "source_max_overlap": _summary([item.source_max_overlap for item in results]),
            "source_max_claim_overlap": _summary([item.source_max_claim_overlap for item in results]),
            "cross_max_overlap": _summary([item.cross_max_overlap for item in results]),
            "cross_max_claim_overlap": _summary([item.cross_max_claim_overlap for item in results]),
            "db_cross_max_overlap": _summary(
                [item.db_cross_max_overlap for item in results if item.db_cross_max_overlap is not None]
            ),
            "db_cross_max_claim_overlap": _summary(
                [item.db_cross_max_claim_overlap for item in results if item.db_cross_max_claim_overlap is not None]
            ),
        },
        "iterations": [
            {
                "user_id": item.user_id,
                "source_max_overlap": item.source_max_overlap,
                "source_max_claim_overlap": item.source_max_claim_overlap,
                "cross_max_overlap": item.cross_max_overlap,
                "cross_max_claim_overlap": item.cross_max_claim_overlap,
                "db_cross_max_overlap": item.db_cross_max_overlap,
                "db_cross_max_claim_overlap": item.db_cross_max_claim_overlap,
            }
            for item in results
        ],
    }

    print("\nAggregate summary:")
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=True))

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nSaved detailed report to: {args.output_json}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
