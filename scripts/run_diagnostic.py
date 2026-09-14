"""
End-to-end diagnostic harness for Socratic AI.

Spins up the FastAPI backend in a subprocess on a free port, uploads a fixture
(PDF and/or video), runs the full pipeline, joins the resulting session JSONL
with DB telemetry, runs `scripts/checks.py`, drives 5 canned chat queries, and
writes a human-readable markdown report to `logs/diagnostic/<run_id>.md`.

Usage:
    python scripts/run_diagnostic.py --fixture test_assets/sample_notes.pdf
    python scripts/run_diagnostic.py --fixture test_assets/sample_notes.pdf \\
        --video test_assets/sample_video.mp4

(The old --baseline/--progressive modes were removed: GENERATION_PROGRESSIVE_CHUNKING
was a pre-v2.0 knob that no longer exists; windowing is fixed in app/generation.py.)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Ensure repo root is on sys.path when invoked as `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.checks import CheckResult, run_checks, suggest_tunes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTIC_DIR = REPO_ROOT / "logs" / "diagnostic"
SESSION_LOG_DIR = REPO_ROOT / "logs" / "sessions"
DEFAULT_USER_ID = f"diag-{uuid.uuid4().hex[:8]}"


CHAT_PROBE_QUERIES = [
    ("broad", "What is this material about overall?"),
    ("specific", "What is the most important mechanism described?"),
    ("off_topic", "What does the source say about climate policy?"),
    ("deepening", "Go deeper on the most important mechanism."),
    ("ambiguous", "Why does it work that way?"),
    # Tests multi-turn coherence: this probe makes no sense without the prior
    # conversation. A grounded answer here proves recent_turns is actually
    # being used and that the chat is coherent across turns.
    ("followup_coherence", "Building on the previous answer, what assumption is most likely to fail in a real-world setting?"),
]


CHAT_JUDGE_SYSTEM_PROMPT = (
    "You are grading a study chat answer on a 1-5 scale. The user uploaded "
    "study material and asked a question. The assistant produced an answer. "
    "Your job is to grade the answer against this rubric:\n"
    "  5 = excellent: directly answers the question, grounded in the material, "
    "specific and mechanism-level\n"
    "  4 = good: answers the question, mostly grounded, some specifics\n"
    "  3 = adequate: relevant but generic or partly off-target\n"
    "  2 = weak: barely relevant or vague\n"
    "  1 = bail: refuses to answer or produces a one-line dodge\n\n"
    "Special cases:\n"
    "  - For an off-topic question (the material doesn't cover it):\n"
    "      5 = acknowledges the gap concisely AND explicitly names the closest "
    "themes/concepts in the material (using source terms) AND poses a Socratic "
    "question that bridges from the user's interest to what the sources cover.\n"
    "      4 = acknowledges + offers a useful redirect, but the redirect is "
    "generic (no specific source themes named) or the bridge question is weak.\n"
    "      2 = acknowledges the gap but provides nothing useful in return.\n"
    "      1 = generic refusal with no redirect.\n"
    "  - For a deepening request ('go deeper'), grade 5 only if the answer "
    "introduces genuinely new dimensions (mechanism, constraint, tradeoff), "
    "not just rephrases.\n"
    "  - For a followup_coherence probe ('building on the previous answer...'), "
    "grade 5 only if the answer treats the prior conversation as context and "
    "extends from it rather than restarting from scratch; grade 2 if the "
    "answer ignores the prior turn entirely.\n\n"
    "Return ONLY JSON: {\"grade\": int, \"reasoning\": str (≤200 chars)}."
)

CHAT_JUDGE_JSON_SCHEMA = {
    "name": "chat_judge_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "grade": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning": {"type": "string"},
        },
        "required": ["grade", "reasoning"],
    },
}


def _judge_chat_answer(
    *,
    probe_label: str,
    query: str,
    answer: str,
    follow_up: str | None,
    source_titles: list[str],
) -> dict[str, Any]:
    """Call gpt-4o-mini to grade a single chat answer. Returns a dict with
    grade + reasoning + probe label. Errors fall through to grade=0 (no signal)."""
    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    if not api_key:
        return {"probe": probe_label, "grade": 0, "reasoning": "no API key"}
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    user_msg = (
        f"Probe type: {probe_label}\n"
        f"Source titles: {', '.join(source_titles) or '(none)'}\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer or '(empty)'}\n\n"
        f"Follow-up Socratic question (if any): {follow_up or '(none)'}"
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": CHAT_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_schema", "json_schema": CHAT_JUDGE_JSON_SCHEMA},
        )
        import json as _json
        payload = _json.loads(completion.choices[0].message.content or "{}")
        return {
            "probe": probe_label,
            "grade": int(payload.get("grade") or 0),
            "reasoning": str(payload.get("reasoning") or "")[:200],
        }
    except Exception as exc:
        return {"probe": probe_label, "grade": 0, "reasoning": f"judge call failed: {exc!r}"[:200]}


# ──────────────────────────────────────────────────────────────────────────────
# Backend lifecycle
# ──────────────────────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_ready(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception as exc:  # connection refused while starting
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"Backend at {base_url} never became ready: {last_err!r}")


@contextlib.contextmanager
def _backend_process(port: int, env_overrides: dict[str, str]):
    env = os.environ.copy()
    env.update(env_overrides)
    # Ensure venv python is used; falling back to sys.executable.
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python_bin = str(venv_python if venv_python.exists() else sys.executable)
    proc = subprocess.Popen(
        [
            python_bin, "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_ready(f"http://127.0.0.1:{port}")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────


def _headers(user_id: str) -> dict[str, str]:
    return {"X-User-ID": user_id}


def _upload(base_url: str, user_id: str, *, video_path: Path | None, document_paths: list[Path]) -> dict[str, Any]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    if video_path is not None:
        files.append(("video", (video_path.name, video_path.read_bytes(), "video/mp4")))
    for doc in document_paths:
        mime = "application/pdf" if doc.suffix.lower() == ".pdf" else "text/plain"
        files.append(("documents", (doc.name, doc.read_bytes(), mime)))
    r = requests.post(f"{base_url}/upload", headers=_headers(user_id), files=files, timeout=600)
    r.raise_for_status()
    return r.json()


def _process(base_url: str, user_id: str, *, video_id: int | None, document_ids: list[int]) -> dict[str, Any]:
    payload = {"video_source_id": video_id, "document_source_ids": document_ids}
    r = requests.post(f"{base_url}/process", headers=_headers(user_id), json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def _generate(base_url: str, user_id: str, *, video_id: int | None, document_ids: list[int]) -> dict[str, Any]:
    payload = {"video_source_id": video_id, "document_source_ids": document_ids}
    r = requests.post(
        f"{base_url}/generate-tailored-learning",
        headers=_headers(user_id),
        json=payload,
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def _ask(base_url: str, user_id: str, *, session_id: str, source_ids: list[int], query: str) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "source_ids": source_ids,
        "query": query,
        "top_k": 6,
    }
    r = requests.post(f"{base_url}/ask", headers=_headers(user_id), json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────────────


def _find_run_id(session_log_dir: Path, started_at: float) -> tuple[str, Path] | None:
    """Pick the newest gen-* JSONL written after started_at."""
    candidates = sorted(
        session_log_dir.glob("gen-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if c.stat().st_mtime >= started_at - 5:
            return c.stem, c
    return None


def _stage_cost_table_rows(jsonl_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in jsonl_rows if r.get("stage") == "stage_call"]


def _quality_table_rows(jsonl_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in jsonl_rows if r.get("stage") == "novelty"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _render_report(
    *,
    run_id: str,
    mode: str,
    fixture_paths: list[str],
    user_id: str,
    session_jsonl: Path,
    chat_jsonl: Path | None,
    generation_result: dict[str, Any],
    chat_results: list[dict[str, Any]],
    checks: list[CheckResult],
    tunes: list[str],
    duration_s: float,
    started_at: float,
) -> str:
    rows = _read_jsonl(session_jsonl)
    chat_rows = _read_jsonl(chat_jsonl) if chat_jsonl else []
    stage_calls = _stage_cost_table_rows(rows)
    quality = _quality_table_rows(rows)

    # Pull complete cost from api_call_usage DB (covers processing + linking +
    # chat + embeddings, which run outside the generate-tailored-learning
    # session context and so don't land in the JSONL). NB: the cost query
    # window starts from the *diagnostic's* started_at (captured by the
    # harness before /upload), not from session_start (which only fires once
    # /generate-tailored-learning opens its context — too late).
    from config import db_engine as _db
    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    db_user_id = (session_start or {}).get("user_id", user_id)
    # NOTE: caller wires started_at = the harness wall-clock start.
    with _db.connect() as conn:
        from sqlalchemy import text as _text
        db_rolled = conn.execute(
            _text(
                "SELECT call_stage, model_name, "
                "COALESCE(SUM(total_tokens), 0) AS total, "
                "COALESCE(SUM(prompt_tokens), 0) AS prompt, "
                "COALESCE(SUM(completion_tokens), 0) AS completion, "
                "COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :user_id "
                "AND CAST(strftime('%s', created_at) AS INTEGER) >= :since "
                "GROUP BY call_stage, model_name "
                "ORDER BY total DESC"
            ),
            {"user_id": db_user_id, "since": int(max(0.0, started_at - 5))},
        ).mappings().all()
    total_tokens = sum(int(r["total"] or 0) for r in db_rolled)
    total_prompt = sum(int(r["prompt"] or 0) for r in db_rolled)
    total_completion = sum(int(r["completion"] or 0) for r in db_rolled)
    total_calls = sum(int(r["n"] or 0) for r in db_rolled)

    lines: list[str] = []
    lines.append(f"# Diagnostic report — `{run_id}`")
    lines.append("")
    lines.append(f"- **Mode**: `{mode}`")
    lines.append(f"- **Fixtures**: {', '.join(f'`{p}`' for p in fixture_paths)}")
    lines.append(f"- **User**: `{user_id}`")
    lines.append(f"- **Wall clock**: {duration_s:.1f} s")
    lines.append(f"- **Generated at**: {datetime.utcnow().isoformat()}Z")
    lines.append("")

    # ── Failure-mode checklist ────────────────────────────────────────────────
    lines.append("## Failure-mode checklist")
    lines.append("")
    lines.append("| | Check | Evidence |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| {c.badge()} | `{c.name}` | {c.evidence} |")
    lines.append("")

    # ── Top tunes ─────────────────────────────────────────────────────────────
    lines.append("## Top tunes to try")
    lines.append("")
    if tunes:
        for i, tune in enumerate(tunes, 1):
            lines.append(f"{i}. {tune}")
    else:
        lines.append("_Nothing flagged — pipeline looks clean._")
    lines.append("")

    # ── Cost summary ──────────────────────────────────────────────────────────
    lines.append("## Cost")
    lines.append("")
    lines.append(f"- Total tokens (all LLM calls for this user during the run): **{total_tokens:,}** "
                 f"({total_prompt:,} prompt / {total_completion:,} completion)")
    lines.append(f"- LLM calls: **{total_calls}** ({len(stage_calls)} via structured generation)")
    if db_rolled:
        lines.append("")
        lines.append("| Call stage | Model | Calls | Prompt | Completion | Total |")
        lines.append("|---|---|---|---|---|---|")
        for r in db_rolled:
            lines.append(
                f"| `{r['call_stage']}` | `{r['model_name']}` | {r['n']} "
                f"| {int(r['prompt'] or 0):,} | {int(r['completion'] or 0):,} | {int(r['total'] or 0):,} |"
            )
    lines.append("")

    # ── Stage cost table ──────────────────────────────────────────────────────
    if stage_calls:
        lines.append("### Per-stage")
        lines.append("")
        lines.append("| Stage | Model | Attempt | Duration (ms) | Prompt | Completion | Critic |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in stage_calls:
            lines.append(
                f"| `{r.get('stage_name','?')}` | `{r.get('model','?')}` | {r.get('attempt','?')} "
                f"| {r.get('duration_ms','?')} | {r.get('prompt_tokens') or '-'} "
                f"| {r.get('completion_tokens') or '-'} | {'✓' if r.get('critic') else ''} |"
            )
        lines.append("")

    # ── Quality / novelty table ───────────────────────────────────────────────
    if quality:
        lines.append("### Section novelty")
        lines.append("")
        lines.append("| Section | Source | Overlap | Overlap / Total | Violates |")
        lines.append("|---|---|---|---|---|")
        for r in quality:
            flag = "⚠" if r.get("violates_threshold") else ""
            lines.append(
                f"| `{r.get('section_type','?')}` | {r.get('source_id','?')} "
                f"| {float(r.get('overlap_ratio') or 0.0):.3f} "
                f"| {r.get('overlap_sentences','?')} / {r.get('total_sentences','?')} | {flag} |"
            )
        lines.append("")

    # ── Chat smoke ────────────────────────────────────────────────────────────
    if chat_results:
        lines.append("## Chat smoke (5 canned probes)")
        lines.append("")
        lines.append("| Probe | Query | Answer length | Route | Top retrieval |")
        lines.append("|---|---|---|---|---|")
        for r, chat_row in zip(chat_results, chat_rows, strict=False):
            lines.append(
                f"| `{r.get('label','?')}` "
                f"| {r.get('query','?')[:60]} "
                f"| {chat_row.get('answer_length','?')} "
                f"| `{chat_row.get('route','?')}` "
                f"| {chat_row.get('top_retrieval_score','?')} |"
            )
        lines.append("")
        # Show the actual first answer so the human can eyeball "feel"
        if chat_results:
            first = chat_results[0]
            lines.append("### First answer (sample)")
            lines.append("")
            lines.append("> " + (first.get("answer", "") or "").replace("\n", "\n> "))
            lines.append("")
            if first.get("follow_up_question"):
                lines.append(f"**Follow-up Socratic question**: {first['follow_up_question']}")
                lines.append("")

    # ── Source learning shape ─────────────────────────────────────────────────
    lines.append("## Generated learning shape")
    lines.append("")
    video = generation_result.get("video")
    documents = generation_result.get("documents") or []
    if video:
        lines.append(f"- **Video section**: `{video.get('generated_title','?')}` "
                     f"({len(video.get('deep_dive_text','') or '')} chars deep dive, "
                     f"{len(video.get('reflection_points') or [])} reflections)")
    for doc in documents:
        lines.append(f"- **Document section**: `{doc.get('generated_title','?')}` "
                     f"({len(doc.get('deep_dive_text','') or '')} chars deep dive, "
                     f"{len(doc.get('reflection_points') or [])} reflections)")
    insights = generation_result.get("insights") or {}
    if insights.get("synthesis_text"):
        lines.append(f"- **Synthesis**: {len(insights['synthesis_text'])} chars; "
                     f"{len(insights.get('intersections') or [])} intersections")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main flow
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", default=[], help="Path to PDF fixture (repeatable)")
    parser.add_argument("--video", default=None, help="Path to video fixture")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--api-base-url", default=None, help="If set, talk to an already-running backend")
    parser.add_argument("--skip-chat", action="store_true", help="Skip the 5 chat probes")
    parser.add_argument(
        "--validate-cache",
        action="store_true",
        help="After the main run, call /generate again on the same source IDs to validate the cache. Should be near-zero cost; flagged if not.",
    )
    parser.add_argument(
        "--judge-chat",
        action="store_true",
        help="Use gpt-4o-mini to grade each chat probe answer 1-5. Adds ~5 small LLM calls per run.",
    )
    args = parser.parse_args(argv)

    fixture_paths = [Path(p).resolve() for p in args.fixture]
    if not fixture_paths and not args.video:
        parser.error("At least one --fixture (PDF) or --video is required.")
    video_path = Path(args.video).resolve() if args.video else None

    mode = "default"  # single mode since the pre-v2.0 progressive-chunking knob was removed
    env_overrides: dict[str, str] = {"ENABLE_SESSION_LOG": "true"}

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    base_url = args.api_base_url
    backend_cm: contextlib.AbstractContextManager
    if base_url is None:
        port = _find_free_port()
        backend_cm = _backend_process(port, env_overrides)
        base_url = f"http://127.0.0.1:{port}"
    else:
        backend_cm = contextlib.nullcontext()
        print(f"[diagnostic] Using existing backend at {base_url}")

    user_id = args.user_id
    print(f"[diagnostic] Mode: {mode}  User: {user_id}")
    print(f"[diagnostic] Fixtures: {[str(p) for p in fixture_paths]}; video={video_path}")

    with backend_cm:
        # 1. Upload
        print("[diagnostic] Uploading fixtures...")
        upload_result = _upload(base_url, user_id, video_path=video_path, document_paths=fixture_paths)

        video_id: int | None = None
        if upload_result.get("video"):
            video_id = int(upload_result["video"]["source_id"])
        document_ids = [int(d["source_id"]) for d in upload_result.get("documents") or []]
        print(f"[diagnostic] Uploaded video_id={video_id}, document_ids={document_ids}")

        # 2. Process + link
        print("[diagnostic] Processing + linking...")
        _process(base_url, user_id, video_id=video_id, document_ids=document_ids)

        # 3. Generate
        print("[diagnostic] Generating tailored learning...")
        generation_result = _generate(base_url, user_id, video_id=video_id, document_ids=document_ids)

        # 4. Chat probes
        chat_results: list[dict[str, Any]] = []
        chat_session_id = f"diag-chat-{uuid.uuid4().hex[:8]}"
        all_source_ids = ([video_id] if video_id else []) + document_ids
        if not args.skip_chat and all_source_ids:
            print(f"[diagnostic] Running {len(CHAT_PROBE_QUERIES)} chat probes...")
            for label, query in CHAT_PROBE_QUERIES:
                try:
                    resp = _ask(base_url, user_id, session_id=chat_session_id,
                                source_ids=all_source_ids, query=query)
                    chat_results.append({"label": label, "query": query, **resp})
                except requests.HTTPError as exc:
                    chat_results.append({"label": label, "query": query, "error": str(exc)})

        # 5. Cache validation (optional): rerun /generate against the same
        # source IDs. With caches working, the second pass should make ~0
        # additional generation calls.
        cache_validation: dict[str, Any] | None = None
        if args.validate_cache:
            print("[diagnostic] Validating cache: re-running /generate on same source IDs...")
            # Snapshot the highest existing api_call_usage row id for this user
            # so the check can count strictly-newer rows. SQLite's CURRENT_TIMESTAMP
            # only resolves to seconds; using row id avoids the boundary problem
            # where the last chat probe and the rerun start land in the same second.
            from sqlalchemy import text as _text_pre

            from config import db_engine as _db_pre
            with _db_pre.connect() as _conn:
                pre_max_id = _conn.execute(
                    _text_pre("SELECT COALESCE(MAX(id), 0) FROM api_call_usage WHERE user_id = :u"),
                    {"u": user_id},
                ).scalar() or 0
            cache_t0 = time.time()
            try:
                _generate(base_url, user_id, video_id=video_id, document_ids=document_ids)
                cache_validation = {
                    "ok": True,
                    "rerun_started_at": cache_t0,
                    "rerun_duration_s": time.time() - cache_t0,
                    "pre_max_id": int(pre_max_id),
                }
            except Exception as exc:
                cache_validation = {
                    "ok": False,
                    "rerun_started_at": cache_t0,
                    "error": str(exc)[:200],
                    "pre_max_id": int(pre_max_id),
                }

    duration_s = time.time() - started_at

    # Optional LLM-as-judge grading of chat answers. Fires outside the
    # backend subprocess (uses its own OpenAI client) so it doesn't appear in
    # the run's api_call_usage rows under this user_id.
    chat_judgements: list[dict[str, Any]] = []
    if args.judge_chat and chat_results:
        source_titles: list[str] = []
        if generation_result.get("video"):
            source_titles.append(generation_result["video"].get("generated_title") or "video")
        for d in generation_result.get("documents") or []:
            source_titles.append(d.get("generated_title") or "document")
        print(f"[diagnostic] LLM-judging {len(chat_results)} chat answers...")
        for r in chat_results:
            if r.get("error"):
                chat_judgements.append({"probe": r.get("label"), "grade": 0, "reasoning": "chat call failed"})
                continue
            chat_judgements.append(
                _judge_chat_answer(
                    probe_label=str(r.get("label", "?")),
                    query=str(r.get("query", "")),
                    answer=str(r.get("answer", "")),
                    follow_up=r.get("follow_up_question"),
                    source_titles=source_titles,
                )
            )

    # 5. Find the run's session JSONL
    found = _find_run_id(SESSION_LOG_DIR, started_at)
    if found is None:
        print("[diagnostic] WARNING: no gen-*.jsonl found; check ENABLE_SESSION_LOG and wiring.")
        run_id = f"diag-{uuid.uuid4().hex[:8]}"
        session_jsonl = SESSION_LOG_DIR / f"{run_id}.jsonl"
    else:
        run_id, session_jsonl = found
    chat_jsonl = SESSION_LOG_DIR / f"chat-{chat_session_id}.jsonl"

    # 6. Synthesis-paraphrase context for the synthesis check
    source_texts: list[str] = []
    if generation_result.get("video"):
        source_texts.append(generation_result["video"].get("deep_dive_text", "") or "")
    for doc in generation_result.get("documents") or []:
        source_texts.append(doc.get("deep_dive_text", "") or "")
    synthesis_text = (generation_result.get("insights") or {}).get("synthesis_text", "")

    # 7. Run checks
    from config import db_engine
    extras = [chat_jsonl] if chat_jsonl.exists() else []
    checks = run_checks(
        session_jsonl,
        db_engine,
        run_id,
        extra_jsonl_paths=extras,
        source_texts=source_texts,
        synthesis_text=synthesis_text,
        synthesis_payload=generation_result.get("insights"),
        cache_validation=cache_validation,
        chat_judgements=chat_judgements if chat_judgements else None,
        run_started_at=started_at,
    )
    tunes = suggest_tunes(checks, top_n=3)

    # 8. Render report
    report = _render_report(
        run_id=run_id,
        mode=mode,
        fixture_paths=[str(p.relative_to(REPO_ROOT)) for p in fixture_paths] + (
            [str(video_path.relative_to(REPO_ROOT))] if video_path else []
        ),
        user_id=user_id,
        session_jsonl=session_jsonl,
        chat_jsonl=chat_jsonl if chat_jsonl.exists() else None,
        generation_result=generation_result,
        chat_results=chat_results,
        checks=checks,
        tunes=tunes,
        duration_s=duration_s,
        started_at=started_at,
    )
    report_path = DIAGNOSTIC_DIR / f"{run_id}-{mode}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[diagnostic] Report written to {report_path}")
    # Print short summary to terminal
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    print(f"[diagnostic] Result: {len(fails)} fail, {len(warns)} warn, {len(checks)-len(fails)-len(warns)} pass")
    if tunes:
        print("[diagnostic] Tunes to try:")
        for t in tunes:
            print(f"   - {t}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
