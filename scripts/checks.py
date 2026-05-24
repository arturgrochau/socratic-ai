"""
Failure-mode catalog: rules over session telemetry that map to remediation hints.

Every check reads from:
  * `logs/sessions/<run_id>.jsonl` — emitted by app.session_logger
  * DB tables (api_call_usage, generation_stage_events, generation_novelty_ledger)

Each check returns a CheckResult with a status (pass | warn | fail) and a
remediation hint pointing at a specific config knob or prompt file. Used both
by `scripts/run_diagnostic.py` (rendered in the markdown report) and by the
e2e pytest (fails the test on `status == "fail"`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text


Status = Literal["pass", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    status: Status
    evidence: str
    remediation_hint: str
    details: dict[str, Any] = field(default_factory=dict)

    def badge(self) -> str:
        return {"pass": "✅", "warn": "⚠", "fail": "✘"}[self.status]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _records(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("stage") == stage]


# ──────────────────────────────────────────────────────────────────────────────
# Individual checks. Each returns one CheckResult.
# ──────────────────────────────────────────────────────────────────────────────


def check_stage_failures(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Any generation stage that exhausted all retries → fail."""
    with db_engine.connect() as conn:
        # ':critic' looks like a bind param to SQLAlchemy — concat manually.
        failed = conn.execute(
            text(
                "SELECT stage_name, attempt_number FROM generation_stage_events "
                "WHERE run_id = :run_id AND status = 'failed' "
                "AND stage_name NOT LIKE :critic_suffix "
                "AND attempt_number >= 3"
            ),
            {"run_id": run_id, "critic_suffix": "%:critic"},
        ).mappings().all()
    if not failed:
        return CheckResult(
            "stage_failures",
            "pass",
            "No generation stage exhausted its retry budget.",
            "",
        )
    return CheckResult(
        "stage_failures",
        "fail",
        f"{len(failed)} stage(s) failed all retries: {', '.join(row['stage_name'] for row in failed)}",
        "Inspect the stage's prompt — usually a JSON schema mismatch or an over-stuffed input. "
        "Try shortening the user message or splitting the schema.",
        {"failed": [dict(row) for row in failed]},
    )


def check_retry_pressure(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Stages that needed ≥2 attempts → warn (working but fragile)."""
    with db_engine.connect() as conn:
        retried = conn.execute(
            text(
                "SELECT stage_name, MAX(attempt_number) AS max_attempt "
                "FROM generation_stage_events "
                "WHERE run_id = :run_id AND status = 'succeeded' "
                "GROUP BY stage_name HAVING MAX(attempt_number) >= 2"
            ),
            {"run_id": run_id},
        ).mappings().all()
    if not retried:
        return CheckResult("retry_pressure", "pass", "All stages succeeded on attempt 1.", "")
    names = ", ".join(f"{row['stage_name']}(×{row['max_attempt']})" for row in retried)
    return CheckResult(
        "retry_pressure",
        "warn",
        f"Stages that needed retries: {names}",
        "Rewrite the failing field's description in the prompt. Ambiguous schemas cost tokens.",
        {"retried": [dict(row) for row in retried]},
    )


def check_critic_fallback(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Any critic call → warn. Budget exhausted → fail."""
    critic_calls = [r for r in rows if r.get("stage") == "stage_call" and r.get("critic")]
    # back-compat: any record where stage_name ends with ':critic' counts too
    if not critic_calls:
        return CheckResult("critic_fallback", "pass", "Base model handled every stage.", "")
    exhausted = [c for c in critic_calls if c.get("remaining_budget") == 0]
    status: Status = "fail" if exhausted else "warn"
    return CheckResult(
        "critic_fallback",
        status,
        f"{len(critic_calls)} critic fallback(s); budget exhausted on {len(exhausted)}.",
        "Either bump MAX_SOURCE_CRITIC_CALLS_PER_RUN, or simplify the base prompt so gpt-4o-mini doesn't need to escalate.",
        {"stages": [c.get("stage_name") for c in critic_calls]},
    )


def check_section_novelty(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Any section overlap above SECTION_REDUNDANCY_RATIO_THRESHOLD (0.34) → warn."""
    violations = [
        r for r in _records(rows, "novelty") if r.get("violates_threshold")
    ]
    if not violations:
        return CheckResult("section_novelty", "pass", "All sections under the 0.34 overlap threshold.", "")
    summary = ", ".join(
        f"{v['section_type']}({v['overlap_ratio']:.2f})" for v in violations
    )
    return CheckResult(
        "section_novelty",
        "warn",
        f"Sections above overlap threshold: {summary}",
        "Strengthen the 'do not repeat prior context' clause in the offending prompt, or "
        "reorder the prior_texts list so the most-similar prior is first.",
        {"violations": violations},
    )


def check_hard_cuts(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    cuts = _records(rows, "hard_cut")
    if not cuts:
        return CheckResult("hard_cuts", "pass", "No hard cuts fired.", "")
    cut_sections = ", ".join(sorted({c.get("cut_section", "?") for c in cuts}))
    return CheckResult(
        "hard_cuts",
        "warn",
        f"{len(cuts)} hard cut(s) on: {cut_sections}",
        "The pair of sections is fundamentally redundant. Consider merging them in the schema, "
        "or reducing one's word budget so they can't drift into each other's territory.",
        {"cuts": cuts},
    )


def check_low_mechanism(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    flags = _records(rows, "low_mechanism_density")
    if not flags:
        return CheckResult("low_mechanism_density", "pass", "All sources had usable mechanism density.", "")
    ids = ", ".join(str(f.get("source_id")) for f in flags)
    return CheckResult(
        "low_mechanism_density",
        "warn",
        f"Source(s) flagged as low-density: {ids}",
        "Source is likely thin on actual mechanism. Consider a shorter deep-dive (drop MAX_DEEP_DIVE_WORDS) "
        "rather than padding text the source doesn't support.",
        {"flags": flags},
    )


def check_token_budget(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Read total tokens for this run from api_call_usage; warn if over ceiling."""
    from config import DIAGNOSTIC_TOKEN_CEILING

    # api_call_usage doesn't store run_id directly — sum over the user_id for
    # the duration of this run. Use the session log's first/last timestamps.
    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    session_end = next((r for r in rows if r.get("stage") == "session_end"), None)
    if not session_start or not session_end:
        return CheckResult("token_budget", "warn", "Session start/end markers missing.", "Check session_logger wiring.")
    user_id = session_start.get("user_id", "unknown")
    ts_start = session_start.get("ts", 0)
    ts_end = session_end.get("ts", ts_start)
    with db_engine.connect() as conn:
        total = conn.execute(
            text(
                "SELECT COALESCE(SUM(total_tokens), 0) AS total, COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :user_id "
                "AND strftime('%s', created_at) BETWEEN :start AND :end"
            ),
            {"user_id": user_id, "start": int(ts_start - 5), "end": int(ts_end + 5)},
        ).mappings().first()
    total_tokens = int((total or {}).get("total") or 0)
    n_calls = int((total or {}).get("n") or 0)
    status: Status = "pass" if total_tokens <= DIAGNOSTIC_TOKEN_CEILING else "warn"
    return CheckResult(
        "token_budget",
        status,
        f"{total_tokens} tokens across {n_calls} calls (ceiling {DIAGNOSTIC_TOKEN_CEILING}).",
        "Trim MAX_GROUNDING_CHARS in app/generation.py if completions are short, "
        "or raise DIAGNOSTIC_TOKEN_CEILING if this is the new normal.",
        {"total_tokens": total_tokens, "n_calls": n_calls},
    )


def check_prompt_completion_ratio(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Stages where completion < 30% of prompt are over-stuffed."""
    calls = [r for r in _records(rows, "stage_call") if r.get("prompt_tokens") and r.get("completion_tokens")]
    over_stuffed: list[str] = []
    for c in calls:
        prompt = float(c["prompt_tokens"])
        completion = float(c["completion_tokens"])
        if prompt > 0 and completion / prompt < 0.30:
            over_stuffed.append(f"{c.get('stage_name', '?')}({completion/prompt:.0%})")
    if not over_stuffed:
        return CheckResult("prompt_completion_ratio", "pass", "All stages had reasonable input/output ratio.", "")
    return CheckResult(
        "prompt_completion_ratio",
        "warn",
        f"Over-stuffed prompts (completion < 30% of input): {', '.join(over_stuffed)}",
        "Reduce MAX_GROUNDING_CHARS or shorten the role description in the offending stage's prompt.",
        {"over_stuffed": over_stuffed},
    )


def check_chat_quality(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Chat-specific. Only fires when a chat-<session_id>.jsonl is present
    alongside the run JSONL — the diagnostic feeds it explicitly."""
    chat_queries = _records(rows, "chat_query")
    if not chat_queries:
        return CheckResult("chat_quality", "pass", "(no chat traffic in this run)", "")

    refused = [q for q in chat_queries if q.get("contains_refusal_pattern")]
    no_context = [q for q in chat_queries if q.get("route") == "no_context"]
    low_score = [q for q in chat_queries if (q.get("top_retrieval_score") or 0.0) < 0.4 and q.get("route") == "grounded"]

    if refused:
        return CheckResult(
            "chat_quality",
            "fail",
            f"{len(refused)}/{len(chat_queries)} chat answers leaked the refusal pattern.",
            "The relaxed prompt in prompts/interaction.py should suppress this. Check that the "
            "system prompt was actually loaded (no caching of an older version).",
            {"refused_count": len(refused)},
        )
    if len(low_score) > len(chat_queries) // 3 or len(no_context) > len(chat_queries) // 3:
        return CheckResult(
            "chat_quality",
            "warn",
            f"{len(low_score)} low-confidence grounded answers, {len(no_context)} no-context fallbacks "
            f"of {len(chat_queries)} queries.",
            "Either the sources don't cover the question space, or embeddings are stale. "
            "Consider wiring embed-at-ingestion (deferred work).",
            {"low_score": len(low_score), "no_context": len(no_context)},
        )
    return CheckResult(
        "chat_quality",
        "pass",
        f"{len(chat_queries)} chat queries handled cleanly.",
        "",
    )


def check_synthesis_paraphrase(rows: list[dict[str, Any]], db_engine: Any, run_id: str, *, source_texts: list[str] | None = None, synthesis_text: str | None = None) -> CheckResult:
    """Synthesis should not be a paraphrase of any single source. Only runs
    when the diagnostic explicitly passes in the texts."""
    if not synthesis_text or not source_texts:
        return CheckResult("synthesis_paraphrase", "pass", "(no synthesis to score)", "")

    # Reuse the same overlap intuition the pipeline already encodes.
    def tokens(t: str) -> set[str]:
        return {w for w in re.findall(r"[a-zA-Z]{4,}", t.lower())}

    syn_tokens = tokens(synthesis_text)
    if not syn_tokens:
        return CheckResult("synthesis_paraphrase", "pass", "Synthesis was empty/whitespace.", "")
    overlaps = []
    for idx, src in enumerate(source_texts):
        src_tokens = tokens(src)
        if not src_tokens:
            continue
        jaccard = len(syn_tokens & src_tokens) / max(len(syn_tokens | src_tokens), 1)
        overlaps.append((idx, jaccard))
    if not overlaps:
        return CheckResult("synthesis_paraphrase", "pass", "No source text to compare.", "")
    worst = max(overlaps, key=lambda p: p[1])
    if worst[1] > 0.5:
        return CheckResult(
            "synthesis_paraphrase",
            "warn",
            f"Synthesis overlaps source #{worst[0]} at Jaccard={worst[1]:.2f}.",
            "Add an explicit 'do not restate any single source verbatim' clause to "
            "prompts/synthesis_consolidator.py.",
            {"worst_overlap": worst[1]},
        )
    return CheckResult(
        "synthesis_paraphrase",
        "pass",
        f"Synthesis distinct from sources (max Jaccard={worst[1]:.2f}).",
        "",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


CHECK_FUNCTIONS = [
    check_stage_failures,
    check_retry_pressure,
    check_critic_fallback,
    check_section_novelty,
    check_hard_cuts,
    check_low_mechanism,
    check_token_budget,
    check_prompt_completion_ratio,
    check_chat_quality,
]


def run_checks(
    session_jsonl_path: Path,
    db_engine: Any,
    run_id: str,
    *,
    extra_jsonl_paths: list[Path] | None = None,
    source_texts: list[str] | None = None,
    synthesis_text: str | None = None,
) -> list[CheckResult]:
    rows = _load_jsonl(session_jsonl_path)
    for extra in extra_jsonl_paths or []:
        rows.extend(_load_jsonl(extra))

    results: list[CheckResult] = []
    for func in CHECK_FUNCTIONS:
        try:
            results.append(func(rows, db_engine, run_id))
        except Exception as exc:  # never let a check kill the report
            results.append(
                CheckResult(
                    func.__name__,
                    "warn",
                    f"check raised: {exc!r}",
                    "Inspect scripts/checks.py — a check itself failed.",
                )
            )
    results.append(
        check_synthesis_paraphrase(
            rows,
            db_engine,
            run_id,
            source_texts=source_texts,
            synthesis_text=synthesis_text,
        )
    )
    return results


def suggest_tunes(results: list[CheckResult], top_n: int = 3) -> list[str]:
    """Pick the top N actionable items: failures first, then warns with the
    most concrete remediation hint."""
    failures = [r for r in results if r.status == "fail" and r.remediation_hint]
    warns = [r for r in results if r.status == "warn" and r.remediation_hint]
    ordered = failures + warns
    return [f"[{r.name}] {r.remediation_hint}" for r in ordered[:top_n]]
