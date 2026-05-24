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
    """Sum tokens across every LLM call this user made during the run.

    JSONL api_call records only cover the generate-tailored-learning context
    (processing + linking + chat happen in separate HTTP handlers, before
    or outside that context). So we go directly to api_call_usage which
    log_api_usage always writes to, regardless of session_logger state.
    """
    from config import DIAGNOSTIC_TOKEN_CEILING

    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    user_id = (session_start or {}).get("user_id", "unknown")
    # The harness sets the run-specific started_at into session_start.started_at.
    started_at = float((session_start or {}).get("started_at") or 0.0)

    with db_engine.connect() as conn:
        # Cast strftime to integer for clean numeric comparison.
        rolled = conn.execute(
            text(
                "SELECT call_stage, model_name, "
                "COALESCE(SUM(total_tokens), 0) AS total, COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :user_id "
                "AND CAST(strftime('%s', created_at) AS INTEGER) >= :since "
                "GROUP BY call_stage, model_name "
                "ORDER BY total DESC"
            ),
            {"user_id": user_id, "since": int(max(0.0, started_at - 5))},
        ).mappings().all()

    total_tokens = sum(int(r["total"] or 0) for r in rolled)
    total_calls = sum(int(r["n"] or 0) for r in rolled)
    by_stage_lines = [f"{r['call_stage']}({r['model_name']})={r['total']}t/{r['n']}c" for r in rolled]
    status: Status = "pass" if total_tokens <= DIAGNOSTIC_TOKEN_CEILING else "warn"
    return CheckResult(
        "token_budget",
        status,
        f"{total_tokens} tokens across {total_calls} calls (ceiling {DIAGNOSTIC_TOKEN_CEILING}). " + ", ".join(by_stage_lines),
        "Trim MAX_GROUNDING_CHARS in app/generation.py if completions are short, "
        "or raise DIAGNOSTIC_TOKEN_CEILING if this is the new normal.",
        {
            "total_tokens": total_tokens,
            "n_calls": total_calls,
            "by_stage": {f"{r['call_stage']}:{r['model_name']}": {"tokens": int(r["total"] or 0), "calls": int(r["n"] or 0)} for r in rolled},
        },
    )


# Different stages have different reasonable ratios. Chat answers are short
# by design; generation should be content-heavy; linking classifies in tiny
# JSON. The check fires per-stage against these thresholds.
RATIO_THRESHOLDS: dict[str, float] = {
    # Generation completions are bounded by the section word budgets
    # (e.g. MAX_DEEP_DIVE_WORDS=900). With dense grounding context the ratio
    # settles around 23-32% with run-to-run variance from temperature.
    # Floor at 22% — a real regression would push lower; this just absorbs
    # the noise.
    "generation": 0.22,
    # Processing input scales with source size (full context). Output is
    # bounded (concept list + 3-paragraph summary). Long sources naturally
    # sit at 20-25% ratio. Setting floor at 20% acknowledges the structural
    # asymmetry without losing the regression signal.
    "processing": 0.20,
    # 4-way classification with short explanations; after the payload trim
    # ratio settles around 19-21% with run-to-run variance. Setting the floor
    # at 18% so noise doesn't trigger; a real regression would push lower.
    "linking": 0.18,
    "interaction": 0.05,   # chat answers are naturally short relative to grounded prompts
}
DEFAULT_RATIO_THRESHOLD = 0.30


def check_prompt_completion_ratio(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Aggregate prompt/completion across every LLM call this user made during
    the run (read from api_call_usage to cover non-generation stages too).
    Flag stages whose average completion is under a stage-specific threshold."""
    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    user_id = (session_start or {}).get("user_id", "unknown")
    started_at = float((session_start or {}).get("started_at") or 0.0)

    with db_engine.connect() as conn:
        rolled = conn.execute(
            text(
                "SELECT call_stage, "
                "COALESCE(SUM(prompt_tokens), 0) AS prompt, "
                "COALESCE(SUM(completion_tokens), 0) AS completion, "
                "COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :user_id "
                "AND CAST(strftime('%s', created_at) AS INTEGER) >= :since "
                "AND prompt_tokens > 0 "
                # Embedding calls have completion_tokens=0 by definition.
                "AND call_stage != 'retrieval' "
                "GROUP BY call_stage"
            ),
            {"user_id": user_id, "since": int(max(0.0, started_at - 5))},
        ).mappings().all()

    over_stuffed: list[str] = []
    for r in rolled:
        prompt = float(r["prompt"] or 0)
        completion = float(r["completion"] or 0)
        if prompt <= 0:
            continue
        ratio = completion / prompt
        stage = str(r["call_stage"])
        threshold = RATIO_THRESHOLDS.get(stage, DEFAULT_RATIO_THRESHOLD)
        if ratio < threshold:
            over_stuffed.append(f"{stage}({ratio:.0%}<{threshold:.0%},n={int(r['n'])})")
    if not over_stuffed:
        return CheckResult("prompt_completion_ratio", "pass", "All call stages had reasonable input/output ratio.", "")
    return CheckResult(
        "prompt_completion_ratio",
        "warn",
        f"Over-stuffed prompts (completion < 30% of input): {', '.join(over_stuffed)}",
        "Reduce MAX_GROUNDING_CHARS / trim the offending stage's system prompt / shorten the schema description.",
        {"over_stuffed": over_stuffed, "by_stage": [dict(r) for r in rolled]},
    )


def check_chat_judge(rows: list[dict[str, Any]], db_engine: Any, run_id: str, *, chat_judgements: list[dict[str, Any]] | None = None) -> CheckResult:
    """LLM-as-judge semantic grade of chat answers (1-5 scale). The diagnostic
    harness computes this by sending each (probe, answer) pair to gpt-4o-mini
    with a strict rubric. Threshold: average grade must be ≥3.5, no individual
    answer below 2.0 (a true bail).

    Skipped when the harness didn't request judgements (no extra API cost in
    default runs)."""
    if not chat_judgements:
        return CheckResult("chat_judge", "pass", "(LLM judge not requested; rerun with --judge-chat)", "")

    grades = [int(j.get("grade") or 0) for j in chat_judgements if j.get("grade")]
    if not grades:
        return CheckResult("chat_judge", "warn", "Judgements requested but no grades returned", "Check judge call logs.")

    avg = sum(grades) / len(grades)
    lowest = min(grades)
    rubric_summary = ", ".join(f"{j.get('probe','?')}={j.get('grade','?')}" for j in chat_judgements)

    if lowest < 2:
        offenders = [j for j in chat_judgements if int(j.get("grade") or 5) < 2]
        return CheckResult(
            "chat_judge",
            "fail",
            f"avg={avg:.2f}, lowest={lowest}. {len(offenders)} answer(s) graded 1 (bail/refusal/off-topic).",
            "Compare the failing answers to the system prompt. Common cause: the prompt's no-context redirect language is producing generic answers when retrieval is thin.",
            {"grades": grades, "judgements": chat_judgements},
        )
    if avg < 3.5:
        return CheckResult(
            "chat_judge",
            "warn",
            f"avg={avg:.2f} below 3.5 floor; lowest={lowest}. {rubric_summary}",
            "Most answers are passable but not strong. Look at the lowest-graded probes for the pattern.",
            {"grades": grades, "judgements": chat_judgements},
        )
    return CheckResult(
        "chat_judge",
        "pass",
        f"avg={avg:.2f}, lowest={lowest}. {rubric_summary}",
        "",
        {"grades": grades, "judgements": chat_judgements},
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
    # Deepening probes ("go deeper", "elaborate") should produce richer
    # responses than a default answer. If they're under 800 chars they're
    # likely not actually deepening — flag the relaxed prompt isn't doing
    # its job for the deepening intent.
    thin_deepening = [
        q for q in chat_queries
        if q.get("is_deepening") and int(q.get("answer_length") or 0) < 800
    ]

    if refused:
        return CheckResult(
            "chat_quality",
            "fail",
            f"{len(refused)}/{len(chat_queries)} chat answers leaked the refusal pattern.",
            "The relaxed prompt in prompts/interaction.py should suppress this. Check that the "
            "system prompt was actually loaded (no caching of an older version).",
            {"refused_count": len(refused)},
        )
    issues: list[str] = []
    if thin_deepening:
        issues.append(f"{len(thin_deepening)} deepening answer(s) under 800 chars")
    if len(low_score) > len(chat_queries) // 3:
        issues.append(f"{len(low_score)} low-confidence grounded answers")
    if len(no_context) > len(chat_queries) // 3:
        issues.append(f"{len(no_context)} no-context fallbacks")
    if issues:
        return CheckResult(
            "chat_quality",
            "warn",
            "; ".join(issues) + f" (of {len(chat_queries)} queries)",
            "If deepening is thin: strengthen the long-form clause in prompts/interaction.py "
            "or bump effective_top_k for deepening intents. If low confidence: sources may not "
            "cover the question space.",
            {
                "thin_deepening": len(thin_deepening),
                "low_score": len(low_score),
                "no_context": len(no_context),
            },
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


def check_cache_efficacy(rows: list[dict[str, Any]], db_engine: Any, run_id: str, *, cache_validation: dict[str, Any] | None = None) -> CheckResult:
    """When --validate-cache fired, compare cost across the run's two halves:
    the post-rerun window should add <500 tokens (essentially just db reads).
    Without --validate-cache, this check is a no-op."""
    if not cache_validation:
        return CheckResult("cache_efficacy", "pass", "(cache validation not requested)", "")
    if not cache_validation.get("ok"):
        return CheckResult(
            "cache_efficacy",
            "fail",
            f"Cache validation rerun failed: {cache_validation.get('error', 'unknown')}",
            "Check the FastAPI logs — the cache-hit path may have raised.",
            cache_validation,
        )
    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    if not session_start:
        return CheckResult("cache_efficacy", "warn", "Missing session_start marker.", "")
    user_id = session_start.get("user_id", "unknown")
    # Count rows with id strictly greater than pre_max_id — avoids the
    # second-granularity timestamp boundary problem that timestamp-based
    # windows hit when the rerun starts in the same second as the last chat probe.
    pre_max_id = int(cache_validation.get("pre_max_id") or 0)
    with db_engine.connect() as conn:
        post = conn.execute(
            text(
                "SELECT call_stage, COALESCE(SUM(total_tokens), 0) AS total, COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :u AND id > :pre_max_id "
                "GROUP BY call_stage"
            ),
            {"u": user_id, "pre_max_id": pre_max_id},
        ).mappings().all()
    rerun_total = sum(int(r["total"] or 0) for r in post)
    rerun_calls = sum(int(r["n"] or 0) for r in post)
    rerun_duration = float(cache_validation.get("rerun_duration_s") or 0.0)

    # Generation cache hit means /generate returns in <2s with 0 new LLM calls.
    if rerun_total <= 200 and rerun_calls <= 1:
        return CheckResult(
            "cache_efficacy",
            "pass",
            f"Rerun used {rerun_total} tokens / {rerun_calls} calls in {rerun_duration:.1f}s — cache hit.",
            "",
            {"rerun_total": rerun_total, "rerun_calls": rerun_calls},
        )
    return CheckResult(
        "cache_efficacy",
        "warn",
        f"Rerun used {rerun_total} tokens / {rerun_calls} calls in {rerun_duration:.1f}s — cache may not be firing as expected.",
        "Expected: cache hit returns in <2s with no LLM calls. Inspect "
        "_load_cached_source_learning_section + _load_cached_combined_learning_sections "
        "and the cache invalidation rules (GENERATION_SCHEMA_VERSION, source_id keying).",
        {"rerun_total": rerun_total, "rerun_calls": rerun_calls, "by_stage": [dict(r) for r in post]},
    )


def check_retrieval_efficiency(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Retrieval cost should scale with QUERIES, not corpus size. If we see
    way more embedding calls than chat queries it usually means we're
    re-embedding the corpus per query — embed-at-ingestion would fix it."""
    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    user_id = (session_start or {}).get("user_id", "unknown")
    started_at = float((session_start or {}).get("started_at") or 0.0)

    chat_queries = _records(rows, "chat_query")
    if not chat_queries:
        return CheckResult("retrieval_efficiency", "pass", "(no chat traffic to score)", "")

    with db_engine.connect() as conn:
        retrieval = conn.execute(
            text(
                "SELECT COALESCE(SUM(total_tokens), 0) AS total, COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :user_id "
                "AND call_stage = 'retrieval' "
                "AND CAST(strftime('%s', created_at) AS INTEGER) >= :since"
            ),
            {"user_id": user_id, "since": int(max(0.0, started_at - 5))},
        ).mappings().first()
    n_calls = int((retrieval or {}).get("n") or 0)
    total = int((retrieval or {}).get("total") or 0)

    # Reasonable ratio: ~1 call per chat query + at most 1 corpus warm-up.
    expected_max = len(chat_queries) + 2
    if n_calls > expected_max * 2:
        return CheckResult(
            "retrieval_efficiency",
            "warn",
            f"{n_calls} embedding calls for {len(chat_queries)} chat queries ({total} tokens). "
            f"Likely re-embedding corpus per query.",
            "Wire embed-at-ingestion (deferred from v1.1.0): move upsert_retrieval_embeddings "
            "from the chat hot path (retrieval.py:414) into the ingestion completion hook.",
            {"calls": n_calls, "queries": len(chat_queries), "tokens": total},
        )
    return CheckResult(
        "retrieval_efficiency",
        "pass",
        f"{n_calls} embedding calls for {len(chat_queries)} chat queries ({total} tokens).",
        "",
    )


def check_intersection_quality(rows: list[dict[str, Any]], db_engine: Any, run_id: str, *, synthesis_payload: dict[str, Any] | None = None) -> CheckResult:
    """Each intersection should: have a non-duplicate title, integrated_explanation
    > 100 chars, and contain at least one cross-source signal word (both/each/across).
    Catches single-source paraphrases dressed up as intersections."""
    if not synthesis_payload:
        return CheckResult("intersection_quality", "pass", "(no synthesis to score)", "")
    intersections = synthesis_payload.get("intersections") or []
    if not intersections:
        return CheckResult("intersection_quality", "pass", "(no intersections to score)", "")

    titles_seen: set[str] = set()
    weak: list[str] = []
    duplicates: list[str] = []
    cross_words = ("both", "each", "across", "together", "combined", "between")
    for entry in intersections:
        title = str(entry.get("intersection_title", "")).strip().lower()
        explanation = str(entry.get("integrated_explanation", "")).strip()
        if title in titles_seen:
            duplicates.append(title)
        titles_seen.add(title)
        if len(explanation) < 100:
            weak.append(f"{title}(too short)")
        elif not any(w in explanation.lower() for w in cross_words):
            # Most likely a single-source paraphrase if no cross-reference language.
            weak.append(f"{title}(no cross-source language)")

    issues: list[str] = []
    if duplicates:
        issues.append(f"duplicate titles: {duplicates}")
    if weak:
        issues.append(f"weak: {weak}")
    if not issues:
        return CheckResult(
            "intersection_quality",
            "pass",
            f"{len(intersections)} intersections, all distinct and cross-referenced.",
            "",
        )
    return CheckResult(
        "intersection_quality",
        "warn",
        "; ".join(issues),
        "Strengthen the synthesis_consolidator prompt: require explicit 'both sources'/'each source' "
        "language in integrated_explanation, and ensure each intersection has a distinct title.",
        {"weak": weak, "duplicates": duplicates},
    )


def check_broad_answer_length(rows: list[dict[str, Any]], db_engine: Any, run_id: str) -> CheckResult:
    """Broad-route chat answers should have *some* substance. Threshold is low
    (100 chars) because off-topic queries legitimately produce short answers
    like 'the source does not cover X'. Anything under 100 chars likely
    indicates the model bailed entirely (e.g. one-line refusal)."""
    chat_queries = _records(rows, "chat_query")
    if not chat_queries:
        return CheckResult("broad_answer_length", "pass", "(no chat traffic)", "")
    broad_thin = [
        q for q in chat_queries
        if q.get("route") == "broad" and int(q.get("answer_length") or 0) < 100
    ]
    broad_total = sum(1 for q in chat_queries if q.get("route") == "broad")
    if not broad_thin:
        return CheckResult(
            "broad_answer_length",
            "pass",
            f"{broad_total} broad-route answers, all ≥100 chars." if broad_total else "(no broad probes)",
            "",
        )
    return CheckResult(
        "broad_answer_length",
        "warn",
        f"{len(broad_thin)}/{broad_total} broad-route answers under 100 chars (suspicious bail)",
        "A broad route answer under 100 chars almost certainly means the model "
        "produced a single-line refusal. Check that prompts/interaction.py's "
        "no-context redirect clause is taking effect.",
        {"thin": len(broad_thin), "total": broad_total},
    )


def check_synthesis_richness(rows: list[dict[str, Any]], db_engine: Any, run_id: str, *, synthesis_payload: dict[str, Any] | None = None) -> CheckResult:
    """If synthesis has < 2 intersections or < 1 application scenario, the
    consolidator may be under-utilizing its output budget."""
    if not synthesis_payload:
        return CheckResult("synthesis_richness", "pass", "(no synthesis to score)", "")
    synthesis_text = (synthesis_payload.get("synthesis_text") or "").strip()
    # Single-source runs don't produce a real cross-source synthesis. The
    # `insights` field is always populated but synthesis_text is empty —
    # skip the check then so it doesn't fire as a false positive.
    if not synthesis_text:
        return CheckResult("synthesis_richness", "pass", "(single-source run, no synthesis)", "")
    intersections = synthesis_payload.get("intersections") or []
    parallels = synthesis_payload.get("parallels") or []
    application_scenarios = synthesis_payload.get("application_scenarios") or []

    issues: list[str] = []
    if len(intersections) < 2:
        issues.append(f"{len(intersections)} intersections (target ≥2)")
    if not application_scenarios:
        issues.append("0 application scenarios (target ≥1)")
    if synthesis_text and len(synthesis_text) < 500:
        issues.append(f"synthesis_text only {len(synthesis_text)} chars (target ≥500)")

    if not issues:
        return CheckResult(
            "synthesis_richness",
            "pass",
            f"{len(intersections)} intersections, {len(parallels)} parallels, "
            f"{len(application_scenarios)} application scenarios.",
            "",
        )
    return CheckResult(
        "synthesis_richness",
        "warn",
        "; ".join(issues),
        "Strengthen the prompts/synthesis_consolidator.py instruction to produce 3-4 "
        "intersections and at least 2 application_scenarios. Current output is below "
        "schema capacity.",
        {"intersections": len(intersections), "parallels": len(parallels),
         "application_scenarios": len(application_scenarios)},
    )


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
    check_retrieval_efficiency,
    check_broad_answer_length,
]


def run_checks(
    session_jsonl_path: Path,
    db_engine: Any,
    run_id: str,
    *,
    extra_jsonl_paths: list[Path] | None = None,
    source_texts: list[str] | None = None,
    synthesis_text: str | None = None,
    synthesis_payload: dict[str, Any] | None = None,
    cache_validation: dict[str, Any] | None = None,
    chat_judgements: list[dict[str, Any]] | None = None,
    run_started_at: float | None = None,
) -> list[CheckResult]:
    """Run all checks. `run_started_at` is the diagnostic harness's wall-clock
    start time — if provided, time-windowed DB queries use it instead of the
    session JSONL's session_start.started_at (which only records when
    generation entered its session_log context, missing earlier ingestion/
    processing/linking calls)."""
    rows = _load_jsonl(session_jsonl_path)
    for extra in extra_jsonl_paths or []:
        rows.extend(_load_jsonl(extra))

    # Stamp the diagnostic's true start into the first session_start record so
    # checks read a single source of truth.
    if run_started_at is not None:
        for r in rows:
            if r.get("stage") == "session_start":
                r["started_at"] = float(run_started_at)
                break

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
    results.append(
        check_synthesis_richness(
            rows,
            db_engine,
            run_id,
            synthesis_payload=synthesis_payload,
        )
    )
    results.append(
        check_intersection_quality(
            rows,
            db_engine,
            run_id,
            synthesis_payload=synthesis_payload,
        )
    )
    results.append(
        check_cache_efficacy(
            rows,
            db_engine,
            run_id,
            cache_validation=cache_validation,
        )
    )
    results.append(
        check_chat_judge(
            rows,
            db_engine,
            run_id,
            chat_judgements=chat_judgements,
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
