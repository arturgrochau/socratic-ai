"""
Compare two diagnostic runs and print a delta report.

Usage:
    python scripts/diagnostic_compare.py gen-abc12345 gen-def67890

Reads each run's session JSONL + DB telemetry and prints a side-by-side
table of token costs, check pass/warn/fail counts, deep_dive sizes, and
chat answer lengths. Useful for measuring the effect of a tune across
two runs without burning more API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path when invoked as `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_LOG_DIR = REPO_ROOT / "logs" / "sessions"
DIAGNOSTIC_DIR = REPO_ROOT / "logs" / "diagnostic"


def _load_jsonl(run_id: str) -> list[dict[str, Any]]:
    path = SESSION_LOG_DIR / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No JSONL for {run_id} at {path}")
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


def _find_diagnostic_md(run_id: str) -> Path | None:
    matches = sorted(DIAGNOSTIC_DIR.glob(f"{run_id}-*.md"))
    return matches[0] if matches else None


def _summarize(run_id: str, db_engine: Any) -> dict[str, Any]:
    rows = _load_jsonl(run_id)
    session_start = next((r for r in rows if r.get("stage") == "session_start"), None)
    session_end = next((r for r in rows if r.get("stage") == "session_end"), None)
    if not session_start:
        raise RuntimeError(f"{run_id}: missing session_start record")
    user_id = session_start.get("user_id", "unknown")
    started_at = float(session_start.get("started_at") or 0.0)
    duration = float((session_end or {}).get("duration_s") or 0.0)

    # Cost from DB
    with db_engine.connect() as conn:
        rolled = conn.execute(
            text(
                "SELECT call_stage, model_name, "
                "COALESCE(SUM(total_tokens), 0) AS total, COUNT(*) AS n "
                "FROM api_call_usage WHERE user_id = :u "
                "AND CAST(strftime('%s', created_at) AS INTEGER) >= :since "
                "GROUP BY call_stage, model_name"
            ),
            {"u": user_id, "since": int(max(0.0, started_at - 5))},
        ).mappings().all()
    by_stage = {f"{r['call_stage']}:{r['model_name']}": int(r["total"] or 0) for r in rolled}
    total_tokens = sum(by_stage.values())
    total_calls = sum(int(r["n"] or 0) for r in rolled)

    # Chat answer lengths
    chat_queries = [r for r in rows if r.get("stage") == "chat_query"]
    answer_lengths = {
        f"q{i+1}_route={q.get('route')}": int(q.get("answer_length") or 0)
        for i, q in enumerate(chat_queries)
    }

    # Progressive rounds
    progressive_rounds = [r for r in rows if r.get("stage") == "progressive_round"]

    # Checks from the markdown report
    md_path = _find_diagnostic_md(run_id)
    check_counts = {"pass": 0, "warn": 0, "fail": 0}
    mode = "?"
    if md_path:
        for line in md_path.read_text(encoding="utf-8").splitlines():
            if "✅" in line:
                check_counts["pass"] += 1
            elif "⚠" in line and "|" in line:
                check_counts["warn"] += 1
            elif "✘" in line and "|" in line:
                check_counts["fail"] += 1
            if line.startswith("- **Mode**:"):
                mode = line.split("`")[1] if "`" in line else "?"

    return {
        "run_id": run_id,
        "user_id": user_id,
        "mode": mode,
        "duration_s": duration,
        "total_tokens": total_tokens,
        "total_calls": total_calls,
        "by_stage": by_stage,
        "answer_lengths": answer_lengths,
        "progressive_rounds": len(progressive_rounds),
        "checks": check_counts,
    }


def _delta_pct(a: int, b: int) -> str:
    if a == 0:
        return "—" if b == 0 else "+∞"
    pct = (b - a) / a * 100
    return f"{pct:+.0f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="First run id, e.g. gen-abc12345")
    parser.add_argument("run_b", help="Second run id")
    args = parser.parse_args(argv)

    from config import db_engine
    a = _summarize(args.run_a, db_engine)
    b = _summarize(args.run_b, db_engine)

    print(f"# Diagnostic comparison: {args.run_a} → {args.run_b}\n")
    print(f"| | A ({args.run_a}) | B ({args.run_b}) | Δ |")
    print("|---|---|---|---|")
    print(f"| Mode | {a['mode']} | {b['mode']} | |")
    print(f"| Wall clock | {a['duration_s']:.1f}s | {b['duration_s']:.1f}s | {_delta_pct(int(a['duration_s']), int(b['duration_s']))} |")
    print(f"| Total tokens | {a['total_tokens']:,} | {b['total_tokens']:,} | {_delta_pct(a['total_tokens'], b['total_tokens'])} |")
    print(f"| Total calls | {a['total_calls']} | {b['total_calls']} | {_delta_pct(a['total_calls'], b['total_calls'])} |")
    print(f"| Checks pass/warn/fail | {a['checks']['pass']}/{a['checks']['warn']}/{a['checks']['fail']} | {b['checks']['pass']}/{b['checks']['warn']}/{b['checks']['fail']} | |")
    print(f"| Progressive rounds | {a['progressive_rounds']} | {b['progressive_rounds']} | |")

    # Per-stage cost
    print("\n## Per-stage tokens\n")
    print("| Stage | A | B | Δ |")
    print("|---|---|---|---|")
    all_stages = sorted(set(a["by_stage"]) | set(b["by_stage"]))
    for stage in all_stages:
        va = a["by_stage"].get(stage, 0)
        vb = b["by_stage"].get(stage, 0)
        print(f"| `{stage}` | {va:,} | {vb:,} | {_delta_pct(va, vb)} |")

    # Chat answer lengths — only show if either side has data (chat records
    # live in a separate JSONL keyed by the chat session id, not the run id).
    if a["answer_lengths"] or b["answer_lengths"]:
        print("\n## Chat answer lengths\n")
        print("| Probe | A | B | Δ |")
        print("|---|---|---|---|")
        all_probes = sorted(set(a["answer_lengths"]) | set(b["answer_lengths"]))
        for probe in all_probes:
            la = a["answer_lengths"].get(probe, 0)
            lb = b["answer_lengths"].get(probe, 0)
            print(f"| `{probe}` | {la} | {lb} | {_delta_pct(la, lb)} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
