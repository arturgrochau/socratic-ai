"""
Regression suite — runs the diagnostic across all canonical fixture shapes
and produces one aggregate report.

Fixture shapes covered:
  1. Single small PDF (the original test fixture)
  2. Single long text (consensus protocols)
  3. Two long texts (consensus + memory) — exercises documents-only synthesis
  4. Video + small PDF — exercises Whisper + cross-source synthesis
  5. Single long text + cache validation

Usage:
    python scripts/regression_suite.py
    python scripts/regression_suite.py --skip-video  # skip the video fixture (saves Whisper cost)
"""
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIAGNOSTIC = REPO_ROOT / "scripts" / "run_diagnostic.py"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass
class FixtureRun:
    label: str
    args: list[str]
    description: str


SUITE: list[FixtureRun] = [
    FixtureRun(
        label="small_pdf",
        args=["--fixture", "test_assets/sample_notes.pdf"],
        description="Single small PDF (~1.3KB) — baseline non-progressive path",
    ),
    FixtureRun(
        label="long_text",
        args=["--fixture", "test_assets/long_notes_consensus.txt"],
        description="Single long text (~11.5KB) — progressive should fire",
    ),
    FixtureRun(
        label="two_long",
        args=[
            "--fixture", "test_assets/long_notes_consensus.txt",
            "--fixture", "test_assets/long_notes_memory.txt",
        ],
        description="Two long texts — documents-only multi-source synthesis",
    ),
    FixtureRun(
        label="video_plus_pdf",
        args=["--fixture", "test_assets/sample_notes.pdf",
              "--video", "test_assets/sample_video.mp4"],
        description="Video + PDF — Whisper + cross-source synthesis",
    ),
    FixtureRun(
        label="long_with_cache",
        args=["--fixture", "test_assets/long_notes_stress_thermodynamics.txt",
              "--validate-cache"],
        description="Long text with cache validation",
    ),
]


def _run_one(fr: FixtureRun) -> tuple[int, int, int, int, float, str]:
    """Run one fixture and return (pass, warn, fail, exit_code, duration, report_path)."""
    t0 = time.time()
    cmd = [str(PYTHON_BIN), str(RUN_DIAGNOSTIC), *fr.args]
    print(f"\n--- [{fr.label}] {fr.description}")
    print(f"    cmd: {' '.join(fr.args)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    duration = time.time() - t0
    out = proc.stdout
    # Parse "Result: X fail, Y warn, Z pass" line.
    p, w, f = 0, 0, 0
    report_path = ""
    for line in out.splitlines():
        if "Result:" in line:
            try:
                bits = line.split("Result:", 1)[1]
                parts = [b.strip() for b in bits.split(",")]
                for b in parts:
                    n, label = b.split(" ", 1)
                    if "pass" in label:
                        p = int(n)
                    elif "warn" in label:
                        w = int(n)
                    elif "fail" in label:
                        f = int(n)
            except Exception:
                pass
        if "Report written to" in line:
            report_path = line.rsplit(" ", 1)[-1]
    return p, w, f, proc.returncode, duration, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-video", action="store_true", help="Skip the video fixture (saves Whisper API cost)")
    parser.add_argument("--only", action="append", default=[], help="Only run fixtures with these labels (repeatable)")
    parser.add_argument("--judge-chat", action="store_true", help="Pass --judge-chat to each fixture run (adds ~5 small LLM calls per fixture)")
    args = parser.parse_args(argv)
    if args.judge_chat:
        for fr in SUITE:
            fr.args.append("--judge-chat")

    suite = SUITE
    if args.skip_video:
        suite = [f for f in suite if "video" not in f.label]
    if args.only:
        wanted = set(args.only)
        suite = [f for f in suite if f.label in wanted]

    print(f"=== Regression suite: {len(suite)} fixtures ===")
    results = []
    total_t0 = time.time()
    for fr in suite:
        p, w, f, code, duration, report = _run_one(fr)
        results.append((fr.label, p, w, f, code, duration, report))
    total_duration = time.time() - total_t0

    print("\n\n=== AGGREGATE ===")
    print("\n| Fixture | Pass | Warn | Fail | Wall (s) | Report |")
    print("|---|---|---|---|---|---|")
    total_pass = total_warn = total_fail = 0
    bad_codes = 0
    for label, p, w, f, code, dur, report in results:
        marker = "✘" if (f > 0 or code != 0) else ("⚠" if w > 0 else "✅")
        total_pass += p
        total_warn += w
        total_fail += f
        if code != 0:
            bad_codes += 1
        print(f"| {marker} `{label}` | {p} | {w} | {f} | {dur:.0f} | `{Path(report).name if report else '?'}` |")
    print(f"\nTotal: {total_pass} pass, {total_warn} warn, {total_fail} fail across {len(results)} fixtures")
    print(f"Bad exit codes: {bad_codes}")
    print(f"Wall clock: {total_duration:.0f}s")
    return 1 if (total_fail > 0 or bad_codes > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
