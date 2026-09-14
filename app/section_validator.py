"""Pure-code section validation — replaces the former LLM audit pass.

The old `_audit_arc` paid an aux-model call to check things a model is not
needed for: length budgets (sentence/item counts) and restatement (token
overlap between sections, the same measure the ledger uses for dedup). This
module computes those checks in microseconds instead. Violations feed the
existing one-regen loop in app/generation.py unchanged.

Deliberately NOT checked here (and accepted): fuzzy "off-contract" and
"low-diversity" judgments — the contracts remain in the generation prompt
itself, which is where they bind.
"""
from __future__ import annotations

import re
from typing import Any

from app.ledger import _overlap_ratio
from prompts.sections import SectionSpec

# Whole-section token overlap above this against any earlier section in the arc
# counts as restatement. Higher than the ledger's 0.72 claim-level threshold
# because full sections legitimately share topic vocabulary.
RESTATEMENT_OVERLAP_THRESHOLD = 0.60
# Grace margin on sentence ceilings: prompts say "at most N", we flag at N + 2
# so a borderline-but-fine output doesn't trigger a pointless regen.
SENTENCE_GRACE = 2


def _count_sentences(text_value: str) -> int:
    normalized = " ".join(str(text_value or "").split())
    if not normalized:
        return 0
    return len([s for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()])


def _render_field(value: Any) -> str:
    """Flatten a section value (str or list of dicts/strs) into comparable text."""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return str(value or "")
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            parts.append(" ".join(str(v) for v in item.values() if isinstance(v, str)))
        else:
            parts.append(str(item))
    return "\n".join(p for p in parts if p.strip())


def validate_arc(specs: tuple[SectionSpec, ...], structured: dict[str, Any]) -> list[str]:
    """Check every section against its machine-checkable limits.

    Returns violations formatted like the old audit output ("<Title>: ...") so
    the regen path in generation.py consumes them unchanged.
    """
    violations: list[str] = []
    earlier: list[tuple[str, str]] = []  # (title, rendered text) in arc order

    for spec in specs:
        # Composite fields like "synthesis_text+intersections" validate the first part.
        field = spec.field.split("+")[0]
        raw = structured.get(field)
        rendered = _render_field(raw)
        if not rendered.strip():
            continue

        if spec.max_sentences is not None and isinstance(raw, str):
            count = _count_sentences(raw)
            if count > spec.max_sentences + SENTENCE_GRACE:
                violations.append(
                    f"{spec.title}: over budget — {count} sentences against a budget of "
                    f"{spec.budget!r}. Tighten to the budget; cut the least important points."
                )

        if spec.max_items is not None and isinstance(raw, list) and len(raw) > spec.max_items:
            violations.append(
                f"{spec.title}: over budget — {len(raw)} items against a maximum of "
                f"{spec.max_items}. Keep only the strongest {spec.max_items}."
            )

        if spec.check_restatement:
            for earlier_title, earlier_text in earlier:
                if _overlap_ratio(rendered, earlier_text) >= RESTATEMENT_OVERLAP_THRESHOLD:
                    violations.append(
                        f"{spec.title}: restates material already covered in "
                        f"'{earlier_title}'. Replace the overlapping content with "
                        "genuinely new points or omit it."
                    )
                    break

        earlier.append((spec.title, rendered))

    return violations
