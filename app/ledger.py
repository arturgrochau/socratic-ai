"""Knowledge Ledger: the structured backbone of the generation pipeline.

Instead of passing a growing prose blob between stages, accumulation emits typed,
source-grounded atomic claims (KnowledgeUnit). Downstream sections consume slices
of the ledger filtered by cognitive role and are told to reference units, never
restate them. This gives three properties by construction:

  * topology   - every unit is typed (foundational / mechanism / tradeoff / ...)
  * grounding  - every unit carries a source_id and a verbatim evidence anchor
  * novelty    - dedup on append means a claim is recorded once and only once
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import text

from app.json_reliability import safe_json_loads
from config import db_engine

UnitType = Literal[
    "foundational",
    "mechanism",
    "tradeoff",
    "assumption",
    "boundary",
    "example",
    "open_question",
]

ALL_UNIT_TYPES: tuple[UnitType, ...] = (
    "foundational",
    "mechanism",
    "tradeoff",
    "assumption",
    "boundary",
    "example",
    "open_question",
)


class KnowledgeUnit(BaseModel):
    id: str
    source_id: int
    type: UnitType
    claim: str
    evidence: str = ""


# ---------------------------------------------------------------------------
# Dedup / merge / slice
# ---------------------------------------------------------------------------


def _signature(text_value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", str(text_value or "").lower())
        if len(token) >= 4
    }


def _overlap_ratio(left: str, right: str) -> float:
    left_sig = _signature(left)
    right_sig = _signature(right)
    if not left_sig or not right_sig:
        return 0.0
    intersection = len(left_sig & right_sig)
    return intersection / min(len(left_sig), len(right_sig))


def is_duplicate(claim: str, existing: Iterable[KnowledgeUnit], *, threshold: float = 0.72) -> bool:
    return any(_overlap_ratio(claim, unit.claim) >= threshold for unit in existing)


def append_units(
    ledger: list[KnowledgeUnit],
    new_units: Iterable[KnowledgeUnit],
) -> list[KnowledgeUnit]:
    """Append new units, skipping near-duplicates of what's already recorded."""
    for unit in new_units:
        claim = unit.claim.strip()
        if not claim:
            continue
        if is_duplicate(claim, ledger):
            continue
        ledger.append(unit)
    return ledger


def units_of_types(
    ledger: Iterable[KnowledgeUnit],
    types: Iterable[UnitType],
) -> list[KnowledgeUnit]:
    wanted = set(types)
    return [unit for unit in ledger if unit.type in wanted]


def next_unit_id(ledger: Iterable[KnowledgeUnit]) -> int:
    """Highest numeric suffix in existing ids + 1 (ids look like 'u17')."""
    highest = 0
    for unit in ledger:
        match = re.search(r"(\d+)$", unit.id)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_units(units: Iterable[KnowledgeUnit], *, include_evidence: bool = True) -> str:
    """Compact, id-addressable rendering for downstream section prompts."""
    lines: list[str] = []
    for unit in units:
        line = f"[{unit.id} | {unit.type} | source {unit.source_id}] {unit.claim.strip()}"
        if include_evidence and unit.evidence.strip():
            line += f"  (evidence: \"{unit.evidence.strip()}\")"
        lines.append(line)
    return "\n".join(lines)


def parse_units(raw: object, source_id: int, start_index: int) -> list[KnowledgeUnit]:
    """Coerce raw extraction payload items into KnowledgeUnits with stable ids."""
    units: list[KnowledgeUnit] = []
    if not isinstance(raw, list):
        return units
    counter = start_index
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        claim = str(entry.get("claim", "")).strip()
        unit_type = str(entry.get("type", "")).strip()
        if not claim or unit_type not in ALL_UNIT_TYPES:
            continue
        units.append(
            KnowledgeUnit(
                id=f"u{counter}",
                source_id=source_id,
                type=unit_type,  # type: ignore[arg-type]
                claim=claim,
                evidence=str(entry.get("evidence", "")).strip(),
            )
        )
        counter += 1
    return units


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def ensure_ledger_table() -> None:
    with db_engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS source_ledger_units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                units_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, source_id)
            )
        """))


def store_ledger(
    *, user_id: str, source_id: int, units: list[KnowledgeUnit], schema_version: int,
) -> None:
    payload = json.dumps([u.model_dump() for u in units], ensure_ascii=True)
    with db_engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO source_ledger_units (user_id, source_id, schema_version, units_json)
                VALUES (:user_id, :source_id, :schema_version, :units_json)
                ON CONFLICT(user_id, source_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    units_json = excluded.units_json,
                    updated_at = CURRENT_TIMESTAMP
            """),
            {
                "user_id": user_id,
                "source_id": source_id,
                "schema_version": schema_version,
                "units_json": payload,
            },
        )


def load_ledger(
    *, user_id: str, source_id: int, schema_version: int,
) -> list[KnowledgeUnit] | None:
    with db_engine.connect() as connection:
        row = connection.execute(
            text("""
                SELECT units_json, schema_version
                FROM source_ledger_units
                WHERE user_id = :user_id AND source_id = :source_id
                LIMIT 1
            """),
            {"user_id": user_id, "source_id": source_id},
        ).mappings().first()
    if row is None or int(row.get("schema_version") or 0) != schema_version:
        return None
    units: list[KnowledgeUnit] = []
    for entry in safe_json_loads(row.get("units_json"), default=[]):
        try:
            units.append(KnowledgeUnit.model_validate(entry))
        except Exception:
            continue
    return units or None
