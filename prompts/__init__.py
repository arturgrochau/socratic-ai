"""All prompt constants and JSON schemas for the Socratic AI pipeline.

Stages: ledger extraction, consolidation, synthesis, section audit, chat
(+ chat depth classification). Per-source and cross-source section roles, budgets,
and non-overlap rules are declared once in prompts.sections and injected here.
"""
from __future__ import annotations

from prompts.sections import (
    CROSS_SOURCE_SECTIONS,
    PER_SOURCE_SECTIONS,
    render_contracts,
)


_PER_SOURCE_CONTRACTS = render_contracts(PER_SOURCE_SECTIONS)
_CROSS_SOURCE_CONTRACTS = render_contracts(CROSS_SOURCE_SECTIONS)


# ---------------------------------------------------------------------------
# Stage 1: Ledger extraction (per window)
# ---------------------------------------------------------------------------

LEDGER_EXTRACTION_SYSTEM_PROMPT = """\
You extract atomic knowledge units from source material into a structured ledger.

A knowledge unit is ONE self-contained claim. Classify each by type:
  - foundational: the core thesis or a central finding.
  - mechanism: how something works, a causal step, a dependency.
  - tradeoff: a gain weighed against a loss, a comparison of options.
  - assumption: a hidden premise the material rests on.
  - boundary: a condition under which something stops working; a limit.
  - example: a concrete number, case, or scenario from the source.
  - open_question: an unresolved gap, a failure mode, an unanswered question.

RULES:
1. You are given the ledger built from earlier material. Do NOT emit a unit that
   restates a claim already in it. Only emit genuinely new claims from the new
   material. If the new material adds nothing, return an empty list.
2. Each claim is at most two sentences, specific, and self-contained.
3. evidence: a short verbatim phrase from the source that anchors the claim. Do
   not invent evidence; if none fits, use an empty string.
4. Do NOT speculate beyond the source. Do not use em dashes.
5. Prefer fewer, higher-signal units over many shallow restatements.

Return ONLY valid JSON matching the schema."""

LEDGER_EXTRACTION_JSON_SCHEMA = {
    "name": "ledger_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "foundational", "mechanism", "tradeoff",
                                "assumption", "boundary", "example", "open_question",
                            ],
                        },
                        "claim": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["type", "claim", "evidence"],
                },
            },
        },
        "required": ["units"],
    },
}


# ---------------------------------------------------------------------------
# Stage 2: Consolidation (per source) — consumes the typed ledger
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM_PROMPT = f"""\
You organize a typed knowledge ledger into a learning artifact with a strict
cognitive arc. The ledger gives you atomic claims, each tagged with an id, a
type, a source, and supporting evidence. Your job is to ORGANIZE and COMPRESS,
not to re-explain.

SECTION CONTRACTS (each field draws only from the listed unit types):
{_PER_SOURCE_CONTRACTS}

GLOBAL RULES:
1. A claim appears in at most one section. Never restate across fields. Later
   sections build on earlier ones; they do not recap them.
2. Compress. Every sentence must add information not already present. No bridge
   prose ("this highlights the importance of", "in practice", "overall").
3. key_terms: extract real domain terms (not meta-words like "tradeoff"). For
   each: layman (one plain sentence, analogy only if the term is abstract) and
   technical (one precise sentence). At most two sentences total per term.
4. reflection_points: each explanation names the specific reasoning trap (the
   wrong answer a smart person gives) and the correct path. Vary the cognitive
   operation across questions (causal, counterfactual, failure analysis,
   tradeoff, methodological critique). depth_level in foundational/intermediate/
   advanced, at least two advanced.
5. Continuous prose in text fields. No markdown. No bullet lists. No em dashes.

Return ONLY valid JSON matching the schema."""

CONSOLIDATION_JSON_SCHEMA = {
    "name": "source_consolidation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "deep_dive": {"type": "string"},
            "key_terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string"},
                        "layman": {"type": "string"},
                        "technical": {"type": "string"},
                    },
                    "required": ["term", "layman", "technical"],
                },
            },
            "under_surface": {"type": "string"},
            "reflection_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "explanation": {"type": "string"},
                        "depth_level": {
                            "type": "string",
                            "enum": ["foundational", "intermediate", "advanced"],
                        },
                    },
                    "required": ["question", "explanation", "depth_level"],
                },
            },
        },
        "required": [
            "summary",
            "deep_dive",
            "key_terms",
            "under_surface",
            "reflection_points",
        ],
    },
}


# ---------------------------------------------------------------------------
# Stage 3: Synthesis (cross-source) — consumes the merged ledger
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = f"""\
You synthesize a merged, cross-source knowledge ledger into a comparative
artifact. Each ledger unit carries its source id, so you can ground every
comparison in specific claims.

SECTION CONTRACTS:
{_CROSS_SOURCE_CONTRACTS}

GLOBAL RULES:
1. Synthesis means COMPARISON, not summary. Do not summarize each source in
   sequence. Isolate convergences, contradictions, methodological differences,
   shared assumptions, and unresolved gaps.
2. Every intersection MUST cite supporting claims from at least two different
   sources via attributed_sentences (each with the verbatim text and its
   source_id). An intersection without cross-source grounding is invalid; omit
   it rather than fabricate.
3. Do not invent topics absent from the ledger. If the sources share little,
   say so plainly and produce fewer items.
4. questions: write exactly four options each and set answer_index to the
   correct one, varying which index is correct across questions. Questions must
   be HARDBALL: each tests a cross-source mechanism or its application, never
   single-source recall or recognition of the summary. The three wrong options
   must each be a PLAUSIBLE misconception a strong student would actually pick
   (a real confusion grounded in the ledger), not filler or obviously-wrong
   noise. Provide option_explanations with exactly one entry per option,
   index-aligned to options: for the correct option, explain why it is right and
   name the mechanism; for each wrong option, name the specific misconception it
   represents and state what the concept ACTUALLY is (e.g. "No, X is actually
   about ..."). The top-level explanation gives the overall correct reasoning.
   No "Option A" labels anywhere.
5. Continuous prose in text fields. No markdown. No em dashes.

Return ONLY valid JSON matching the schema."""

SYNTHESIS_JSON_SCHEMA = {
    "name": "cross_source_synthesis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "synthesis_text": {"type": "string"},
            "intersections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "integrated_explanation": {"type": "string"},
                        "attributed_sentences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "source_id": {"type": "integer"},
                                },
                                "required": ["text", "source_id"],
                            },
                        },
                    },
                    "required": [
                        "title",
                        "why_it_matters",
                        "integrated_explanation",
                        "attributed_sentences",
                    ],
                },
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "answer_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                        "option_explanations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "question",
                        "options",
                        "answer_index",
                        "explanation",
                        "option_explanations",
                    ],
                },
            },
            "application_scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scenario_title": {"type": "string"},
                        "scenario_prompt": {"type": "string"},
                        "transfer_steps": {"type": "array", "items": {"type": "string"}},
                        "common_pitfall": {"type": "string"},
                    },
                    "required": [
                        "scenario_title",
                        "scenario_prompt",
                        "transfer_steps",
                        "common_pitfall",
                    ],
                },
            },
        },
        "required": [
            "synthesis_text",
            "intersections",
            "questions",
            "application_scenarios",
        ],
    },
}


# ---------------------------------------------------------------------------
# Stage 4: Section audit (cheap, runs on the aux model)
# ---------------------------------------------------------------------------

AUDIT_SYSTEM_PROMPT = """\
You are a strict editor checking one or more generated sections against their
contracts and the knowledge ledger. Each section block lists its role, its
non-overlap rule, its length budget, and its content. Sections are given in arc
order, so earlier sections are the "prior" material that later sections must not
restate. Prefix every violation you report with the offending section's title.

Flag a violation ONLY when it clearly breaks a rule:
  - restatement: the section repeats a claim already shown in an earlier section.
  - off_contract: the section does something its role forbids.
  - over_budget: the section clearly exceeds its length budget.
  - ungrounded: a comparative item lacks support from two different sources.
  - low_diversity: questions mostly test the same shallow operation or recall.

Be conservative: if the section is acceptable, return ok=true and no violations.
Do NOT rewrite the section. Just report.

Return ONLY valid JSON matching the schema."""

AUDIT_JSON_SCHEMA = {
    "name": "section_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "violations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ok", "violations"],
    },
}


# ---------------------------------------------------------------------------
# Stage 5: Chat (depth classification + budget-aware answering)
# ---------------------------------------------------------------------------

# Tier -> (one-line budget instruction, context scale 0..1)
CHAT_DEPTH_BUDGETS: dict[str, tuple[str, float]] = {
    "lookup": ("Answer in at most two sentences. State the fact and stop.", 0.34),
    "explain": ("Answer in one short paragraph (3-5 sentences).", 0.6),
    "analyze": ("Answer in at most three paragraphs of genuine analysis.", 1.0),
    "deepen": (
        "The user wants more depth on the prior answer. Add genuinely new "
        "dimensions (hidden assumptions, failure modes, edge cases) in at most "
        "three paragraphs. Do NOT paraphrase what you already said.",
        1.0,
    ),
}

DEPTH_CLASSIFIER_SYSTEM_PROMPT = """\
Classify the scope of a study question so the answer length matches the question.

Tiers:
  - lookup: a definition or single fact. Tiny answer.
  - explain: "how/why" about one concept. Short answer.
  - analyze: comparison, synthesis, tradeoff, or multi-part reasoning. Fuller answer.
  - deepen: the user asks to go deeper or elaborate on the previous answer.

Use the recent turns to detect deepen requests. Return ONLY valid JSON."""

DEPTH_CLASSIFIER_JSON_SCHEMA = {
    "name": "chat_depth",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["lookup", "explain", "analyze", "deepen"],
            },
        },
        "required": ["tier"],
    },
}

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a study partner helping the user understand their uploaded materials.

RULES:
1. Answer the question directly in the first sentence. No preamble.
2. Ground every claim in the provided context. If the materials do not cover the
   question, say so in one clause and pivot to the closest relevant theme.
3. Be specific. Name mechanisms, constraints, and tradeoffs. Use domain terms
   from the materials.
4. If you draw a connection beyond what the sources state, prefix that sentence
   with "Inferred extension:".
5. No markdown headings or bullet lists. Continuous prose. No em dashes.
6. LENGTH: {budget} Match answer size to question size. Do not pad.

Return ONLY valid JSON matching the schema."""

CHAT_JSON_SCHEMA = {
    "name": "chat_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
    },
}


def build_chat_system_prompt(tier: str) -> str:
    budget, _ = CHAT_DEPTH_BUDGETS.get(tier, CHAT_DEPTH_BUDGETS["explain"])
    return CHAT_SYSTEM_PROMPT_TEMPLATE.format(budget=budget)


def chat_context_scale(tier: str) -> float:
    _, scale = CHAT_DEPTH_BUDGETS.get(tier, CHAT_DEPTH_BUDGETS["explain"])
    return scale
