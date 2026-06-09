"""All prompt constants and JSON schemas for the Socratic AI pipeline.

Stages: ledger extraction (parallel map), consolidation, insights + quiz
(parallel pair), chat. Per-source and cross-source section roles, budgets, and
non-overlap rules are declared once in prompts.sections and injected here;
budget/restatement enforcement is pure code (app/section_validator.py).
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

# Context-free by design: windows are extracted in PARALLEL (map step), so no
# call sees the ledger built from other windows. Cross-window duplicates are
# removed by the pure-code dedup merge in app/ledger.py (reduce step). This is
# the map-reduce pattern; it replaced the sequential each-call-sees-the-full-
# prior-ledger design, whose token cost grew quadratically with source length.
LEDGER_EXTRACTION_SYSTEM_PROMPT = """\
You extract atomic knowledge units from a section of source material into a
structured ledger.

A knowledge unit is ONE self-contained claim. Classify each by type:
  - foundational: the core thesis or a central finding.
  - mechanism: how something works, a causal step, a dependency.
  - tradeoff: a gain weighed against a loss, a comparison of options.
  - assumption: a hidden premise the material rests on.
  - boundary: a condition under which something stops working; a limit.
  - example: a concrete number, case, or scenario from the source.
  - open_question: an unresolved gap, a failure mode, an unanswered question.

RULES:
1. Each claim is at most two sentences, specific, and self-contained.
2. Emit each distinct claim ONCE; do not emit near-duplicate phrasings of the
   same point. If the section teaches nothing substantive, return an empty list.
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
1. BE SELECTIVE. Most educational value comes from a few important ideas, not
   from covering everything. Within each section keep only the points that
   genuinely matter and cut the rest. Shorter and sharper beats longer.
2. A claim appears in at most one section. Never restate across fields. Later
   sections build on earlier ones; they do not recap them.
3. BE DIRECT. State the concrete observation in plain words. No abstract labels
   standing in for the actual point, no bridge prose ("this highlights the
   importance of", "in practice", "overall"), no warm-up sentences.
4. Distinguish fact from inference: state what the source says plainly; when you
   draw a conclusion the source does not state, mark it as your inference.
5. key_terms: only concepts the source actually teaches. For each: layman (one
   plain sentence, analogy only if abstract) and technical (one precise
   sentence). At most two sentences total per term.
6. reflection_points: each explanation names the specific reasoning trap (the
   wrong answer a smart person gives) and the correct path. Questions are
   concrete and specific to this source; do NOT reuse template stems like "What
   if X changed?" or "What assumptions underlie X?". Vary the cognitive operation
   (causal, counterfactual, failure analysis, tradeoff, methodological critique).
   depth_level in foundational/intermediate/advanced, at least two advanced.
7. Never write ledger ids (like u4), "source <number>", or "(evidence: ...)" in
   any text field; that scaffolding is for your reasoning only. Refer to the
   material as "the source".
8. Continuous prose in text fields. No markdown. No bullet lists. No em dashes.

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
# Stage 3: Insights + Quiz (two focused calls, run in PARALLEL)
# ---------------------------------------------------------------------------
# The former single synthesis mega-prompt asked a mini model for five different
# artifacts at once. Splitting the quiz into its own focused call measurably
# improves question quality and costs no wall-clock because both calls run
# concurrently against the same rendered ledger.

INSIGHTS_SYSTEM_PROMPT = f"""\
You synthesize a merged, cross-source knowledge ledger into a comparative
artifact. Each ledger unit carries its source id, so you can ground every
comparison in specific claims.

SECTION CONTRACTS:
{_CROSS_SOURCE_CONTRACTS}

GLOBAL RULES:
1. LEAD WITH HIERARCHY. key_takeaways: the 1-3 most important things the learner
   should leave with, most important first, each one sentence. These are the
   headline; everything else is detail.
2. BE SELECTIVE AND DIRECT. Synthesis means COMPARISON, not summary, and only the
   comparisons that genuinely matter. Open synthesis_text with the single most
   important relationship, stated concretely. Do NOT begin by restating that the
   sources are similar or different, and do NOT follow the template
   "they agree -> they differ -> limitations -> conclusion". State observations
   in plain words, not abstract labels. Cut anything that is not among the few
   most significant points.
3. Every intersection MUST cite supporting claims from at least two different
   sources via attributed_sentences (each with the verbatim text and its
   source_id). An intersection without cross-source grounding is invalid; omit
   it rather than fabricate.
4. Distinguish fact from inference: a synthesized connection is YOUR inference,
   not something either source states. Word it as such ("together these imply")
   rather than asserting it as a source fact.
5. Refer to each source by its name from the SOURCE LEGEND in the user message.
   Never write "source <number>", ledger ids (u4), or "(evidence: ...)" in any
   text field.
6. Do not invent topics absent from the ledger. If the sources share little,
   say so in one sentence and produce fewer items.
7. Continuous prose in text fields. No markdown. No em dashes.

Return ONLY valid JSON matching the schema."""

INSIGHTS_JSON_SCHEMA = {
    "name": "synthesis_insights",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "key_takeaways": {"type": "array", "items": {"type": "string"}},
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
            "key_takeaways",
            "synthesis_text",
            "intersections",
            "application_scenarios",
        ],
    },
}

QUIZ_SYSTEM_PROMPT = """\
You write a short, hard multiple-choice quiz from a knowledge ledger. The quiz
tests genuine understanding of the material's mechanisms, tradeoffs, and limits,
never rote recall.

RULES:
1. 3-4 questions, exactly four options each. Set answer_index to the correct
   option and VARY which index is correct across questions.
2. If the user message says the ledger spans MULTIPLE sources, every question
   must test the relationship between them (resolving a tension, applying the
   combined insight); if it is a SINGLE source, test that source's mechanisms,
   tradeoffs, and boundaries.
3. Reject any question answerable from general domain knowledge without the
   material, and any with a single obviously-correct option.
4. The three wrong options must each be a PLAUSIBLE position a strong student
   would actually pick: a real misconception or partial truth grounded in the
   material, not filler.
5. option_explanations: exactly one entry per option, index-aligned to options.
   For the correct option, explain why it is right and name the mechanism. For
   each wrong option, name the specific misconception it represents and state
   what the concept ACTUALLY is (e.g. "No, X is actually about ..."). The
   top-level explanation gives the overall correct reasoning.
6. No "Option A" labels anywhere. Refer to sources by name from the SOURCE
   LEGEND if given; never "source <number>" or ledger ids. No em dashes.

Return ONLY valid JSON matching the schema."""

QUIZ_JSON_SCHEMA = {
    "name": "quiz_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
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
        },
        "required": ["questions"],
    },
}


# ---------------------------------------------------------------------------
# Stage 3b: Single-source deepening — one source, no cross-source grounding
# ---------------------------------------------------------------------------
# Reuses INSIGHTS_JSON_SCHEMA (same fields); the quiz is its own parallel call
# (QUIZ_SYSTEM_PROMPT). intersections stays empty (there is nothing to compare
# across); the value is the apply-it scenarios, the headline takeaways, and a
# short "bigger picture" synthesis grounded in the one source. This is why a
# single document is no longer a dead end.

SINGLE_SOURCE_INSIGHTS_SYSTEM_PROMPT = """\
You deepen a SINGLE source's knowledge ledger into an active study artifact: the
headline takeaways, a short "bigger picture" reflection, and concrete apply-it
scenarios. Each ledger unit carries its source id and evidence, so ground
everything in specific claims.

PRODUCE:
  - key_takeaways: the 1-3 most important things to leave with, most important
    first, one sentence each. These are the headline.
  - synthesis_text: 1-2 tight paragraphs on the bigger picture — how the source's
    main ideas fit together, why they matter, and where they transfer. This is
    integration and implication, NOT a recap of the summary. State your inference
    as inference ("together these imply"), not as a source fact.
  - application_scenarios: 2 concrete "do this" scenarios that transfer the ideas
    to a real situation. Each has a specific prompt, ordered transfer_steps, and a
    real common_pitfall a learner would actually hit.
  - intersections: ALWAYS an empty array (single source, nothing to compare).

RULES:
1. BE SELECTIVE AND DIRECT. Only what genuinely matters. No preamble, no abstract
   labels standing in for the point, no padding.
2. Do not invent topics absent from the ledger.
3. Refer to the material as "the source". Never write "source <number>", ledger ids
   (u4), or "(evidence: ...)" in any text field.
4. Continuous prose in text fields. No markdown. No em dashes.

Return ONLY valid JSON matching the schema."""


# ---------------------------------------------------------------------------
# Stage 4: Chat (heuristic depth tier + budget-aware answering)
# ---------------------------------------------------------------------------
# The former LLM audit pass and LLM depth classifier were replaced by code:
# app/section_validator.py enforces budgets/restatement, and
# app/interaction.py:_heuristic_depth picks the chat tier. Counting sentences
# and keyword-routing a 4-way tier are not a model's job.

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
