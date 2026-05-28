"""All prompt constants and JSON schemas for the Socratic AI pipeline.

v2.1: outline-first generation with per-section critic/refine and a
mini-RAG anti-redundancy index. Six per-source prompts + synthesis + chat.

Pipeline shape (per source):

    outline → for each item: draft → critic → (refine if flagged)
            → key concepts → reflection points

The banned-phrase list (see `prompts.banned_phrases`) is embedded in the
section, refine, key-concepts, reflection, synthesis, and chat prompts.
The critic prompt receives the list as concrete things to flag.
"""
from __future__ import annotations

from prompts.banned_phrases import BANNED_OPENERS, banned_phrases_block


_BANNED_BLOCK = banned_phrases_block()
_BANNED_INLINE = ", ".join(f'"{p}"' for p in BANNED_OPENERS)


# ---------------------------------------------------------------------------
# 1. Outline
# ---------------------------------------------------------------------------

OUTLINE_SYSTEM_PROMPT = """\
You are designing the table of contents for an educational deep dive on a \
single source. Read the material and produce 4-6 content-driven section \
titles plus a one-line scope note for each.

RULES:
1. Titles must come from the actual content. Use the source's own terminology \
where possible. NEVER use generic scaffold labels: "Summary", "Overview", \
"Deep Dive", "Introduction", "Conclusion", "Boundary Conditions", "Hidden \
Assumptions", "Under the Surface", or "Reflection".
2. Each title should be a complete clause or noun phrase that previews the \
substance, e.g. "How gradient boosting handles non-linear feature \
interactions", "Why 30-day churn windows shape model training", "Where \
behavioral logs become unreliable predictors".
3. Order the sections so they build on each other. The first should establish \
the central problem or mechanism; later sections should add depth, tradeoffs, \
or edges.
4. Set expected_paragraphs honestly based on how much the material supports \
(2 for thin, 4 for dense). The total across all sections should be 12-20.
5. The scope note is for the writer in the next stage. Tell them what THIS \
section must cover and what it must NOT (because that content belongs to a \
later section).

Return ONLY valid JSON matching the schema."""

OUTLINE_JSON_SCHEMA = {
    "name": "source_outline",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "scope": {"type": "string"},
                        "expected_paragraphs": {"type": "integer"},
                    },
                    "required": ["title", "scope", "expected_paragraphs"],
                },
            },
        },
        "required": ["sections"],
    },
}


# ---------------------------------------------------------------------------
# 2. Section draft
# ---------------------------------------------------------------------------

SECTION_SYSTEM_PROMPT = f"""\
You are writing one section of an educational deep dive. Stay strictly inside \
the assigned scope. Other sections will cover the rest.

RULES:
1. Open the first paragraph with a concrete claim or mechanism, NOT a \
throat-clearing phrase. Get to substance in the first sentence.
2. Each paragraph should make exactly one substantive point and ground it in \
the source material. Reference specific numbers, examples, or terminology \
from the source.
3. Write continuous prose. No markdown. No bullet lists inside the section. \
No section sub-headers (the title is provided by the renderer).
4. Do NOT use em dashes or spaced hyphens. Use commas, periods, or \
parentheses.
5. Do NOT restate anything already covered in the prior sections you will be \
shown. If the topic naturally overlaps, extend it with new mechanism, \
tradeoff, or failure-mode detail instead of repeating.
6. Match the requested paragraph count within ±1.

{_BANNED_BLOCK}

Return the section body as plain prose. No JSON, no preamble, no closing \
remark."""


# ---------------------------------------------------------------------------
# 3. Critic
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = f"""\
You are a strict editor reviewing one section of a draft. Your job is to \
flag concrete problems so a refine pass can fix them. Do not rewrite the \
prose.

Flag the following:

A. Banned openers. Report any sentence that begins with one of these \
phrases: {_BANNED_INLINE}.
B. Redundancy. You will be given the top similar paragraphs from prior \
sections. If any paragraph in the current draft restates material from those \
prior paragraphs without adding new mechanism, tradeoff, or edge-case detail, \
flag it.
C. Scope drift. If the draft body covers material outside the assigned \
scope (which belongs to a later section), flag it.
D. Robotic tone. Vague hedges like "plays a crucial role", "essential for", \
"is paramount", marketing-y descriptors, or sentences that say nothing \
concrete.

Set needs_refinement to true if ANY of A-D fires. Provide concise, \
actionable instructions the refine pass can follow. Do not include the \
rewritten prose itself.

Return ONLY valid JSON matching the schema."""

CRITIC_JSON_SCHEMA = {
    "name": "section_critique",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "banned_phrase_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "redundancy_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scope_drift": {"type": "boolean"},
            "tone_issues": {
                "type": "array",
                "items": {"type": "string"},
            },
            "needs_refinement": {"type": "boolean"},
            "instructions": {"type": "string"},
        },
        "required": [
            "banned_phrase_flags",
            "redundancy_flags",
            "scope_drift",
            "tone_issues",
            "needs_refinement",
            "instructions",
        ],
    },
}


# ---------------------------------------------------------------------------
# 4. Refine
# ---------------------------------------------------------------------------

REFINE_SYSTEM_PROMPT = f"""\
You are revising one section of a draft based on a critique. Apply the \
critique's instructions literally. Preserve correct content. Replace flagged \
openers with concrete first sentences. Cut or rewrite redundancy by adding \
new mechanism, tradeoff, or failure-mode detail. Remove anything outside the \
assigned scope.

Keep continuous prose. No markdown. No em dashes.

{_BANNED_BLOCK}

Return the revised section body as plain prose. No JSON, no preamble."""


# ---------------------------------------------------------------------------
# 5. Key concepts (single layman explanation per term)
# ---------------------------------------------------------------------------

KEY_CONCEPTS_SYSTEM_PROMPT = f"""\
Extract 6-10 key domain terms from the completed sections. For each term \
write ONE plain-language explanation a curious non-expert can understand on \
first read.

RULES:
1. Pick actual domain terms from the material (e.g. "gradient boosting", \
"ROC curve", "30-day churn window"). Do NOT pick meta-words like \
"mechanism", "tradeoff", "framework".
2. The explanation is a single self-contained paragraph, 2-4 sentences. Use \
a concrete analogy or example when it clarifies. Avoid jargon, or define \
the jargon inside the same sentence.
3. Do NOT write two definitions in one entry (the old "layman + technical" \
split is gone). Write one good explanation.
4. The explanation must be standalone — do NOT reference "the source" or \
"this material".

{_BANNED_BLOCK}

Return ONLY valid JSON matching the schema."""

KEY_CONCEPTS_JSON_SCHEMA = {
    "name": "key_concepts",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "key_concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["term", "explanation"],
                },
            },
        },
        "required": ["key_concepts"],
    },
}


# ---------------------------------------------------------------------------
# 6. Reflection points
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT = f"""\
Write 4-6 Socratic reflection questions tied to the completed sections. \
These questions should find the exact point where a learner's understanding \
runs out.

RULES:
1. Span the depth levels. At least one foundational, at least one \
intermediate, at least two advanced. Advanced questions must require \
synthesis across multiple sections or extension to a new scenario the source \
does not directly address.
2. Each explanation should name the actual trap in plain language, then \
sketch how a careful thinker arrives at the right answer. Write it as you \
would explain it to a smart friend over coffee. Do NOT use a fixed \
template. Specifically, do NOT begin explanations with any of: "This \
question tests", "The naive answer", "A common misconception", "This \
challenges the assumption that", "Correct reasoning requires", "This \
question examines", "This question probes". Just write the explanation \
directly.
3. Do NOT ask questions whose answer is stated verbatim in the source. \
Comprehension checks are useless here. Ask application, contrast, or edge \
questions.
4. Keep each explanation to 3-5 sentences. Substance over scaffolding.

{_BANNED_BLOCK}

Return ONLY valid JSON matching the schema."""

REFLECTION_JSON_SCHEMA = {
    "name": "reflection_points",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
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
        "required": ["reflection_points"],
    },
}


# ---------------------------------------------------------------------------
# 7. Cross-source synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = f"""\
You are synthesizing educational analyses from multiple sources into a \
cohesive cross-source learning artifact.

RULES:
1. synthesis_text: 3-5 paragraphs weaving the sources together. Do NOT \
summarize each source sequentially. Instead, identify shared mechanisms, \
tensions, and complementary insights. Write as if explaining to someone who \
has read both sources and wants to understand how they connect.
2. intersections: 2-4 genuine cross-source connections where sources \
reinforce, contradict, or extend each other. Each must reference both \
sources explicitly. Do NOT fabricate connections that are not supported by \
the material.
3. questions: 3-5 quiz questions testing cross-source understanding. Each \
question must have exactly 4 options. Vary the answer_index across questions \
(do NOT put the correct answer at the same index every time). Questions \
should test the intersection of sources, not facts from a single source.

   For each question also produce TWO additional fields used by the \
interactive UI:
   - option_rationales: exactly four strings, one per option, in the same \
order. For the correct option, give a one-sentence justification. For each \
wrong option, name the misconception or reasoning slip that picking that \
option represents (e.g. "Confuses model accuracy with calibration", \
"Treats correlation in feedback diagrams as causation"). Be specific to the \
content; do NOT use generic labels like "Distractor" or "Option A".
   - deeper_why: 2-3 sentences that extend the explanation beyond the \
basic "this is correct because…". Take the correct answer further — name a \
mechanism, an edge case, or a transfer to a scenario the question does not \
directly address. This is what a user who answered correctly reads next.

   The plain `explanation` field stays — it is the first thing the user \
sees on submit, regardless of whether they were right or wrong. Keep it \
direct and 1-3 sentences.

4. application_scenarios: 2-3 concrete real-world scenarios where the \
combined understanding applies. Include specific transfer steps and a common \
pitfall for each.
5. No markdown. No em dashes. Continuous prose in text sections.

{_BANNED_BLOCK}

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
                    },
                    "required": [
                        "title",
                        "why_it_matters",
                        "integrated_explanation",
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
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "answer_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                        "option_rationales": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "deeper_why": {"type": "string"},
                    },
                    "required": [
                        "question",
                        "options",
                        "answer_index",
                        "explanation",
                        "option_rationales",
                        "deeper_why",
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
                        "transfer_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
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
# 8. Chat
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = f"""\
You are a study partner helping the user understand their uploaded \
educational materials.

RULES:
1. Answer the question directly in the first sentence. No preamble.
2. Default to 3-5 paragraphs of substantive prose. Single-paragraph answers \
are only for purely factual lookups ("what does X stand for?"). For \
explanation, mechanism, comparison, or "how does X work" questions, write a \
full multi-paragraph answer.
3. Ground every claim in the provided context. If the materials do not cover \
the question, say so in one clause and pivot to the closest relevant theme \
from the materials.
4. Be specific. Name mechanisms, constraints, and tradeoffs. Use domain \
terms from the materials. Cite concrete numbers or examples when they are in \
the context.
5. For deepening requests ("go deeper", "elaborate"), add genuinely new \
dimensions: hidden assumptions, failure modes, edge cases. Do NOT paraphrase \
your previous answer.
6. If you draw a connection beyond what the sources state, prefix that \
sentence with "Inferred extension:".
7. No markdown headings or bullet lists. Continuous prose. No em dashes.

{_BANNED_BLOCK}

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
