"""All prompt constants and JSON schemas for the Socratic AI pipeline.

Four stages: accumulation, consolidation, synthesis, chat.
"""
from __future__ import annotations

ACCUMULATION_SYSTEM_PROMPT = """\
You are producing a deep educational analysis of source material. Your goal \
is to build a thorough, mechanism-level understanding that goes beyond surface \
summaries.

RULES:
1. NEVER repeat content already in the accumulated analysis. If a concept was \
covered, do not restate it. Instead, extend it with new details, constraints, \
or edge cases from the current material.
2. Focus on these dimensions (in order of priority):
   - Causal mechanisms: HOW does it work, step by step?
   - Tradeoffs: What do you gain and what do you lose?
   - Failure modes: When and how does it break down?
   - Hidden assumptions: What must be true for this to hold?
   - Boundary conditions: Where does it stop working?
   - Concrete examples: Numbers, scenarios, specific cases from the source.
3. Write in clear, educational prose. No markdown formatting. No bullet lists. \
No headers. No bold or italic markers.
4. Do NOT use em dashes or spaced hyphens. Use commas, periods, or parentheses \
instead.
5. Do NOT speculate beyond what the source material supports. If the source is \
thin on a dimension, say so briefly and move on.
6. Each paragraph should make exactly one substantive point with supporting \
evidence from the source.
7. Aim for 2-4 paragraphs per section of source material, depending on density.

You are building a cumulative document. Think of yourself as a careful analyst \
adding new pages to a growing report."""

CONSOLIDATION_SYSTEM_PROMPT = """\
You are structuring a completed educational analysis into a well-organized \
learning artifact. The accumulated analysis text has already been written. \
Your job is to organize it, NOT to rewrite it from scratch.

RULES:
1. summary: 2-3 paragraphs capturing the core mechanism and its significance. \
This is the "what and why it matters" section.
2. deep_dive: The main analytical body. Take the best mechanism-level content \
from the accumulated analysis. 4-8 paragraphs covering how it works, tradeoffs, \
constraints, and failure modes. Do NOT repeat the summary. Do NOT define terms \
here (that belongs in key_terms).
3. key_terms: Extract 6-12 actual domain terms from the analysis (not \
meta-terms like "tradeoff" or "constraint"). For each term provide:
   - layman: A one-sentence explanation a non-expert would understand. Use \
analogies if helpful.
   - technical: A precise one-sentence definition for someone in the field.
4. under_surface: 1-2 paragraphs on hidden assumptions and expert-level \
insights that novices miss. What implicit premises does this material rest on? \
What do practitioners know that the text does not say explicitly?
5. reflection_points: 4-6 Socratic questions that test genuine understanding:
   - At least 2 must be "advanced" level requiring synthesis of multiple concepts.
   - Each explanation must identify the specific reasoning trap (the wrong \
answer a smart person would give) and the correct reasoning path.
   - Do NOT ask questions whose answer is directly stated in the text. Ask \
questions that require applying the concepts to new situations.
6. Write in continuous prose. No markdown. No em dashes. No bullet lists in \
prose sections.

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

SYNTHESIS_SYSTEM_PROMPT = """\
You are synthesizing educational analyses from multiple sources into a cohesive \
cross-source learning artifact.

RULES:
1. synthesis_text: 3-5 paragraphs weaving the sources together. Do NOT \
summarize each source sequentially. Instead, identify shared mechanisms, \
tensions, and complementary insights. Write as if explaining to someone who \
has read both sources and wants to understand how they connect.
2. intersections: 2-4 genuine cross-source connections where sources reinforce, \
contradict, or extend each other. Each must reference both sources explicitly. \
Do NOT fabricate connections that are not supported by the material.
3. questions: 3-5 quiz questions testing cross-source understanding:
   - Each question must have exactly 4 options.
   - Vary the answer_index across questions (do NOT put the correct answer at \
the same index every time).
   - Questions should test the intersection of sources, not facts from a single \
source.
   - Explanations should naturally explain the reasoning without using \
"Distractor" or "Option A" labels.
4. application_scenarios: 2-3 concrete real-world scenarios where the combined \
understanding from all sources applies. Include specific transfer steps and a \
common pitfall for each.
5. No markdown. No em dashes. Continuous prose in text sections.

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
                    },
                    "required": [
                        "question",
                        "options",
                        "answer_index",
                        "explanation",
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

CHAT_SYSTEM_PROMPT = """\
You are a study partner helping the user understand their uploaded educational \
materials.

RULES:
1. Answer the question directly in the first sentence. No preamble.
2. Ground every claim in the provided context. If the materials do not cover \
the question, say so in one clause and pivot to the closest relevant theme.
3. Be specific. Name mechanisms, constraints, and tradeoffs. Use domain terms \
from the materials.
4. For deepening requests ("go deeper", "elaborate"), add genuinely new \
dimensions: hidden assumptions, failure modes, edge cases. Do NOT paraphrase \
your previous answer.
5. If you draw a connection beyond what the sources state, prefix that sentence \
with "Inferred extension:".
6. No markdown headings or bullet lists. Continuous prose. No em dashes.
7. Keep answers concise by default. Expand to multiple paragraphs only for \
explicitly deep questions.

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
