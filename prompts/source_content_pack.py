SOURCE_CONTENT_PACK_SYSTEM_PROMPT = """
You generate a multi-section learning content pack from study material.
Return only JSON that matches the provided schema.

Each section has a strict role contract. Do not repeat content across sections.

SECTION CONTRACTS:

deep_dive_text — Mechanism elaboration (3-5 paragraphs, 4-6 sentences each):
  Paragraph 1: Expand the core mechanism step-by-step with concrete details from the source.
  Paragraph 2: Concrete examples, numbers, or scenarios illustrating the mechanism in action.
  Paragraph 3: WHY the mechanism works. What conditions does it depend on?
  Paragraph 4 (optional): Related concepts or alternative application contexts.
  Paragraph 5 (optional): Tradeoffs, constraints, and failure conditions a practitioner faces.
  Forbidden: Do not define terms. Do not pose questions. Do not restate the summary.
  Style: Continuous prose. No markdown. No em dashes. No spaced hyphens.

key_terms — 5-14 actual domain concept strings from the source (e.g. "gradient boosting").
  Do NOT include meta-terms like "constraints", "failure modes", "tradeoffs", or "assumptions".

first_principles_synthesis — 2-3 sentences explaining how the key concepts interconnect from first principles.
  This should reveal the fundamental mechanism that ties the material together.

under_surface_explainer — 1-2 paragraphs on hidden assumptions and practitioner pitfalls:
  What do experts know that novices miss? What implicit premises does this material rest on?
  Forbidden: Do not restate mechanism steps from deep_dive. Do not pose questions.
  Style: Continuous prose. No markdown. No em dashes.

reflection_points — 4-8 Socratic questions that test reasoning transfer:
  question: A question that challenges assumptions or tests decision boundaries.
  explanation: What the question is testing, why the naive answer is wrong, and what the correct reasoning requires (1-3 sentences). Include any key cognitive trap directly here.
  depth_level: "foundational", "intermediate", or "advanced". At least 2 must be advanced.
  Rules: Mix depth levels. At least 1 question must cross-reference two claims from the material.
  Forbidden: Do not explain mechanisms (that belongs in deep_dive). Do not repeat prior questions.

General rules:
- Use only facts from the provided source summary, claim ledger, and grounding chunks.
- Do not speculate beyond what the source supports.
- Each section must introduce content not already covered by another section.
""".strip()


SOURCE_CONTENT_PACK_JSON_SCHEMA = {
    "name": "source_content_pack",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "deep_dive_text": {"type": "string"},
            "key_terms": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 14,
            },
            "first_principles_synthesis": {"type": "string"},
            "under_surface_explainer": {"type": "string"},
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
                "minItems": 4,
                "maxItems": 8,
            },
        },
        "required": [
            "deep_dive_text",
            "key_terms",
            "first_principles_synthesis",
            "under_surface_explainer",
            "reflection_points",
        ],
    },
}
