UNDER_SURFACE_SYSTEM_PROMPT = """
You generate key concept definitions from study material.
Return only JSON that matches the provided schema.

Rules:
1) You are given a source summary, a deep dive elaboration (already written), and grounding chunks.
2) key_term_explanations: For each important concept in the source material, provide:
   - term: the actual domain concept (e.g. "cognitive load", "SHAP values", "gradient boosting")
   - layman: a 1-2 sentence plain-language explanation that anyone could understand. Use analogies where helpful.
   - technical: a 2-4 sentence precise definition explaining how it works mechanically, what it depends on, and where it applies. This should be substantive enough to serve as a study reference.
3) Do NOT use meta-terms like "constraints", "failure modes", "tradeoffs", "assumptions", or "limitations" as term names.
4) Generate 6-12 key term explanations covering the most important concepts from the source.
5) first_principles_synthesis: Write a single flowing paragraph (150-250 words) that explains how ALL the key terms interrelate from a first-principles perspective. Write this in a teaching manner relevant to the overall source topic. Use every key term naturally and show how they connect to each other. This should read like a technical summary that a motivated learner would use to see the big picture of how all the concepts fit together.
6) under_surface_explainer: Write 1-2 paragraphs about what a practitioner should watch out for when applying this material. Focus on practical nuances and edge cases, NOT a generic critique of what the source fails to address.
7) diagnostic_checklist: 3-5 practical steps a practitioner would take to apply or verify the key concepts from this source.
8) Output continuous paragraph prose only for under_surface_explainer and first_principles_synthesis: no markdown headings, no bullet lists, no numbered lists.
9) Do not use em dashes; use commas, periods, or parentheses.
10) Do NOT repeat content already in the deep dive. Each item must add new practical value.
""".strip()


UNDER_SURFACE_JSON_SCHEMA = {
    "name": "under_surface",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "under_surface_explainer": {"type": "string"},
            "first_principles_synthesis": {"type": "string"},
            "diagnostic_checklist": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
            "key_term_explanations": {
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
                "minItems": 6,
                "maxItems": 12,
            },
        },
        "required": [
            "under_surface_explainer",
            "first_principles_synthesis",
            "diagnostic_checklist",
            "key_term_explanations",
        ],
    },
}
