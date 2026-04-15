UNDER_SURFACE_SYSTEM_PROMPT = """
You generate a longer technical-but-plain-language explainer from grounded source material.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided summary, deep-dive text, progression outline, key terms, and grounding chunks.
2) Assume the summary and deep dive were already read; avoid repeating them.
3) under_surface_explainer must be extensive and first-principles: explain why each core mechanism behaves as it does.
4) Explicitly include causal chains, assumptions, constraints, tradeoffs, and failure modes.
5) Cover progression-aware behavior: foundational stage, integration stage, and edge-case stage.
6) diagnostic_checklist should provide practical checks at each stage (foundational, integration, edge-case).
7) key_term_explanations should deepen mental models by connecting terms to underlying mechanisms.
8) Prefer depth over brevity while staying grounded and coherent (roughly 700-1300 words).
""".strip()


UNDER_SURFACE_JSON_SCHEMA = {
    "name": "under_surface",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "under_surface_explainer": {"type": "string"},
            "diagnostic_checklist": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 8,
            },
            "key_term_explanations": {
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
                "minItems": 4,
                "maxItems": 10,
            },
        },
        "required": [
            "under_surface_explainer",
            "diagnostic_checklist",
            "key_term_explanations",
        ],
    },
}
