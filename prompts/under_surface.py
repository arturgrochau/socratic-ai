UNDER_SURFACE_SYSTEM_PROMPT = """
You generate a longer technical-but-plain-language explainer from grounded source material.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided summary, deep-dive text, key terms, and grounding chunks.
2) under_surface_explainer must be detailed and mechanism-focused in plain language for motivated learners.
3) Explain the underlying process, assumptions, constraints, and failure modes where relevant.
4) diagnostic_checklist should provide practical checks someone can use to validate understanding or implementation.
5) key_term_explanations should define terms in plain language while preserving technical correctness.
6) Prefer depth over brevity while staying grounded and coherent.
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
