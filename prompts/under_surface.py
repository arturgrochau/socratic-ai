UNDER_SURFACE_SYSTEM_PROMPT = """
You generate a longer technical-but-plain-language explainer from grounded source material.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided summary, deep-dive text, progression outline, key terms, and grounding chunks.
2) Assume the summary and deep dive were already read; avoid repeating them.
3) under_surface_explainer is hidden-assumption focused only: expose latent assumptions, system limits, and second-order effects.
4) Do not provide a step-by-step mechanism walkthrough, practical how-to, or restated deep-dive constraints.
5) Cover progression-aware assumptions: foundational assumptions, integration assumptions, and edge-case assumption failures.
6) diagnostic_checklist should provide practical checks at each stage (foundational, integration, edge-case).
7) key_term_explanations, if present, must avoid dictionary-style restatement and only capture hidden assumptions tied to the term.
8) Prefer depth over brevity while staying grounded and coherent (roughly 850-1600 words).
9) Every paragraph must introduce a new assumption, hidden dependency, or system limit not already explicit in prior sections.
10) Output continuous paragraph prose only for under_surface_explainer: no markdown headings, no bullet lists, no numbered lists.
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
                "minItems": 0,
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
