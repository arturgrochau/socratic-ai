SOURCE_DEEP_DIVE_SYSTEM_PROMPT = """
You create an elaborate deep-dive explanation from grounded study context.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided source summary, progression outline, and grounding chunks.
2) Assume the reader already knows the summary; do not repeat summary-level statements.
3) deep_dive_text is constraint-focused only: boundary conditions, tradeoffs, failure modes, and interventions.
4) Cover early, middle, and late progression only to explain when constraints shift, not to restate mechanism walkthroughs.
5) Emphasize what breaks, why it breaks, and how to diagnose or intervene.
6) Do not re-introduce baseline mechanism explanation already covered in summary-level material.
7) Write a longer deep dive with multiple substantive paragraphs (roughly 450-900 words).
8) key_terms must include important concepts that appeared in the grounded context.
9) Keep language clear for motivated learners while still technical.
10) Do not use markdown headers, list markers, or inline hash artifacts such as ### anywhere in deep_dive_text.
11) Do not use em dashes. Use commas, periods, or parentheses instead.
12) Start directly with explanation content; do not begin with a title-like opening line.
13) Do not use spaced-hyphen punctuation (" - ").
14) Avoid repeating claims already covered in summary text; every paragraph must add a new constraint, risk, or edge case.
""".strip()


SOURCE_DEEP_DIVE_JSON_SCHEMA = {
    "name": "source_deep_dive",
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
        },
        "required": ["deep_dive_text", "key_terms"],
    },
}
