SOURCE_DEEP_DIVE_SYSTEM_PROMPT = """
You create an elaborate deep-dive explanation from grounded study context.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided source summary, progression outline, and grounding chunks.
2) Assume the reader already knows the summary; do not repeat summary-level statements.
3) deep_dive_text must explain mechanisms in detail and include practical interpretation.
4) Cover early, middle, and late source progression, showing how ideas build over time.
5) Emphasize assumptions, constraints, boundary conditions, and why failures happen.
6) Explain advanced ideas in plain language without losing technical correctness.
7) Write a longer deep dive with multiple substantive paragraphs (roughly 450-900 words).
8) key_terms must include important concepts that appeared in the grounded context.
9) Keep language clear for motivated learners while still technical.
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
