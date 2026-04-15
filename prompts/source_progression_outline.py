SOURCE_PROGRESSION_OUTLINE_SYSTEM_PROMPT = """
You create a concise progression outline from grounded source material.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided summary and grounding chunks.
2) Explain the source progression from beginning to middle to ending.
3) Surface how assumptions, complexity, and failure boundaries evolve over that progression.
4) Keep the outline practical and mechanism-focused for downstream deep-dive generation.
5) transition_points must describe concrete stage transitions, not generic observations.
""".strip()


SOURCE_PROGRESSION_OUTLINE_JSON_SCHEMA = {
    "name": "source_progression_outline",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "progression_outline": {"type": "string"},
            "transition_points": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 8,
            },
        },
        "required": ["progression_outline", "transition_points"],
    },
}
