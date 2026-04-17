SOCRATIC_REFLECTION_SYSTEM_PROMPT = """
You generate deep Socratic reflection prompts from grounded study material.
Return only JSON that matches the provided schema.
Rules:
1) Reflection points must challenge reasoning, not trivia recall.
2) Mix depth levels: foundational, intermediate, and advanced.
3) explanation should focus on transfer framing: what to test, what assumption to question, and what decision boundary to inspect.
4) under_the_hood should surface hidden implications for reasoning quality, not replay the mechanism explanation from prior sections.
5) under_the_hood must be longer and richer (roughly 160-320 words each) with reasoning traps and transfer checks.
6) Do not restate deep-dive or under-surface text; each item must add a new evaluation lens.
7) Translate technical ideas into practical thinking checks where useful, while keeping core terms precise.
8) Advanced items should be materially deeper than foundational items.
""".strip()


SOCRATIC_REFLECTION_JSON_SCHEMA = {
    "name": "socratic_reflection",
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
                        "under_the_hood": {"type": "string"},
                        "depth_level": {
                            "type": "string",
                            "enum": ["foundational", "intermediate", "advanced"],
                        },
                    },
                    "required": [
                        "question",
                        "explanation",
                        "under_the_hood",
                        "depth_level",
                    ],
                },
                "minItems": 4,
                "maxItems": 8,
            }
        },
        "required": ["reflection_points"],
    },
}
