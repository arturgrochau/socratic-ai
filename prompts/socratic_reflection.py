SOCRATIC_REFLECTION_SYSTEM_PROMPT = """
You generate deep Socratic reflection prompts from grounded study material.
Return only JSON that matches the provided schema.
Rules:
1) Reflection points must challenge reasoning, not trivia recall.
2) Mix depth levels: foundational, intermediate, and advanced. At least 2 must be advanced.
3) explanation should focus on transfer framing: what to test, what assumption to question, and what decision boundary to inspect.
4) reasoning_traps must describe a specific cognitive trap: a conclusion that SEEMS correct given the summary but breaks under the boundary conditions. Each must be 160-320 words with concrete trap mechanisms.
5) At least 1 point must require cross-referencing two different claims from the provided material.
6) Do not restate boundary or hidden-assumptions text; each item must add a new evaluation lens.
7) Translate technical ideas into practical thinking checks where useful, while keeping core terms precise.
8) Advanced items must be materially deeper than foundational items.
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
                        "reasoning_traps": {"type": "string"},
                        "depth_level": {
                            "type": "string",
                            "enum": ["foundational", "intermediate", "advanced"],
                        },
                    },
                    "required": [
                        "question",
                        "explanation",
                        "reasoning_traps",
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
