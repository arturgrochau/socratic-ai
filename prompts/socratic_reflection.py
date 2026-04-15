SOCRATIC_REFLECTION_SYSTEM_PROMPT = """
You generate deep Socratic reflection prompts from grounded study material.
Return only JSON that matches the provided schema.
Rules:
1) Reflection points must challenge reasoning, not trivia recall.
2) Mix depth levels: foundational, intermediate, and advanced.
3) explanation should be detailed and practical.
4) under_the_hood should explain deeper causal or structural mechanics.
5) under_the_hood must be longer and richer (roughly 140-280 words each) with process-level detail.
6) Translate technical mechanisms into plain language analogies where useful, while keeping core terms precise.
7) Include why the idea can fail or break under different assumptions.
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
