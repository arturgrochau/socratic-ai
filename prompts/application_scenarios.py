APPLICATION_SCENARIOS_SYSTEM_PROMPT = """
You generate grounded transfer scenarios that help the learner apply cross-source ideas.
Return only JSON that matches the required schema.
Rules:
1) Scenarios must be realistic and clearly connected to the provided materials.
2) Each scenario must include a practical prompt and ordered transfer steps.
3) Surface one common pitfall that a strong student might still make.
4) Keep each scenario concise, specific, and non-overlapping with the others.
""".strip()


APPLICATION_SCENARIOS_JSON_SCHEMA = {
    "name": "application_scenarios",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "application_scenarios": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scenario_title": {"type": "string"},
                        "scenario_prompt": {"type": "string"},
                        "transfer_steps": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 5,
                            "items": {"type": "string"},
                        },
                        "common_pitfall": {"type": "string"},
                    },
                    "required": [
                        "scenario_title",
                        "scenario_prompt",
                        "transfer_steps",
                        "common_pitfall",
                    ],
                },
            },
        },
        "required": ["application_scenarios"],
    },
}
