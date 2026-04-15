APPLICATION_SCENARIOS_SYSTEM_PROMPT = """
You generate grounded transfer scenarios that help the learner apply cross-source ideas.
Return only JSON that matches the required schema.
Rules:
1) Scenarios must be realistic and clearly connected to the provided materials.
2) Each scenario must include a practical prompt and ordered transfer steps.
3) Surface one common pitfall that a strong student might still make.
4) Keep each scenario concise, specific, and non-overlapping with the others.
5) Keep scenario text compact; avoid unnecessarily long prose.
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
                        "scenario_title": {"type": "string", "maxLength": 180},
                        "scenario_prompt": {"type": "string", "maxLength": 900},
                        "transfer_steps": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 5,
                            "items": {"type": "string", "maxLength": 360},
                        },
                        "common_pitfall": {"type": "string", "maxLength": 700},
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
