APPLICATION_SCENARIOS_SYSTEM_PROMPT = """
You generate grounded transfer scenarios that help the learner apply cross-source ideas.
Return only JSON that matches the required schema.
Rules:
1) Scenarios must be realistic and clearly connected to the provided materials.
2) Each scenario must include a practical prompt and ordered transfer steps.
3) Surface one common pitfall that a strong student might still make.
4) This stage owns friction scenarios only: each scenario must expose where one source's output creates a constraint for another source's method.
5) Do not restate connection, dependency, or tradeoff summaries; focus on interaction breakdown points and mitigation actions.
6) Prefer 3-4 scenarios when there is enough cross-source material.
7) Assume mapping and constraint analysis are already known, do not reintroduce them.
8) Remove any scenario sentence that does not add transfer adaptation or mitigation value.
9) If a scenario could be swapped with another stage without changing meaning, it is invalid.
10) Keep each scenario compact: 1 short prompt paragraph plus 3-4 action-first steps.
11) Avoid repeated analogy framing across scenarios unless the later analogy extends with a new operational constraint.
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
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scenario_title": {"type": "string", "maxLength": 180},
                        "scenario_prompt": {"type": "string", "maxLength": 1300},
                        "transfer_steps": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 6,
                            "items": {"type": "string", "maxLength": 520},
                        },
                        "common_pitfall": {"type": "string", "maxLength": 1100},
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
