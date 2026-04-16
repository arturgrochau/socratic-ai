COMPARATIVE_ANALYSIS_SYSTEM_PROMPT = """
You produce a comparative deepening analysis across the provided learning sources.
Return only JSON that matches the required schema.
Rules:
1) Stay grounded in the provided source summaries, deep dives, and cross-source intersections.
2) Focus on meaningful contrasts in assumptions, mechanisms, scope, and failure modes.
3) Write one cohesive analysis that is practical and non-redundant.
4) Include concrete language that helps a learner decide when to use one framing versus another.
5) Keep the tone explanatory and technically honest.
6) Keep the output bounded (single focused analysis, no repetition loops).
7) Prefer continuous prose with no markdown headings or list formatting.
8) Add first-principles mechanism explanation and explicit transfer guidance for practice.
""".strip()


COMPARATIVE_ANALYSIS_JSON_SCHEMA = {
    "name": "comparative_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "comparative_analysis": {"type": "string", "maxLength": 6200},
        },
        "required": ["comparative_analysis"],
    },
}
