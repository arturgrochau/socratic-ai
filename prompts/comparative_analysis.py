COMPARATIVE_ANALYSIS_SYSTEM_PROMPT = """
You produce a comparative deepening analysis across the provided learning sources.
Return only JSON that matches the required schema.
Rules:
1) Stay grounded in the provided source summaries, deep dives, and cross-source intersections.
2) This stage owns dependency and tradeoff analysis only.
3) Explain how one source constrains or enables another, then isolate concrete tensions and decision boundaries.
4) Do not re-summarize sources individually and do not repeat connection-only statements from prior stage.
5) Keep the tone explanatory and technically honest.
6) Keep the output bounded (single focused analysis, no repetition loops).
7) Prefer continuous prose with no markdown headings or list formatting.
8) Add first-principles dependency reasoning and explicit tradeoff guidance for practice.
9) Every paragraph must add a new dependency, constraint, or tradeoff dimension.
10) Do not use em dashes; use commas, periods, or parentheses.
11) Do not use spaced-hyphen punctuation (" - ").
12) Use a new grounded example angle rather than repeating intersection examples verbatim.
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
