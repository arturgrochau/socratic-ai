SOURCE_TITLE_SYSTEM_PROMPT = """
You generate concise, descriptive study titles for a single source summary.
Return only JSON that matches the provided schema.
The title should be specific, accurate, and easy to understand.
Use 4 to 10 words.
""".strip()


SOURCE_TITLE_JSON_SCHEMA = {
    "name": "source_title",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
        },
        "required": ["title"],
    },
}
