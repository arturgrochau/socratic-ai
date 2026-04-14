CONCEPT_EXTRACTION_SYSTEM_PROMPT = """
You are an extraction engine that converts study material into structured concepts.
Return only JSON that matches the provided schema.
Do not add fields outside the schema.
Each concept must be concise and grounded in the provided source text.
""".strip()


CONCEPT_EXTRACTION_JSON_SCHEMA = {
    "name": "concept_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"},
                        "key_ideas": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["term", "definition", "key_ideas"],
                },
            }
        },
        "required": ["concepts"],
    },
}
