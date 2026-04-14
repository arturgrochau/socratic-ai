COMBINED_INSIGHTS_SYSTEM_PROMPT = """
You synthesize insights across a video and documents.
Return only JSON that matches the provided schema.
Rules:
1) Use only provided summaries, grounding chunks, and relationship notes.
2) parallels entries must be complete sentences with source attribution.
3) Make the layman_bridge practical and easy to apply.
4) synthesis_text must be elaborate and connect multiple sources in plain language.
5) emphasis_terms should include important keywords for each sentence.
""".strip()


COMBINED_INSIGHTS_JSON_SCHEMA = {
    "name": "combined_insights",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "parallels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "source_id": {"type": "integer"},
                        "source_type": {
                            "type": "string",
                            "enum": ["video", "document"],
                        },
                        "emphasis_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 6,
                        },
                    },
                    "required": ["text", "source_id", "source_type", "emphasis_terms"],
                },
                "minItems": 5,
                "maxItems": 10,
            },
            "layman_bridge": {"type": "string"},
            "synthesis_text": {"type": "string"},
        },
        "required": ["parallels", "layman_bridge", "synthesis_text"],
    },
}
