COMBINED_INSIGHTS_SYSTEM_PROMPT = """
You extract cross-source relationships from consolidated source sections.
Return only JSON that matches the provided schema.
Rules:
1) Use only the provided consolidated source sections.
2) Identify intersections: where do sources share, reinforce, or challenge each other?
3) Identify tensions: where do sources disagree or create constraints for each other?
4) Identify transfer bridges: how can insights from one source be applied to the other?
5) attributed_sentences must be short sentence-level snippets with valid source attribution.
6) Keep outputs bounded and focused: concise titles, compact attributed sentences, no unnecessary repetition.
7) Do not re-summarize individual sources; focus on RELATIONSHIPS between them.
8) emphasis_terms should include important keywords for each sentence.
9) Do not use em dashes; use commas, periods, or parentheses.
10) Do not use spaced-hyphen punctuation (" - ").
""".strip()


COMBINED_INSIGHTS_JSON_SCHEMA = {
    "name": "combined_insights",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intersections": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "intersection_title": {"type": "string", "maxLength": 140},
                        "why_it_matters": {"type": "string", "maxLength": 1000},
                        "integrated_explanation": {"type": "string", "maxLength": 3600},
                        "attributed_sentences": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string", "maxLength": 320},
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
                        },
                        "inferred_extension": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                        "inference_label": {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "enum": ["inferred_extension"],
                                },
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": [
                        "intersection_title",
                        "why_it_matters",
                        "integrated_explanation",
                        "attributed_sentences",
                        "inferred_extension",
                        "inference_label",
                    ],
                },
            },
            "tensions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string", "maxLength": 140},
                        "source_a_claim": {"type": "string", "maxLength": 400},
                        "source_b_claim": {"type": "string", "maxLength": 400},
                        "resolution_or_tradeoff": {"type": "string", "maxLength": 600},
                    },
                    "required": ["title", "source_a_claim", "source_b_claim", "resolution_or_tradeoff"],
                },
            },
            "transfer_bridges": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "from_source_id": {"type": "integer"},
                        "to_source_id": {"type": "integer"},
                        "transfer_mechanism": {"type": "string", "maxLength": 500},
                        "adaptation_needed": {"type": "string", "maxLength": 400},
                    },
                    "required": ["from_source_id", "to_source_id", "transfer_mechanism", "adaptation_needed"],
                },
            },
        },
        "required": ["intersections", "tensions", "transfer_bridges"],
    },
}
