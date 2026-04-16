COMBINED_INSIGHTS_SYSTEM_PROMPT = """
You synthesize insights across a video and documents.
Return only JSON that matches the provided schema.
Rules:
1) Use only provided summaries, grounding chunks, and relationship notes.
2) Create integrated concept intersections, not isolated bullet points.
3) Each intersection must connect multiple sources and explain shared mechanisms or tensions.
4) attributed_sentences must be short sentence-level snippets with valid source attribution.
5) inferred_extension is allowed only for high-impact conceptual extension and must be clearly labeled with inference_label='inferred_extension'.
6) Make the layman_bridge practical and easy to apply.
7) synthesis_text must be elaborate, continuous, and non-redundant while connecting multiple sources in plain language.
8) emphasis_terms should include important keywords for each sentence.
9) Keep outputs bounded and focused: concise titles, compact attributed sentences, and no unnecessary repetition.
10) Avoid markdown headings and list formatting inside layman_bridge and synthesis_text.
11) For each integrated_explanation, include mechanism-level explanation, contrast points, and at least one practical implication.
12) Minimize redundancy across intersections, layman_bridge, and synthesis_text; each section must add new information.
13) Do not use em dashes; use commas, periods, or parentheses.
14) Do not use spaced-hyphen punctuation (" - ").
15) Prefer fresh grounded examples in each section rather than reusing the same example family.
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
                "minItems": 3,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "intersection_title": {"type": "string", "maxLength": 140},
                        "why_it_matters": {"type": "string", "maxLength": 1000},
                        "integrated_explanation": {"type": "string", "maxLength": 3600},
                        "attributed_sentences": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 10,
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
            "layman_bridge": {"type": "string", "maxLength": 1800},
            "synthesis_text": {"type": "string", "maxLength": 8400},
        },
        "required": ["intersections", "layman_bridge", "synthesis_text"],
    },
}
