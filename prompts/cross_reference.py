CROSS_REFERENCE_SYSTEM_PROMPT = """
You compare exactly two extracted concepts.
Return only JSON that matches the provided schema.

Use relation_type values as follows:
- reinforces: both concepts support or strengthen the same idea
- new_info: target concept contributes distinct information not in source concept
- contradiction: concepts conflict directly
- partial_overlap: concepts overlap in part but differ in scope or details

Keep explanation short and specific to the compared concepts.
Set confidence as a number from 0.0 to 1.0.
""".strip()


CROSS_REFERENCE_JSON_SCHEMA = {
    "name": "cross_reference_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relation_type": {
                "type": "string",
                "enum": [
                    "reinforces",
                    "new_info",
                    "contradiction",
                    "partial_overlap",
                ],
            },
            "explanation": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["relation_type", "explanation", "confidence"],
    },
}


CROSS_REFERENCE_BATCH_SYSTEM_PROMPT = """
You classify relationships between multiple pairs of extracted concepts.
Return only JSON that matches the provided schema.

You will receive an array of concept pairs. For each pair, return a result with the same pair_index.
Do not skip any pair. Return results in pair_index order.

Use relation_type values as follows:
- reinforces: both concepts support or strengthen the same idea
- new_info: target concept contributes distinct information not in source concept
- contradiction: concepts conflict directly
- partial_overlap: concepts overlap in part but differ in scope or details

Keep each explanation short and specific to the compared concepts.
Set confidence as a number from 0.0 to 1.0.
""".strip()


CROSS_REFERENCE_BATCH_JSON_SCHEMA = {
    "name": "cross_reference_batch_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pair_index": {"type": "integer"},
                        "relation_type": {
                            "type": "string",
                            "enum": [
                                "reinforces",
                                "new_info",
                                "contradiction",
                                "partial_overlap",
                            ],
                        },
                        "explanation": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["pair_index", "relation_type", "explanation", "confidence"],
                },
            }
        },
        "required": ["results"],
    },
}
