CHUNK_DIFF_SYSTEM_PROMPT = """
You extract novel claims from a chunk window that are NOT already stated in the source summary.
Return only JSON that matches the provided schema.
Rules:
1) A claim is: a mechanism detail, a constraint, a failure condition, a tradeoff, an assumption, or a practical implication.
2) Only extract claims that add information BEYOND what the summary already states.
3) If the chunk window only restates what the summary says, return an empty claims array.
4) Each claim must be a single, self-contained factual statement grounded in the chunk text.
5) grounding_quote must be a short verbatim or near-verbatim excerpt from the chunk window that supports the claim.
6) claim_type must be one of: mechanism, constraint, failure, tradeoff, assumption, implication.
7) Prefer specificity: numbers, thresholds, named techniques, concrete examples over vague generalizations.
8) Do not invent claims that are not supported by the chunk text.
9) Do not include claims that merely define a term without adding operational detail.
""".strip()


CHUNK_DIFF_JSON_SCHEMA = {
    "name": "chunk_diff_claims",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string", "maxLength": 300},
                        "grounding_quote": {"type": "string", "maxLength": 200},
                        "claim_type": {
                            "type": "string",
                            "enum": [
                                "mechanism",
                                "constraint",
                                "failure",
                                "tradeoff",
                                "assumption",
                                "implication",
                            ],
                        },
                    },
                    "required": ["claim", "grounding_quote", "claim_type"],
                },
                "minItems": 0,
                "maxItems": 5,
            }
        },
        "required": ["claims"],
    },
}
