COMBINED_QUIZ_SYSTEM_PROMPT = """
You generate a rigorous quiz grounded in overlap between sources.
Return only JSON that matches the provided schema.
Rules:
1) Questions must test concepts that appear across the provided materials.
2) Use high-discrimination trickiness through plausible distractors, but avoid ambiguity.
3) Distractors must be topic-adjacent and plausible to a strong student.
4) Avoid obvious giveaway distractors using absolute cue words such as "solely", "always", "never", "entirely", or "only" unless directly grounded in the source evidence.
5) Wrong options should often be true in a nearby context, but not true for the exact question asked.
3) Provide exactly 4 options per question.
4) answer_index must match the correct option.
5) explanation should only explain why distractors fail in this scope; do not re-teach the full mechanism narrative from prior sections.
6) under_the_hood must add one new edge condition or tradeoff angle not already stated in prior synthesis text.
6) source_evidence should cite concrete facts/sections from the provided payload and avoid repeating the same quote pattern across many questions.
7) Keep question bodies and explanations concise enough to avoid excessively large JSON payloads.
8) Assume mapping, constraint, and transfer stages are already known, do not reteach them.
9) Remove any explanation sentence that does not add new decision, implication, or edge-condition value.
10) If question explanations could be swapped with earlier narrative sections without meaning change, output is invalid.
""".strip()


COMBINED_QUIZ_JSON_SCHEMA = {
    "name": "combined_quiz",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 6,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string", "maxLength": 360},
                        "options": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 240},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "answer_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                        },
                        "explanation": {"type": "string", "maxLength": 1800},
                        "under_the_hood": {"type": "string", "maxLength": 2400},
                        "difficulty_level": {
                            "type": "string",
                            "enum": ["foundational", "intermediate", "advanced"],
                        },
                        "question_type": {
                            "type": "string",
                            "enum": ["cross_source", "synthesis", "application"],
                        },
                        "source_evidence": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 320},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                    },
                    "required": [
                        "question",
                        "options",
                        "answer_index",
                        "explanation",
                        "under_the_hood",
                        "difficulty_level",
                        "question_type",
                        "source_evidence",
                    ],
                },
            },
            "study_advice": {"type": "string", "maxLength": 1400},
        },
        "required": ["questions", "study_advice"],
    },
}
