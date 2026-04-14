COMBINED_QUIZ_SYSTEM_PROMPT = """
You generate a rigorous quiz grounded in overlap between sources.
Return only JSON that matches the provided schema.
Rules:
1) Questions must test concepts that appear across the provided materials.
2) Use moderate trickiness through plausible distractors, but avoid ambiguity.
3) Provide exactly 4 options per question.
4) answer_index must match the correct option.
5) explanation and under_the_hood should be elaborate and educational.
6) source_evidence should cite concrete facts/sections from the provided payload.
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
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "answer_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                        },
                        "explanation": {"type": "string"},
                        "under_the_hood": {"type": "string"},
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
                            "items": {"type": "string"},
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
            "study_advice": {"type": "string"},
        },
        "required": ["questions", "study_advice"],
    },
}
