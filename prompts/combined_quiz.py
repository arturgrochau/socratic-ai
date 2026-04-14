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
5) explanation should explicitly contrast why the correct option fits and why each distractor fails for this specific question.
6) under_the_hood must be technical, first-principles oriented, and mechanism-focused (causal chain, assumptions, constraints, and tradeoffs).
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
