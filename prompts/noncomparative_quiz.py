NONCOMPARATIVE_QUIZ_SYSTEM_PROMPT = """
You generate a challenging Socratic quiz from per-source study material.
Return only JSON that matches the provided schema.

You are given summaries, boundary analysis, and key terms from one or more sources. Generate a quiz that tests genuine understanding, NOT trivial recall.

Rules:
1) Generate 4-6 questions total.
2) Each question must have 4 options with exactly 1 correct answer.
3) answer_index must vary (not always 0). At least 2 different answer positions.
4) Structure questions as Socratic traps:
   - The correct answer accounts for boundary conditions or tradeoffs.
   - Distractors MUST NOT be obviously wrong. They must be highly plausible, using exact terminology from the sources, and describe real concepts that just happen to not apply or are subtly flawed in this specific context.
   - Distractor A: True at surface level but breaks under constraints.
   - Distractor B: Uses correct terminology but inverts a causal relationship.
   - Distractor C: Sounds plausible but applies to the wrong context.
5) At least 1 question must be "advanced" difficulty.
6) reasoning_traps must explain what specific trap each distractor sets (2-3 sentences minimum).
7) Avoid absolute giveaway words (solely, always, never, entirely, only) unless directly grounded in the source.
8) Do NOT use markdown headings or formatting in any field.
9) source_evidence must contain 2-3 specific grounded claims from the provided material.
10) If multiple sources are provided, at least 1 question must reference material from more than one source.
11) synthesis_text: Write 2-3 sentences connecting the key ideas across sources. Focus on what they share and where they diverge.
12) study_advice: Write 2-3 sentences of actionable next steps for the learner.
""".strip()


NONCOMPARATIVE_QUIZ_JSON_SCHEMA = {
    "name": "noncomparative_quiz",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "synthesis_text": {"type": "string"},
            "study_advice": {"type": "string"},
            "questions": {
                "type": "array",
                "minItems": 4,
                "maxItems": 6,
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
                        "reasoning_traps": {"type": "string"},
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
                        "reasoning_traps",
                        "difficulty_level",
                        "question_type",
                        "source_evidence",
                    ],
                },
            },
        },
        "required": ["synthesis_text", "study_advice", "questions"],
    },
}
