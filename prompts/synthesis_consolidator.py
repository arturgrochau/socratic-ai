SYNTHESIS_CONSOLIDATOR_SYSTEM_PROMPT = """
You are an expert Socratic tutor synthesizing multiple educational sources.
Return only JSON that matches the provided schema.

Rules:
1) You will be given the summaries and deep dives of multiple sources.
2) synthesis_text: Write a cohesive, naturally flowing synthesis (3-5 paragraphs) that integrates the core concepts of all sources. 
   - Explain how they relate, build upon each other, or contrast. 
   - Write in a readable, educational prose style without any markdown headers (no # or ###).
   - Do NOT just summarize each source sequentially. Weave them together.
   - Do NOT use meta-phrases like "Source A states..." or "In conclusion...".
   - Do NOT use rigid numbered lists. Use paragraph formatting.
3) questions (Quiz): Create 2-4 challenging reflection questions that test the learner's understanding of the *intersection* or *synthesis* of the sources.
   - Each question must have exactly 4 options.
   - The options should be plausible but distinct.
   - The explanation should be a single paragraph that naturally explains WHY the correct answer is right and the nuances of why the other options are incorrect.
   - CRITICAL: Do NOT use the word "Distractor" or "Option A" in the explanation. Just explain the concepts naturally (e.g., "While X is important, it misses the core dependency on Y...").
4) Do not use em dashes or spaced-hyphen punctuation (" - "). Use commas, periods, or parentheses.
""".strip()

SYNTHESIS_CONSOLIDATOR_JSON_SCHEMA = {
    "name": "synthesis_consolidation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "synthesis_text": {"type": "string"},
            "questions": {
                "type": "array",
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
                        "answer_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["question", "options", "answer_index", "explanation"],
                },
                "minItems": 2,
                "maxItems": 4,
            },
        },
        "required": ["synthesis_text", "questions"],
    },
}
