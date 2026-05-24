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
3) intersections: Identify 2-4 genuine intersection points where two or more sources reinforce, contradict, or extend each other.
   - Each intersection needs a short title (5-8 words), a why_it_matters sentence, and an integrated_explanation paragraph (2-4 sentences) showing how the sources combine.
   - These must be REAL cross-source connections, not summaries of one source.
   - Every integrated_explanation MUST explicitly reference multiple sources using phrases like "both sources", "each source", "across the sources", "together", or "between". If you cannot honestly write such a sentence about an intersection, drop it and pick a different one.
   - Every intersection_title must be distinct. Do not repeat titles across the list.
4) application_scenarios: Produce 2-3 concrete real-world scenarios where the combined understanding would be useful.
   - Each scenario_title is 5-8 words.
   - scenario_prompt is one sentence describing the situation.
   - transfer_steps is a list of 3-4 short action items.
   - common_pitfall is one sentence on the most likely mistake.
5) questions (Quiz): Create 2-4 challenging reflection questions that test the learner's understanding of the *intersection* or *synthesis* of the sources.
   - Each question must have exactly 4 options.
   - The options should be plausible but distinct.
   - The explanation should be a single paragraph that naturally explains WHY the correct answer is right and the nuances of why the other options are incorrect.
   - CRITICAL: Do NOT use the word "Distractor" or "Option A" in the explanation. Just explain the concepts naturally (e.g., "While X is important, it misses the core dependency on Y...").
6) Do not use em dashes or spaced-hyphen punctuation (" - "). Use commas, periods, or parentheses.
""".strip()

SYNTHESIS_CONSOLIDATOR_JSON_SCHEMA = {
    "name": "synthesis_consolidation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "synthesis_text": {"type": "string"},
            "intersections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "intersection_title": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "integrated_explanation": {"type": "string"},
                    },
                    "required": ["intersection_title", "why_it_matters", "integrated_explanation"],
                },
                "minItems": 2,
                "maxItems": 4,
            },
            "application_scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scenario_title": {"type": "string"},
                        "scenario_prompt": {"type": "string"},
                        "transfer_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 4,
                        },
                        "common_pitfall": {"type": "string"},
                    },
                    "required": ["scenario_title", "scenario_prompt", "transfer_steps", "common_pitfall"],
                },
                "minItems": 2,
                "maxItems": 3,
            },
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
        "required": ["synthesis_text", "intersections", "application_scenarios", "questions"],
    },
}
