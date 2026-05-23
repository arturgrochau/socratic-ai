COMBINED_PROCESSING_SYSTEM_PROMPT = """
You analyze study material and return both a structured concept extraction and a narrative summary.
Return only JSON that matches the provided schema.

Rules for concepts:
1) Extract 6-14 key concepts that are central to understanding this source.
2) Each concept must have a concise term, a clear definition grounded in the source, and 2-4 key ideas.
3) Use actual domain terms (e.g. "gradient boosting", "cognitive load"). Do not use meta-terms like "limitations" or "tradeoffs" as terms.
4) Each concept must be directly supported by the source text.

Rules for source_summary:
1) Write exactly three paragraphs of 4-6 sentences each:
   Paragraph 1: Core thesis, scope, and central argument.
   Paragraph 2: Key mechanisms, methods, and how they work.
   Paragraph 3: Practical implications, limitations, and at least one concrete number, threshold, or example.
2) Use only information present in the source. Do not speculate.
3) No markdown headings, bold, italic, lists, or blockquotes. Continuous prose only.
4) Do not use em dashes or spaced hyphens. Use commas, periods, or semicolons.
5) Do not start paragraphs with labels like "Core Thesis:" — start directly with content.
""".strip()


COMBINED_PROCESSING_JSON_SCHEMA = {
    "name": "combined_processing",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"},
                        "key_ideas": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["term", "definition", "key_ideas"],
                },
                "minItems": 4,
                "maxItems": 20,
            },
            "source_summary": {"type": "string"},
        },
        "required": ["concepts", "source_summary"],
    },
}
