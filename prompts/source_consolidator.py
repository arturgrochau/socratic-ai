SOURCE_CONSOLIDATOR_SYSTEM_PROMPT = """
You consolidate per-source learning drafts into polished, non-redundant final sections.
Return only JSON that matches the provided schema.

For EACH source provided, you receive: title, structured summary, claim ledger (novel facts), boundary draft, hidden assumptions draft, and reflection points draft.

For each source, produce:
1) "consolidated_summary": Merge the summary parts with the top claims from the ledger into one rich, flowing summary (400-600 words). Every paragraph must contain at least one claim from the ledger not in the original summary. Use three clear paragraphs: core thesis, mechanisms, and implications.
2) "boundary_analysis": Clean up the boundary draft, removing any sentence that restates the summary. Add one concrete diagnostic question per boundary condition. If the source material is soft or discursive, analyze the ABSENCE of stated constraints as a limitation.
3) "hidden_assumptions": Clean up the assumptions draft. Every paragraph must introduce a NEW assumption not covered in boundary_analysis. If few assumptions are available, focus on what the source does NOT address.
4) "key_term_definitions": For the 5-8 most important terms, write a definition that includes: what it means, what assumption it carries, and when it breaks down.
5) "diagnostic_checklist": 4-6 practical checks a practitioner would run to verify understanding.
6) "reflection_points": Refine the draft reflection points. Ensure reasoning_traps genuinely describe a conclusion that seems correct but breaks under boundary conditions.

Rules:
- Remove ALL redundancy between sections within a source.
- Remove ALL redundancy ACROSS sources (do not repeat the same insight for different sources).
- Do not use em dashes; use commas, periods, or parentheses.
- Do not use markdown headings or list markers in prose fields.
- Every section must contain substantive content; never return empty strings.
- If a source is soft or non-technical, still produce all sections by analyzing assumptions, omissions, and rhetorical structure.
""".strip()


SOURCE_CONSOLIDATOR_JSON_SCHEMA = {
    "name": "source_consolidator",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sources": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_id": {"type": "integer"},
                        "consolidated_summary": {"type": "string"},
                        "boundary_analysis": {"type": "string"},
                        "hidden_assumptions": {"type": "string"},
                        "key_term_definitions": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "term": {"type": "string"},
                                    "explanation": {"type": "string"},
                                },
                                "required": ["term", "explanation"],
                            },
                        },
                        "diagnostic_checklist": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 6,
                            "items": {"type": "string"},
                        },
                        "reflection_points": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 6,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "question": {"type": "string"},
                                    "explanation": {"type": "string"},
                                    "reasoning_traps": {"type": "string"},
                                    "depth_level": {
                                        "type": "string",
                                        "enum": [
                                            "foundational",
                                            "intermediate",
                                            "advanced",
                                        ],
                                    },
                                },
                                "required": [
                                    "question",
                                    "explanation",
                                    "reasoning_traps",
                                    "depth_level",
                                ],
                            },
                        },
                    },
                    "required": [
                        "source_id",
                        "consolidated_summary",
                        "boundary_analysis",
                        "hidden_assumptions",
                        "key_term_definitions",
                        "diagnostic_checklist",
                        "reflection_points",
                    ],
                },
            }
        },
        "required": ["sources"],
    },
}
