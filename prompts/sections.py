"""Declarative section registry for the cognitive arc.

Section roles, the ledger unit types they consume, their non-overlap rules, and
their length budgets live here in one place rather than being scattered across
prompt strings. The registry is rendered into a contract block that is injected
into the consolidation (per-source) and synthesis (cross-source) prompts, and is
also what the generic audit pass checks each section against.

Ordering follows the cognitive arc:
  thesis -> mechanisms -> tradeoffs -> implications -> open questions -> glossary
  -> retrieval practice
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    field: str          # response field this section populates
    role: str           # one-line cognitive contract
    consumes: tuple[str, ...]   # ledger unit types that feed it
    forbids: str        # explicit non-overlap rule injected into the prompt
    budget: str         # length / shape contract
    audit: tuple[str, ...] = dc_field(default_factory=tuple)  # audit rule keys


# Per-source arc (produced by one consolidation call, mapped onto
# SourceLearningSection fields).
PER_SOURCE_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        id="core_thesis",
        title="Core Thesis",
        field="summary",
        role="State the single load-bearing claim of the source and why it matters.",
        consumes=("foundational",),
        forbids="No mechanism walk-throughs, no term definitions, no examples.",
        budget="One tight paragraph, at most 4 sentences.",
        audit=("budget_paragraph",),
    ),
    SectionSpec(
        id="mechanisms",
        title="Mechanisms & Relationships",
        field="deep_dive",
        role="Explain ONLY how things work: causal steps, dependencies, dynamics.",
        consumes=("mechanism",),
        forbids="Do not restate the thesis. Do not define terms (that is the glossary). No filler transitions.",
        budget="As many paragraphs as the mechanisms require, each making one new point. No padding.",
        audit=("novelty",),
    ),
    SectionSpec(
        id="open_questions",
        title="Open Questions & Limitations",
        field="under_surface",
        role="Surface boundary conditions, failure modes, hidden assumptions, and unresolved gaps.",
        consumes=("boundary", "open_question", "assumption"),
        forbids="Do not repeat mechanism explanations. Do not reassure or summarize.",
        budget="1-2 paragraphs.",
        audit=("budget_paragraph", "novelty"),
    ),
    SectionSpec(
        id="glossary",
        title="Glossary",
        field="key_terms",
        role="Compress key domain terms into quick-reference anchors.",
        consumes=(),
        forbids="Do not re-explain concepts already covered. No analogy unless the term is genuinely abstract.",
        budget="6-12 actual domain terms. At most 2 short sentences total per term.",
        audit=("glossary_budget",),
    ),
    SectionSpec(
        id="retrieval_practice",
        title="Retrieval Practice",
        field="reflection_points",
        role="Pose questions that force transfer, failure analysis, or counterfactual reasoning.",
        consumes=("mechanism", "tradeoff", "open_question"),
        forbids="No questions whose answer is stated verbatim in the source. No generic 'think carefully' prompts.",
        budget="4-6 questions spanning at least 3 distinct cognitive operations.",
        audit=("diversity",),
    ),
)

# Cross-source arc (produced by one synthesis call, mapped onto Combined fields).
CROSS_SOURCE_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        id="comparative_tradeoffs",
        title="Comparative Tradeoffs",
        field="synthesis_text+intersections",
        role=(
            "Compare and contrast the sources along explicit dimensions: convergences, "
            "contradictions, methodological differences, shared assumptions, unresolved gaps."
        ),
        consumes=("tradeoff", "assumption", "mechanism", "foundational"),
        forbids=(
            "Do NOT use the template 'they agree, there is tension, they are complementary, "
            "adaptability matters'. Every comparative item must cite ledger unit ids from at "
            "least two different sources. Do not fabricate connections."
        ),
        budget="3-5 paragraphs of genuine comparison; 2-4 grounded intersections.",
        audit=("grounding",),
    ),
    SectionSpec(
        id="operational_implications",
        title="Operational Implications",
        field="application_scenarios",
        role="Translate the combined understanding into concrete, do-this transfer scenarios.",
        consumes=("mechanism", "tradeoff", "boundary"),
        forbids="No restating the comparison. Each scenario needs specific steps and a real pitfall.",
        budget="2-3 scenarios.",
        audit=(),
    ),
    SectionSpec(
        id="cross_retrieval_practice",
        title="Retrieval Practice",
        field="questions",
        role="Quiz the intersection of sources with transfer/reasoning questions.",
        consumes=("mechanism", "tradeoff", "open_question"),
        forbids="No single-source recall. No recognition-of-summary questions. Vary the answer index.",
        budget="3-5 four-option questions.",
        audit=("diversity",),
    ),
)


def render_contracts(specs: tuple[SectionSpec, ...]) -> str:
    """Render a section-contract block for injection into a system prompt."""
    blocks: list[str] = []
    for spec in specs:
        consumes = ", ".join(spec.consumes) if spec.consumes else "any (derive from prior sections)"
        blocks.append(
            f"- {spec.title} (field: {spec.field})\n"
            f"    Role: {spec.role}\n"
            f"    Consumes ledger units of type: {consumes}\n"
            f"    Must NOT: {spec.forbids}\n"
            f"    Budget: {spec.budget}"
        )
    return "\n".join(blocks)
