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
        role="State the one load-bearing claim in the first sentence, then say why it matters.",
        consumes=("foundational",),
        forbids="No mechanism walk-throughs, no term definitions, no examples, no warm-up sentence.",
        budget="At most 3 sentences. The first sentence is the claim itself.",
        audit=("budget_paragraph",),
    ),
    SectionSpec(
        id="mechanisms",
        title="Mechanisms & Relationships",
        field="deep_dive",
        role=(
            "Select only the 2-4 most important or most surprising mechanisms and explain how each "
            "works. Lead with the least obvious one."
        ),
        consumes=("mechanism",),
        forbids=(
            "Do NOT paraphrase or list every mechanism in the source; pick the few that carry the "
            "most weight. Do not restate the thesis. Do not define terms. No filler transitions."
        ),
        budget="2-4 short paragraphs, one mechanism each. Omit the rest rather than pad.",
        audit=("novelty",),
    ),
    SectionSpec(
        id="open_questions",
        title="Open Questions & Limitations",
        field="under_surface",
        role="Name the 1-3 genuinely unresolved tensions, failure modes, or hidden assumptions, concretely.",
        consumes=("boundary", "open_question", "assumption"),
        forbids="Do not repeat mechanism explanations. Do not reassure or summarize. No generic caveats.",
        budget="1 short paragraph, the few that actually matter.",
        audit=("budget_paragraph", "novelty"),
    ),
    SectionSpec(
        id="glossary",
        title="Key Concepts",
        field="key_terms",
        role="Define only the concepts the source actually teaches or explains.",
        consumes=(),
        forbids=(
            "Skip terms merely mentioned in passing or assumed-known; include a term only if the "
            "source explains it. No analogy unless the term is genuinely abstract."
        ),
        budget="3-8 taught concepts. At most 2 short sentences total per term.",
        audit=("glossary_budget",),
    ),
    SectionSpec(
        id="retrieval_practice",
        title="Retrieval Practice",
        field="reflection_points",
        role="Pose concrete, content-specific questions that force transfer or resolve a real tension in this source.",
        consumes=("mechanism", "tradeoff", "open_question"),
        forbids=(
            "No questions whose answer is stated verbatim. No generic 'think carefully' prompts. "
            "Do NOT reuse template stems like 'What if X changed?', 'What assumptions underlie X?', "
            "or 'What would happen if X weren't done?'; name the actual mechanism or decision."
        ),
        budget="3-5 questions spanning at least 3 distinct cognitive operations.",
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
            "Open with the single most important relationship between the sources, stated directly. "
            "Then give only the few comparisons that genuinely matter (a real contradiction, a "
            "methodological difference that changes conclusions, a shared blind spot)."
        ),
        consumes=("tradeoff", "assumption", "mechanism", "foundational"),
        forbids=(
            "No preamble restating that the sources are similar or different. Do NOT follow the "
            "template 'they agree -> they differ -> limitations -> conclusion'. State concrete "
            "observations, not abstract labels. Every intersection must be grounded in claims from "
            "at least two different sources. Do not fabricate connections, do not pad to fill space."
        ),
        budget="2-3 tight paragraphs; only the 2-3 most significant grounded intersections.",
        audit=("grounding",),
    ),
    SectionSpec(
        id="operational_implications",
        title="Operational Implications",
        field="application_scenarios",
        role="Translate the combined understanding into concrete, do-this transfer scenarios.",
        consumes=("mechanism", "tradeoff", "boundary"),
        forbids="No restating the comparison. Each scenario needs specific steps and a real pitfall.",
        budget="2 scenarios, the most useful ones.",
        audit=(),
    ),
    SectionSpec(
        id="cross_retrieval_practice",
        title="Retrieval Practice",
        field="questions",
        role=(
            "Each question tests whether the learner understood the SPECIFIC comparison just "
            "presented (the key takeaways and intersections) by making them resolve a tension "
            "between the sources or apply the combined insight."
        ),
        consumes=("mechanism", "tradeoff", "open_question"),
        forbids=(
            "No single-source recall. No questions answerable from general domain knowledge without "
            "the synthesis. No recognition-of-summary. No single obviously-correct option; the "
            "distractors are plausible positions. Vary the answer index."
        ),
        budget="3-4 four-option questions, each tied to a stated takeaway or intersection.",
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
