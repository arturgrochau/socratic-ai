from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadRequestMeta(BaseModel):
    video_filename: str | None = None
    video_url: str | None = None
    document_filenames: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptPayload(BaseModel):
    text: str
    segments: list[TranscriptSegment]


class DocumentPageText(BaseModel):
    page_number: int
    text: str


class VideoIngestionRecord(BaseModel):
    source_id: int
    source_type: Literal["video"] = "video"
    filename: str
    mime_type: str | None = None
    transcript: TranscriptPayload


class DocumentIngestionRecord(BaseModel):
    source_id: int
    source_type: Literal["document"] = "document"
    filename: str
    mime_type: str | None = None
    pages: list[DocumentPageText]


class IngestionResponse(BaseModel):
    message: str
    request: UploadRequestMeta
    video: VideoIngestionRecord | None = None
    documents: list[DocumentIngestionRecord] = Field(default_factory=list)


class ConceptItem(BaseModel):
    term: str
    definition: str
    key_ideas: list[str]


class ConceptExtractionPayload(BaseModel):
    concepts: list[ConceptItem]


class ProcessSourceResponse(BaseModel):
    source_id: int
    source_type: Literal["video", "document"]
    model_name: str
    chunk_count: int
    extraction: ConceptExtractionPayload
    source_summary: str | None = None


class CrossReferenceResult(BaseModel):
    relation_type: Literal[
        "reinforces",
        "new_info",
        "contradiction",
        "partial_overlap",
    ]
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class LinkedConcept(BaseModel):
    concept_id: str
    source_id: int
    source_type: Literal["video", "document"]
    concept_index: int
    term: str
    definition: str
    key_ideas: list[str]


class LinkingEdgeRecord(BaseModel):
    source_concept_id: str
    target_concept_id: str
    source_source_id: int
    target_source_id: int
    relation_type: Literal[
        "reinforces",
        "new_info",
        "contradiction",
        "partial_overlap",
    ]
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str


class RetrievalHit(BaseModel):
    concept_id: str | None = None
    source_id: int
    source_type: Literal["video", "document"]
    field_type: Literal["term", "definition", "key_idea", "raw_chunk"]
    text: str
    score: float
    chunk_type: Literal["transcript", "document"] | None = None
    chunk_index: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    page_number: int | None = None


class RetrievedContext(BaseModel):
    query: str
    source_ids: list[int]
    concept_ids: list[str]
    hits: list[RetrievalHit]
    relationships: list[LinkingEdgeRecord]
    raw_hit_count: int = 0
    concept_hit_count: int = 0


class AskRequest(BaseModel):
    session_id: str
    source_ids: list[int]
    query: str
    top_k: int = Field(default=8, ge=1, le=20)


class AskModelOutput(BaseModel):
    answer: str
    follow_up_question: str | None = None


class AskResponse(BaseModel):
    session_id: str
    answer: str
    follow_up_question: str | None = None
    concept_ids: list[str]
    model_name: str


class InteractionTurnRecord(BaseModel):
    query: str
    answer: str
    follow_up_question: str | None = None


class InteractionSessionState(BaseModel):
    session_id: str
    source_ids: list[int]
    turns: list[InteractionTurnRecord] = Field(default_factory=list)


class ProcessPipelineRequest(BaseModel):
    video_source_id: int | None = None
    document_source_ids: list[int] = Field(default_factory=list)


class ProcessPipelineLinkResult(BaseModel):
    video_source_id: int
    document_source_id: int
    candidate_pairs: int
    stored_edges: int


class ProcessPipelineResponse(BaseModel):
    status_message: str
    processed_source_ids: list[int]
    linked_pairs: list[ProcessPipelineLinkResult]


class GenerateTailoredLearningRequest(BaseModel):
    video_source_id: int | None = None
    document_source_ids: list[int] = Field(default_factory=list)


class ClaimLedgerEntry(BaseModel):
    claim: str
    grounding_quote: str
    claim_type: Literal["mechanism", "constraint", "failure", "tradeoff", "assumption", "implication"]


class ReflectionPoint(BaseModel):
    question: str
    explanation: str
    reasoning_traps: str = ""
    depth_level: Literal["foundational", "intermediate", "advanced"]


class KeyTermExplanation(BaseModel):
    term: str
    layman: str
    technical: str


class SourceLearningSection(BaseModel):
    source_id: int
    source_type: Literal["video", "document"]
    source_name: str
    generated_title: str
    summary_text: str
    deep_dive_text: str
    key_terms: list[str]
    under_surface_explainer: str = ""
    first_principles_synthesis: str = ""
    diagnostic_checklist: list[str] = Field(default_factory=list)
    key_term_explanations: list[KeyTermExplanation] = Field(default_factory=list)
    reflection_points: list[ReflectionPoint]
    low_mechanism_density: bool = False
    model_name: str
    schema_version: int = 6


class AttributedSentence(BaseModel):
    text: str
    source_id: int
    source_type: Literal["video", "document"]
    emphasis_terms: list[str]


class InsightIntersection(BaseModel):
    intersection_title: str
    why_it_matters: str
    integrated_explanation: str
    attributed_sentences: list[AttributedSentence]
    inferred_extension: str | None = None
    inference_label: Literal["inferred_extension"] | None = None


class CrossSourceTension(BaseModel):
    title: str
    source_a_claim: str
    source_b_claim: str
    resolution_or_tradeoff: str


class TransferBridge(BaseModel):
    from_source_id: int
    to_source_id: int
    transfer_mechanism: str
    adaptation_needed: str


class ApplicationScenario(BaseModel):
    scenario_title: str
    scenario_prompt: str
    transfer_steps: list[str]
    common_pitfall: str


class CombinedInsightSection(BaseModel):
    synthesis_text: str
    intersections: list[InsightIntersection] = Field(default_factory=list)
    parallels: list[AttributedSentence] = Field(default_factory=list)
    layman_bridge: str = ""
    comparative_analysis: str = ""
    application_scenarios: list[ApplicationScenario] = Field(default_factory=list)
    model_name: str
    schema_version: int = 5


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer_index: int = Field(ge=0, le=3)
    explanation: str
    source_evidence: list[str] = Field(default_factory=list)


class CombinedQuizSection(BaseModel):
    questions: list[QuizQuestion]
    study_advice: str = ""
    model_name: str
    schema_version: int = 2


class GenerateTailoredLearningResponse(BaseModel):
    status_message: str
    source_ids: list[int]
    video: SourceLearningSection | None = None
    documents: list[SourceLearningSection] = Field(default_factory=list)
    insights: CombinedInsightSection
    quiz: CombinedQuizSection


class APICallUsageRecord(BaseModel):
    user_id: str
    call_stage: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_count: int = 1


class ValidationRunResult(BaseModel):
    stage: str
    passed: bool
    details: str
