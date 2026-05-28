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


class RetrievalHit(BaseModel):
    source_id: int
    source_type: Literal["video", "document"]
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
    hits: list[RetrievalHit]
    raw_hit_count: int = 0


class AskRequest(BaseModel):
    session_id: str
    source_ids: list[int]
    query: str
    top_k: int = Field(default=8, ge=1, le=20)


class AskModelOutput(BaseModel):
    answer: str


class AskResponse(BaseModel):
    session_id: str
    answer: str
    model_name: str


class InteractionTurnRecord(BaseModel):
    query: str
    answer: str


class InteractionSessionState(BaseModel):
    session_id: str
    source_ids: list[int]
    turns: list[InteractionTurnRecord] = Field(default_factory=list)


class GenerateTailoredLearningRequest(BaseModel):
    video_source_id: int | None = None
    document_source_ids: list[int] = Field(default_factory=list)


class ReflectionPoint(BaseModel):
    question: str
    explanation: str
    depth_level: Literal["foundational", "intermediate", "advanced"]


class KeyTermExplanation(BaseModel):
    """v2.1: single layman-tone explanation. The pre-v2.1 `layman`/`technical`
    pair is coalesced into `explanation` when reading old cache rows."""

    term: str
    explanation: str


class LearningSection(BaseModel):
    """A content-driven section in a per-source learning artifact.

    The title is generated from the material (e.g. "Why behavioral logs
    predict churn") rather than a fixed scaffold label."""

    title: str
    body: str


class SourceLearningSection(BaseModel):
    source_id: int
    source_type: Literal["video", "document"]
    source_name: str
    generated_title: str
    # v2.1: dynamic, content-driven sections. The fixed-label fields below
    # are kept for backward read of v2.0 cached rows.
    sections: list[LearningSection] = Field(default_factory=list)
    summary_text: str = ""
    deep_dive_text: str = ""
    under_surface_explainer: str = ""
    key_terms: list[str] = Field(default_factory=list)
    key_term_explanations: list[KeyTermExplanation] = Field(default_factory=list)
    reflection_points: list[ReflectionPoint]
    model_name: str
    schema_version: int = 13


class InsightIntersection(BaseModel):
    title: str
    why_it_matters: str
    integrated_explanation: str


class ApplicationScenario(BaseModel):
    scenario_title: str
    scenario_prompt: str
    transfer_steps: list[str]
    common_pitfall: str


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer_index: int = Field(ge=0, le=3)
    explanation: str
    # v2.1 additions for interactive quiz feedback.
    # option_rationales[i] is a 1-2 sentence note: for the correct option,
    # a justification; for wrong options, the misconception that picking
    # that option represents.
    option_rationales: list[str] = Field(default_factory=list)
    # deeper_why extends the explanation for users who got it right —
    # mechanism, edge case, or transfer to a new scenario.
    deeper_why: str = ""


class CombinedInsightSection(BaseModel):
    synthesis_text: str
    intersections: list[InsightIntersection] = Field(default_factory=list)
    application_scenarios: list[ApplicationScenario] = Field(default_factory=list)
    model_name: str
    schema_version: int = 6


class CombinedQuizSection(BaseModel):
    questions: list[QuizQuestion]
    model_name: str
    schema_version: int = 3


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
