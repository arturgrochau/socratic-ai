from __future__ import annotations

from app.generation import generate_tailored_learning
from app.linking import link_source_pair
from app.models import (
    GenerateTailoredLearningResponse,
    ProcessPipelineLinkResult,
    ProcessPipelineResponse,
)
from app.processing import process_source


def run_processing_and_linking(
    video_source_id: int,
    document_source_ids: list[int],
    user_id: str,
) -> ProcessPipelineResponse:
    if video_source_id <= 0:
        raise ValueError("video_source_id must be a positive integer.")

    normalized_document_ids = sorted(
        {source_id for source_id in document_source_ids if source_id > 0}
    )
    if not normalized_document_ids:
        raise ValueError("At least one valid document source_id is required.")

    process_source(video_source_id, user_id)
    processed_source_ids = [video_source_id]

    for document_source_id in normalized_document_ids:
        process_source(document_source_id, user_id)
        processed_source_ids.append(document_source_id)

    linked_pairs: list[ProcessPipelineLinkResult] = []
    for document_source_id in normalized_document_ids:
        link_result = link_source_pair(video_source_id, document_source_id, user_id)
        linked_pairs.append(
            ProcessPipelineLinkResult(
                video_source_id=video_source_id,
                document_source_id=document_source_id,
                candidate_pairs=int(link_result["candidate_pairs"]),
                stored_edges=int(link_result["stored_edges"]),
            )
        )

    return ProcessPipelineResponse(
        status_message="Processing and linking completed successfully.",
        processed_source_ids=processed_source_ids,
        linked_pairs=linked_pairs,
    )


def run_tailored_learning_generation(
    video_source_id: int,
    document_source_ids: list[int],
    user_id: str,
) -> GenerateTailoredLearningResponse:
    return generate_tailored_learning(
        video_source_id=video_source_id,
        document_source_ids=document_source_ids,
        user_id=user_id,
    )
