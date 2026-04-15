from __future__ import annotations

import unittest
from unittest.mock import patch

from app.generation import _normalize_ids
from app.ingestion import validate_youtube_url
from app.workflow import run_processing_and_linking


class NormalizeIdsTests(unittest.TestCase):
    def test_normalize_ids_video_only(self) -> None:
        video_id, document_ids = _normalize_ids(9, [])
        self.assertEqual(video_id, 9)
        self.assertEqual(document_ids, [])

    def test_normalize_ids_documents_only(self) -> None:
        video_id, document_ids = _normalize_ids(None, [3, 3, 8])
        self.assertIsNone(video_id)
        self.assertEqual(document_ids, [3, 8])

    def test_normalize_ids_requires_at_least_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one valid source_id is required"):
            _normalize_ids(None, [])


class ValidateYoutubeUrlTests(unittest.TestCase):
    def test_validate_youtube_watch_url(self) -> None:
        result = validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("youtube.com/watch", result)

    def test_validate_youtube_short_url(self) -> None:
        result = validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        self.assertIn("youtu.be", result)

    def test_validate_youtube_url_rejects_non_youtube(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-video YouTube URLs"):
            validate_youtube_url("https://example.com/video.mp4")


class WorkflowOptionalSourceTests(unittest.TestCase):
    @patch("app.workflow.link_source_pair")
    @patch("app.workflow.process_source")
    def test_docs_only_skips_linking(self, mock_process_source, mock_link_source_pair) -> None:
        response = run_processing_and_linking(
            video_source_id=None,
            document_source_ids=[11, 15],
            user_id="demo-user",
        )

        self.assertEqual(response.processed_source_ids, [11, 15])
        self.assertEqual(response.linked_pairs, [])
        self.assertEqual(mock_process_source.call_count, 2)
        mock_link_source_pair.assert_not_called()

    @patch("app.workflow.link_source_pair")
    @patch("app.workflow.process_source")
    def test_mixed_mode_still_links_video_to_documents(self, mock_process_source, mock_link_source_pair) -> None:
        mock_link_source_pair.return_value = {"candidate_pairs": 4, "stored_edges": 2}

        response = run_processing_and_linking(
            video_source_id=7,
            document_source_ids=[5, 9],
            user_id="demo-user",
        )

        self.assertEqual(response.processed_source_ids, [7, 5, 9])
        self.assertEqual(len(response.linked_pairs), 2)
        self.assertEqual(mock_process_source.call_count, 3)
        self.assertEqual(mock_link_source_pair.call_count, 2)

    def test_workflow_requires_at_least_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one valid source_id is required"):
            run_processing_and_linking(
                video_source_id=None,
                document_source_ids=[],
                user_id="demo-user",
            )


if __name__ == "__main__":
    unittest.main()
