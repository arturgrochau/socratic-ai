from __future__ import annotations

import unittest

from app.generation import _normalize_ids
from app.ingestion import validate_youtube_url


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


if __name__ == "__main__":
    unittest.main()
