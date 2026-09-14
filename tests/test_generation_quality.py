from __future__ import annotations

import unittest

from app.generation import _derive_title, _ground_intersections, _group_chunks_into_windows
from app.interaction import _truncate_text


class GroupChunksIntoWindowsTests(unittest.TestCase):
    def test_empty_chunks(self) -> None:
        self.assertEqual(_group_chunks_into_windows([]), [])

    def test_small_set_uses_window_3(self) -> None:
        # Heuristic: <=10 chunks group into windows of 3 (fewer ledger calls).
        chunks = ["a", "b", "c", "d"]
        windows = _group_chunks_into_windows(chunks)
        self.assertEqual(len(windows), 2)
        self.assertIn("a", windows[0])
        self.assertIn("b", windows[0])
        self.assertIn("c", windows[0])
        self.assertIn("d", windows[1])

    def test_large_set_uses_window_4(self) -> None:
        # >10 chunks group into windows of 4.
        chunks = [str(i) for i in range(12)]
        windows = _group_chunks_into_windows(chunks)
        self.assertEqual(len(windows), 3)

    def test_explicit_window_size(self) -> None:
        chunks = ["a", "b", "c", "d", "e"]
        windows = _group_chunks_into_windows(chunks, window_size=2)
        self.assertEqual(len(windows), 3)

    def test_single_chunk(self) -> None:
        windows = _group_chunks_into_windows(["only one"])
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], "only one")


class DeriveTitleTests(unittest.TestCase):
    def test_simple_filename(self) -> None:
        self.assertEqual(_derive_title("my_document.pdf", "document"), "My Document")

    def test_filename_with_dashes(self) -> None:
        self.assertEqual(_derive_title("lecture-notes.txt", "document"), "Lecture Notes")

    def test_no_extension(self) -> None:
        self.assertEqual(_derive_title("readme", "document"), "Readme")

    def test_empty_filename(self) -> None:
        result = _derive_title("", "video")
        self.assertEqual(result, "Untitled Video")


class StripSourceArtifactsTests(unittest.TestCase):
    def test_strips_source_numbers(self) -> None:
        from frontend.format import strip_source_artifacts

        out = strip_source_artifacts("As source 2 shows, and source 8 disagrees.")
        self.assertNotIn("source 2", out)
        self.assertNotIn("source 8", out)

    def test_strips_ledger_ids_and_evidence(self) -> None:
        from frontend.format import strip_source_artifacts

        out = strip_source_artifacts(
            'The loop is unstable [u4 | mechanism | source 2] (evidence: "delay causes overshoot").'
        )
        self.assertNotIn("u4", out)
        self.assertNotIn("evidence:", out)
        self.assertNotIn("source 2", out)
        self.assertIn("The loop is unstable", out)

    def test_keeps_plain_prose(self) -> None:
        from frontend.format import strip_source_artifacts

        text = "Feedback delay drives oscillation."
        self.assertEqual(strip_source_artifacts(text), text)


class GroundIntersectionsTests(unittest.TestCase):
    @staticmethod
    def _entry(title: str, source_ids: list[int]) -> dict:
        return {
            "title": title,
            "why_it_matters": "because",
            "integrated_explanation": "explained",
            "attributed_sentences": [
                {"text": f"claim from {sid}", "source_id": sid} for sid in source_ids
            ],
        }

    def test_keeps_two_real_sources(self) -> None:
        out = _ground_intersections([self._entry("ok", [1, 2])], valid_source_ids={1, 2})
        self.assertEqual(len(out), 1)
        self.assertEqual({s.source_id for s in out[0].attributed_sentences}, {1, 2})

    def test_rejects_single_source(self) -> None:
        out = _ground_intersections([self._entry("solo", [1, 1])], valid_source_ids={1, 2})
        self.assertEqual(out, [])

    def test_rejects_fabricated_source_id(self) -> None:
        # Cites two distinct ids but 999 was never loaded -> only one real source -> dropped.
        out = _ground_intersections([self._entry("fake", [1, 999])], valid_source_ids={1, 2})
        self.assertEqual(out, [])

    def test_prunes_fabricated_attribution_but_keeps_grounded(self) -> None:
        # Two real sources + one fabricated: kept, but the fake attribution is removed.
        out = _ground_intersections([self._entry("mixed", [1, 2, 999])], valid_source_ids={1, 2})
        self.assertEqual(len(out), 1)
        self.assertEqual({s.source_id for s in out[0].attributed_sentences}, {1, 2})


class TruncateTextTests(unittest.TestCase):
    def test_short_text_untouched(self) -> None:
        self.assertEqual(_truncate_text("Short.", 100), "Short.")

    def test_prefers_sentence_boundary(self) -> None:
        text = "First idea here. Second idea here. Third idea here."
        out = _truncate_text(text, 30)
        self.assertTrue(out.endswith("..."))
        self.assertLessEqual(len(out), 30)
        # Cuts at a sentence end, not mid-clause.
        self.assertEqual(out, "First idea here....")

    def test_hard_cut_when_first_sentence_overruns(self) -> None:
        text = "Thisisoneverylongunbrokensentencewithnopunctuation that keeps going."
        out = _truncate_text(text, 20)
        self.assertLessEqual(len(out), 20)
        self.assertTrue(out.endswith("..."))


if __name__ == "__main__":
    unittest.main()
