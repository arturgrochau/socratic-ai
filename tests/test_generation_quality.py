from __future__ import annotations

import unittest

from app.generation import _group_chunks_into_windows, _derive_title


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


if __name__ == "__main__":
    unittest.main()
