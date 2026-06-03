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


if __name__ == "__main__":
    unittest.main()
