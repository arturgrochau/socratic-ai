from __future__ import annotations

import unittest

from app.json_reliability import parse_json_object, safe_json_loads


class ParseJsonObjectTests(unittest.TestCase):
    def test_parse_json_object_success(self) -> None:
        payload = parse_json_object('{"title": "Test"}', stage_name="source_title")
        self.assertEqual(payload["title"], "Test")

    def test_parse_json_object_empty_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty JSON content"):
            parse_json_object("", stage_name="source_title")

    def test_parse_json_object_malformed_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            parse_json_object('{"title": "oops"', stage_name="source_title")

    def test_parse_json_object_requires_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected JSON object"):
            parse_json_object('["not", "an", "object"]', stage_name="source_title")


class SafeJsonLoadsTests(unittest.TestCase):
    def test_safe_json_loads_valid_list(self) -> None:
        parsed = safe_json_loads('["a", "b"]', default=[])
        self.assertEqual(parsed, ["a", "b"])

    def test_safe_json_loads_type_mismatch_returns_default(self) -> None:
        parsed = safe_json_loads('{"a": 1}', default=[])
        self.assertEqual(parsed, [])

    def test_safe_json_loads_malformed_returns_default(self) -> None:
        parsed = safe_json_loads('{"a":', default={})
        self.assertEqual(parsed, {})

    def test_safe_json_loads_none_returns_default(self) -> None:
        parsed = safe_json_loads(None, default={"fallback": True})
        self.assertEqual(parsed, {"fallback": True})


if __name__ == "__main__":
    unittest.main()
