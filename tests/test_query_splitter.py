import unittest
from unittest.mock import patch

from app.processing.query_splitter import parse_subqueries, split_query


class ParseSubqueriesTests(unittest.TestCase):
    def test_parses_json_array(self):
        self.assertEqual(
            parse_subqueries('["food", "navigation"]', "fallback"),
            ["food", "navigation"],
        )

    def test_strips_markdown_fences(self):
        self.assertEqual(
            parse_subqueries('```json\n["food", "navigation"]\n```', "fallback"),
            ["food", "navigation"],
        )

    def test_deduplicates_case_insensitively(self):
        self.assertEqual(
            parse_subqueries('["Food", "food", "Navigation"]', "fallback"),
            ["Food", "Navigation"],
        )

    def test_falls_back_on_invalid_json(self):
        self.assertEqual(parse_subqueries("not json", "food and navigation"), ["food and navigation"])

    def test_falls_back_on_empty_array(self):
        self.assertEqual(parse_subqueries("[]", "single topic"), ["single topic"])

    def test_falls_back_on_non_string_items(self):
        self.assertEqual(parse_subqueries("[1, 2]", "single topic"), ["single topic"])


class SplitQueryTests(unittest.TestCase):
    @patch("app.processing.query_splitter.ONLINE", False)
    def test_offline_returns_single_query_unchanged(self):
        self.assertEqual(split_query("food and navigation"), ["food and navigation"])

    @patch("app.processing.query_splitter.ONLINE", True)
    @patch("app.processing.query_splitter.cloud_llm.complete")
    def test_online_splits_multi_topic_query(self, mock_complete):
        mock_complete.return_value = '["food", "navigation"]'
        self.assertEqual(split_query("food and navigation"), ["food", "navigation"])

    @patch("app.processing.query_splitter.ONLINE", True)
    @patch("app.processing.query_splitter.cloud_llm.complete")
    def test_online_falls_back_when_llm_fails(self, mock_complete):
        mock_complete.side_effect = RuntimeError("missing key")
        self.assertEqual(split_query("food and navigation"), ["food and navigation"])


if __name__ == "__main__":
    unittest.main()
