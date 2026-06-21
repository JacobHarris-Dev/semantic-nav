import unittest

from app.processing.result_merger import merge_subquery_results


class ResultMergerTests(unittest.TestCase):
    def test_reserves_top_hit_per_subquery(self):
        subquery_hits = [
            (
                "food",
                [
                    {"text": "food chunk", "score": 0.9, "start": 0, "end": 10},
                    {"text": "other food", "score": 0.5, "start": 20, "end": 30},
                ],
            ),
            (
                "navigation",
                [
                    {"text": "nav chunk", "score": 0.8, "start": 100, "end": 110},
                ],
            ),
        ]

        merged = merge_subquery_results(subquery_hits, max_results=3)

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["text"], "food chunk")
        self.assertEqual(merged[0]["matched_subquery"], "food")
        self.assertEqual(merged[1]["text"], "nav chunk")
        self.assertEqual(merged[1]["matched_subquery"], "navigation")
        self.assertEqual(merged[2]["text"], "other food")
        self.assertEqual(merged[2]["matched_subquery"], "food")

    def test_deduplicates_by_span(self):
        shared_hit = {"text": "shared chunk", "score": 0.7, "start": 50, "end": 60}
        subquery_hits = [
            ("food", [shared_hit]),
            ("navigation", [dict(shared_hit, score=0.6)]),
        ]

        merged = merge_subquery_results(subquery_hits, max_results=3)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["matched_subquery"], "food")

    def test_respects_max_results(self):
        subquery_hits = [
            (
                "food",
                [
                    {"text": "a", "score": 0.9},
                    {"text": "b", "score": 0.4},
                ],
            ),
            (
                "navigation",
                [
                    {"text": "c", "score": 0.8},
                    {"text": "d", "score": 0.3},
                ],
            ),
        ]

        merged = merge_subquery_results(subquery_hits, max_results=2)

        self.assertEqual(len(merged), 2)
        texts = {item["text"] for item in merged}
        self.assertEqual(texts, {"a", "c"})

    def test_skips_subquery_with_no_hits(self):
        subquery_hits = [
            ("food", []),
            ("navigation", [{"text": "nav chunk", "score": 0.8}]),
        ]

        merged = merge_subquery_results(subquery_hits, max_results=3)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "nav chunk")


if __name__ == "__main__":
    unittest.main()
