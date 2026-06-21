import unittest
from unittest.mock import patch

from app.main import (
    normalize_confidence_threshold,
    pipeline_needs_switch,
    resolve_subquery_confidence_threshold,
)


class MainTests(unittest.TestCase):
    def test_normalize_confidence_threshold_accepts_percentage(self):
        self.assertEqual(normalize_confidence_threshold(20), 0.2)
        self.assertEqual(normalize_confidence_threshold(100), 1.0)

    def test_normalize_confidence_threshold_accepts_decimal(self):
        self.assertEqual(normalize_confidence_threshold(0.2), 0.2)
        self.assertEqual(normalize_confidence_threshold(1.0), 1.0)

    def test_normalize_confidence_threshold_rejects_negative(self):
        with self.assertRaises(ValueError):
            normalize_confidence_threshold(-1)

    def test_normalize_confidence_threshold_rejects_over_100(self):
        with self.assertRaises(ValueError):
            normalize_confidence_threshold(101)


class SubqueryConfidenceThresholdTests(unittest.TestCase):
    def test_uses_api_value_when_provided(self):
        self.assertEqual(
            resolve_subquery_confidence_threshold(40, 0.3),
            0.4,
        )

    @patch("app.main.SUBQUERY_CONFIDENCE_THRESHOLD", 25)
    def test_uses_env_default_when_api_value_missing(self):
        self.assertEqual(
            resolve_subquery_confidence_threshold(None, 0.3),
            0.25,
        )

    @patch("app.main.SUBQUERY_CONFIDENCE_THRESHOLD", None)
    def test_falls_back_to_main_threshold(self):
        self.assertEqual(
            resolve_subquery_confidence_threshold(None, 0.3),
            0.3,
        )


class PipelineSwitchTests(unittest.TestCase):
    def test_no_switch_when_same_document_alias_and_chunker(self):
        self.assertFalse(
            pipeline_needs_switch(
                "pdf_example.pdf",
                "semantic",
                has_engine=True,
                current_document="app/data/pdf_example.pdf",
                current_chunker="semantic",
            )
        )

    def test_switch_when_chunker_changes(self):
        self.assertTrue(
            pipeline_needs_switch(
                "pdf_example.pdf",
                "fixed",
                has_engine=True,
                current_document="pdf_example.pdf",
                current_chunker="semantic",
            )
        )

    def test_switch_when_document_changes(self):
        self.assertTrue(
            pipeline_needs_switch(
                "txt_example.txt",
                "semantic",
                has_engine=True,
                current_document="pdf_example.pdf",
                current_chunker="semantic",
            )
        )

    def test_switch_when_engine_not_loaded(self):
        self.assertTrue(
            pipeline_needs_switch(
                "pdf_example.pdf",
                "semantic",
                has_engine=False,
                current_document="pdf_example.pdf",
                current_chunker="semantic",
            )
        )

    def test_no_switch_when_only_query_params_would_change(self):
        self.assertFalse(
            pipeline_needs_switch(
                "pdf_example.pdf",
                "semantic",
                has_engine=True,
                current_document="pdf_example.pdf",
                current_chunker="semantic",
            )
        )


if __name__ == "__main__":
    unittest.main()
