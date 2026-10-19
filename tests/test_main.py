import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import (
    app,
    normalize_confidence_threshold,
    pipeline_needs_switch,
    resolve_subquery_confidence_threshold,
)

client = TestClient(app)


class CorsTests(unittest.TestCase):
    def test_status_includes_cors_header_for_allowed_origin(self):
        response = client.get("/status", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_preflight_options_returns_cors_headers(self):
        response = client.options(
            "/search",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
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
                "gatsby.pdf",
                "semantic",
                has_engine=True,
                current_document="app/data/gatsby.pdf",
                current_chunker="semantic",
            )
        )

    def test_switch_when_chunker_changes(self):
        self.assertTrue(
            pipeline_needs_switch(
                "gatsby.pdf",
                "fixed",
                has_engine=True,
                current_document="gatsby.pdf",
                current_chunker="semantic",
            )
        )

    def test_switch_when_document_changes(self):
        self.assertTrue(
            pipeline_needs_switch(
                "history.pdf",
                "semantic",
                has_engine=True,
                current_document="gatsby.pdf",
                current_chunker="semantic",
            )
        )

    def test_switch_when_engine_not_loaded(self):
        self.assertTrue(
            pipeline_needs_switch(
                "gatsby.pdf",
                "semantic",
                has_engine=False,
                current_document="gatsby.pdf",
                current_chunker="semantic",
            )
        )

    def test_no_switch_when_only_query_params_would_change(self):
        self.assertFalse(
            pipeline_needs_switch(
                "gatsby.pdf",
                "semantic",
                has_engine=True,
                current_document="gatsby.pdf",
                current_chunker="semantic",
            )
        )


class PreviewTests(unittest.TestCase):
    def test_preview_meta_returns_page_count(self):
        response = client.get("/preview/meta", params={"document": "gatsby.pdf"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document"], "gatsby.pdf")
        self.assertGreater(payload["page_count"], 0)

    def test_preview_text_returns_document_text(self):
        client.get(
            "/load",
            params={"document": "gatsby.pdf", "chunker_type": "semantic"},
        )
        response = client.get(
            "/preview/text",
            params={"document": "gatsby.pdf", "chunker_type": "semantic"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document"], "gatsby.pdf")
        self.assertGreater(payload["length"], 0)
        self.assertIsInstance(payload["text"], str)
        self.assertIsInstance(payload["page_sections"], list)
        self.assertIsInstance(payload["chunks"], list)
        if payload["page_sections"]:
            self.assertIn("page", payload["page_sections"][0])
            self.assertIn("start", payload["page_sections"][0])
        if payload["chunks"]:
            self.assertIn("start", payload["chunks"][0])
            self.assertIn("end", payload["chunks"][0])
            self.assertIn("text", payload["chunks"][0])

    def test_preview_page_returns_png(self):
        response = client.get(
            "/preview/page",
            params={"document": "gatsby.pdf", "page": 1},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "image/png")
        self.assertGreater(len(response.content), 100)


if __name__ == "__main__":
    unittest.main()
