import unittest
from unittest.mock import patch

from app.processing.query_rewriter import rewrite_query


class QueryRewriterTests(unittest.TestCase):
    @patch("app.processing.query_rewriter.ONLINE", False)
    def test_rewrite_query_offline_returns_original(self):
        raw = "Take me to the part where migration, hunting, and extinction is mentioned"
        self.assertEqual(rewrite_query(raw), raw)


if __name__ == "__main__":
    unittest.main()
