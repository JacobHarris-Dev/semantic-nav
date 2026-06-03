import unittest

from app.processing.query_normalizer import normalize_query


class QueryNormalizerTests(unittest.TestCase):
    def test_removes_boilerplate_and_intent_phrases(self):
        raw = "Could you please take me to the part of the document where they discuss machine learning?"
        self.assertEqual(normalize_query(raw), "machine learning")


if __name__ == "__main__":
    unittest.main()
