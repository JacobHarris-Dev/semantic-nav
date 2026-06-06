import unittest

from app.main import normalize_confidence_threshold


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


if __name__ == "__main__":
    unittest.main()
