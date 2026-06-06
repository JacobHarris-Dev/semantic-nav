import unittest
from unittest.mock import Mock

import numpy as np

from app.embedding.retrieval import Retrieval


class DummyIndex:
    def __init__(self, distances, indices):
        self._distances = np.array([distances], dtype=np.float32)
        self._indices = np.array([indices], dtype=np.int64)

    def search(self, query_vector, k):
        return self._distances, self._indices


class DummyStore:
    def __init__(self, chunks, distances, indices):
        self.chunks = chunks
        self.index = DummyIndex(distances, indices)
        self.model = Mock()
        self.model.encode.return_value = np.zeros((1, 1), dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def test_query_filters_by_confidence_threshold(self):
        chunks = [
            {"text": "chunk one"},
            {"text": "chunk two"},
            {"text": "chunk three"},
        ]
        distances = [1.0, 10.0, 0.1]
        indices = [0, 1, 2]

        store = DummyStore(chunks, distances, indices)
        retrieval = Retrieval(store)

        results = retrieval.query("test", k=3, confidence_threshold=0.3)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "chunk three")
        self.assertEqual(results[1]["text"], "chunk one")

    def test_query_respects_max_results(self):
        chunks = [
            {"text": "one"},
            {"text": "two"},
            {"text": "three"},
            {"text": "four"},
        ]
        distances = [0.1, 0.2, 0.3, 0.4]
        indices = [0, 1, 2, 3]

        store = DummyStore(chunks, distances, indices)
        retrieval = Retrieval(store)

        results = retrieval.query("test", k=4, confidence_threshold=0.0, max_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "one")
        self.assertEqual(results[1]["text"], "two")


if __name__ == "__main__":
    unittest.main()
