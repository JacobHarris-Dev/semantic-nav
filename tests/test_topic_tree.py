import unittest

import numpy as np

from app.processing.topic_tree import (
    build_topic_tree,
    normalize_outline_param,
    title_from_centroid_chunk,
)


class TopicTreeTests(unittest.TestCase):
    def _sample_chunks(self):
        return [
            {"text": "Grading policy and exam weights", "start": 0, "end": 40},
            {"text": "Late submission penalties apply", "start": 41, "end": 80},
            {"text": "Attendance requirements for labs", "start": 81, "end": 120},
            {"text": "Academic integrity and plagiarism", "start": 121, "end": 160},
            {"text": "AI tool usage restrictions", "start": 161, "end": 200},
            {"text": "Disability accommodation requests", "start": 201, "end": 240},
        ]

    def test_title_from_centroid_chunk_uses_nearest_chunk(self):
        chunks = self._sample_chunks()
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        title = title_from_centroid_chunk([0, 1], embeddings, chunks)
        self.assertIn("Grading", title)

    def test_build_topic_tree_returns_nested_structure(self):
        chunks = self._sample_chunks()
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.9],
            ],
            dtype=np.float32,
        )

        root = build_topic_tree(
            chunks,
            embeddings,
            branch_factor=2,
            max_depth=3,
        )

        self.assertIsNotNone(root)
        self.assertEqual(root["chunk_count"], 6)
        self.assertIn("title", root)
        self.assertIn("children", root)
        self.assertGreaterEqual(len(root["children"]), 1)

    def test_build_topic_tree_handles_single_chunk(self):
        chunks = [{"text": "Only section", "start": 0, "end": 12}]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

        root = build_topic_tree(chunks, embeddings, branch_factor=10, max_depth=3)

        self.assertEqual(root["title"], "Only section")
        self.assertEqual(root["chunk_count"], 1)
        self.assertNotIn("children", root)

    def test_build_topic_tree_returns_none_for_empty_document(self):
        root = build_topic_tree([], np.empty((0, 2), dtype=np.float32))
        self.assertIsNone(root)

    def test_normalize_outline_param_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            normalize_outline_param(0, "branch_factor", minimum=1, maximum=50)


if __name__ == "__main__":
    unittest.main()
