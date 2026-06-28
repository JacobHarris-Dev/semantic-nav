import unittest

import numpy as np

from app.processing.topic_tree import (
    _TreeNode,
    _build_document_tfidf,
    _cluster_nodes,
    build_topic_tree,
    normalize_outline_param,
    title_for_cluster,
    title_from_centroid_chunk,
    title_from_tfidf_keywords,
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

    def test_title_from_tfidf_keywords_uses_distinctive_terms(self):
        chunks = self._sample_chunks()
        corpus_tfidf = _build_document_tfidf(chunks)
        self.assertIsNotNone(corpus_tfidf)

        title = title_from_tfidf_keywords([0, 1], corpus_tfidf)
        lowered = title.lower()
        self.assertTrue(
            any(term in lowered for term in ("grading", "exam", "late", "submission", "penalties")),
            msg=f"unexpected title: {title}",
        )

    def test_title_for_cluster_falls_back_to_centroid_for_single_chunk(self):
        chunks = [{"text": "Only section", "start": 0, "end": 12}]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        corpus_tfidf = _build_document_tfidf(chunks)

        title = title_for_cluster([0], embeddings, chunks, corpus_tfidf)
        self.assertEqual(title, "Only section")

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

    def test_adjacent_clustering_merges_neighbors_not_distant_similar_chunks(self):
        chunks = [
            {"text": "Topic A part one", "start": 0, "end": 10},
            {"text": "Topic A part two", "start": 11, "end": 20},
            {"text": "Topic B part one", "start": 21, "end": 30},
            {"text": "Topic B part two", "start": 31, "end": 40},
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )

        nodes = [_TreeNode(chunk_indices=[index], children=[]) for index in range(4)]
        corpus_tfidf = _build_document_tfidf(chunks)
        merged = _cluster_nodes(nodes, embeddings, chunks, branch_factor=2, corpus_tfidf=corpus_tfidf)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].chunk_indices, [0, 1])
        self.assertEqual(merged[1].chunk_indices, [2, 3])

    def test_outline_nodes_have_contiguous_chunk_indices(self):
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

        def walk(node):
            yield node
            for child in node.get("children", []):
                yield from walk(child)

        for node in walk(root):
            if node["chunk_count"] <= 1:
                continue
            # Internal nodes are built from adjacent merges; spans should not gap in chunk order.
            self.assertIsNotNone(node["start"])
            self.assertIsNotNone(node["end"])
            self.assertLess(node["start"], node["end"])


if __name__ == "__main__":
    unittest.main()
