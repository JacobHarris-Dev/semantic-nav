import unittest
from unittest.mock import patch

import numpy as np

from app.processing.outline_planner import (
    _merge_small_sections,
    _partition_contiguous,
    build_outline_sections,
    plan_outline_offline,
)
from app.processing.topic_tree import (
    _build_document_tfidf,
    build_topic_tree,
    normalize_outline_param,
    title_for_cluster,
    title_from_centroid_chunk,
    title_from_tfidf_keywords,
)
from app.processing.outline_planner import OutlineSection


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

    def _sample_embeddings(self):
        return np.array(
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
        embeddings = self._sample_embeddings()

        root = build_topic_tree(
            chunks,
            embeddings,
            branch_factor=2,
            max_depth=3,
            max_sections=4,
        )

        self.assertIsNotNone(root)
        self.assertEqual(root["chunk_count"], 6)
        self.assertIn("title", root)
        self.assertIn("children", root)
        self.assertGreaterEqual(len(root["children"]), 1)
        self.assertLessEqual(len(root["children"]), 4)

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

    def test_offline_partition_covers_all_chunks(self):
        chunks = self._sample_chunks()
        embeddings = self._sample_embeddings()
        ranges = _partition_contiguous(0, len(chunks) - 1, 3, embeddings)

        covered = [index for group in ranges for index in group]
        self.assertEqual(sorted(covered), list(range(len(chunks))))

    def test_offline_plan_respects_max_sections(self):
        chunks = self._sample_chunks()
        embeddings = self._sample_embeddings()

        _, sections = plan_outline_offline(
            chunks,
            embeddings,
            max_sections=3,
            branch_factor=2,
            min_section_chunks=1,
        )

        self.assertLessEqual(len(sections), 3)
        covered = sorted(index for section in sections for index in section.chunk_indices)
        self.assertEqual(covered, list(range(len(chunks))))

    def test_merge_small_sections_combines_singletons(self):
        sections = [
            OutlineSection(title="A", chunk_indices=[0]),
            OutlineSection(title="B", chunk_indices=[1, 2]),
        ]
        merged = _merge_small_sections(sections, min_chunks=2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(sorted(merged[0].chunk_indices), [0, 1, 2])

    @patch("app.processing.outline_planner.cloud_llm.complete")
    @patch("app.processing.outline_planner.ONLINE", True)
    def test_online_mode_uses_llm_plan_titles(self, mock_complete):
        mock_complete.return_value = (
            '{"title":"Course policies","sections":['
            '{"title":"Grading and attendance","chunk_ids":[0,1,2]},'
            '{"title":"Integrity and accommodations","chunk_ids":[3,4,5]}'
            "]}"
        )

        chunks = self._sample_chunks()
        embeddings = self._sample_embeddings()

        root = build_topic_tree(
            chunks,
            embeddings,
            branch_factor=3,
            max_depth=2,
            max_sections=4,
        )

        self.assertIsNotNone(root)
        self.assertEqual(root["title"], "Course policies")
        self.assertEqual(len(root["children"]), 2)
        self.assertEqual(root["children"][0]["title"], "Grading and attendance")
        mock_complete.assert_called()

    def test_outline_nodes_have_contiguous_chunk_indices(self):
        chunks = self._sample_chunks()
        embeddings = self._sample_embeddings()

        root = build_topic_tree(
            chunks,
            embeddings,
            branch_factor=2,
            max_depth=3,
            max_sections=4,
        )

        def walk(node):
            yield node
            for child in node.get("children", []):
                yield from walk(child)

        for node in walk(root):
            if node["chunk_count"] <= 1:
                continue
            self.assertIsNotNone(node["start"])
            self.assertIsNotNone(node["end"])
            self.assertLess(node["start"], node["end"])

    def test_build_outline_sections_subdivides_large_sections(self):
        chunks = self._sample_chunks()
        embeddings = self._sample_embeddings()

        _, sections = build_outline_sections(
            chunks,
            embeddings,
            max_sections=2,
            branch_factor=2,
            max_depth=2,
            min_section_chunks=1,
        )

        self.assertGreater(len(sections), 0)
        self.assertTrue(any(section.children for section in sections))


if __name__ == "__main__":
    unittest.main()
