import tempfile
import unittest
from pathlib import Path

from app.debug.cluster_dump import debug_clusters_path, write_clusters_debug_file


class ClusterDumpTests(unittest.TestCase):
    def test_debug_clusters_path_uses_document_stem_and_outline_params(self):
        path = debug_clusters_path(
            "app/data/pdf_example.pdf",
            "semantic",
            6,
            2,
            Path("cache/debug"),
        )

        self.assertEqual(
            path,
            Path("cache/debug/pdf_example.semantic.bf6.d2.clusters.txt"),
        )

    def test_write_clusters_debug_file_streams_outline_tree(self):
        root = {
            "title": "Course overview",
            "chunk_count": 4,
            "start": 0,
            "end": 100,
            "children": [
                {
                    "title": "Grading policy",
                    "chunk_count": 2,
                    "start": 0,
                    "end": 40,
                },
                {
                    "title": "Attendance policy",
                    "chunk_count": 2,
                    "start": 41,
                    "end": 100,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = write_clusters_debug_file(
                "app/data/sample.txt",
                "semantic",
                branch_factor=6,
                max_depth=2,
                chunk_count=4,
                root=root,
                debug_dir=Path(temp_dir),
            )
            self.assertTrue(output_path.exists())
            contents = output_path.read_text(encoding="utf-8")

        self.assertIn("Document: app/data/sample.txt", contents)
        self.assertIn("Chunker: semantic", contents)
        self.assertIn("Branch factor: 6", contents)
        self.assertIn("Max depth: 2", contents)
        self.assertIn("Total chunks: 4", contents)
        self.assertIn("Title: Course overview", contents)
        self.assertIn("Title: Grading policy", contents)
        self.assertIn("--- Cluster (chars 0-40, 2 chunks) ---", contents)


if __name__ == "__main__":
    unittest.main()
