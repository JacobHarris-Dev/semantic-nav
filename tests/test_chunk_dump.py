import tempfile
import unittest
from pathlib import Path

from app.debug.chunk_dump import debug_chunks_path, write_chunks_debug_file


class ChunkDumpTests(unittest.TestCase):
    def test_debug_chunks_path_uses_document_stem_and_chunker(self):
        path = debug_chunks_path("app/data/pdf_example.pdf", "semantic", Path("cache/debug"))

        self.assertEqual(path, Path("cache/debug/pdf_example.semantic.chunks.txt"))

    def test_write_chunks_debug_file_streams_all_chunks(self):
        chunks = [
            {"text": "First chunk.", "start": 0, "end": 12},
            {"text": "Second chunk.", "start": 13, "end": 26},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = write_chunks_debug_file(
                "app/data/sample.txt",
                "fixed",
                chunks,
                debug_dir=Path(temp_dir),
            )
            self.assertTrue(output_path.exists())
            contents = output_path.read_text(encoding="utf-8")

        self.assertIn("Document: app/data/sample.txt", contents)
        self.assertIn("Chunker: fixed", contents)
        self.assertIn("Total chunks: 2", contents)
        self.assertIn("--- Chunk 0 (chars 0-12) ---", contents)
        self.assertIn("First chunk.", contents)
        self.assertIn("--- Chunk 1 (chars 13-26) ---", contents)
        self.assertIn("Second chunk.", contents)


if __name__ == "__main__":
    unittest.main()
