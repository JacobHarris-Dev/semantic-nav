import unittest

from app.main import resolve_document_path, infer_document_type


class DocumentSelectionTests(unittest.TestCase):
    def test_resolve_document_path_accepts_bare_filename(self):
        resolved = resolve_document_path("gatsby.pdf")

        self.assertTrue(resolved.endswith("app/data/gatsby.pdf"))

    def test_resolve_document_path_accepts_explicit_path(self):
        resolved = resolve_document_path("app/data/easter_island.pdf")

        self.assertTrue(resolved.endswith("app/data/easter_island.pdf"))

    def test_infer_document_type_from_extension(self):
        self.assertEqual(infer_document_type("app/data/md_example.md"), "md")
        self.assertEqual(infer_document_type("app/data/csv_example.csv"), "csv")


if __name__ == "__main__":
    unittest.main()
