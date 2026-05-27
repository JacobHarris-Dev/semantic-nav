import unittest

from app.main import resolve_document_path, infer_document_type


class DocumentSelectionTests(unittest.TestCase):
    def test_resolve_document_path_accepts_bare_filename(self):
        resolved = resolve_document_path("pdf_example.pdf")

        self.assertTrue(resolved.endswith("app/data/pdf_example.pdf"))

    def test_resolve_document_path_accepts_explicit_path(self):
        resolved = resolve_document_path("app/data/txt_example.txt")

        self.assertTrue(resolved.endswith("app/data/txt_example.txt"))

    def test_infer_document_type_from_extension(self):
        self.assertEqual(infer_document_type("app/data/md_example.md"), "md")
        self.assertEqual(infer_document_type("app/data/csv_example.csv"), "csv")


if __name__ == "__main__":
    unittest.main()
