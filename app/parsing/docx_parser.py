from docx import Document
from app.parsing.base_parser import BaseParser


class DOCXParser(BaseParser):
    """
    Parse docx file and returns raw text
    """
    def parse(self, file_path):
        doc = Document(file_path)

        text = []

        for paragraph in doc.paragraphs:
            text.append(paragraph.text)

        return "\n".join(text)