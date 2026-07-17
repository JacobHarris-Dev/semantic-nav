"""Document preview helpers for the reader UI and search result highlighting.

Used by GET /preview/meta, /preview/text, and /preview/page in main.py.
Search hits are mapped back into the document via locate_text_span().
"""

from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

import fitz

from app.parsing.md_parser import read_markdown
from app.parsing.pdf_parser import extract_text_from_pdf
from app.parsing.txt_parser import read_txt

_PREVIEW_DOC_CACHE: dict[tuple[str, str], fitz.Document] = {}
_RENDER_DPI = 144
_BASE_SCALE = _RENDER_DPI / 72


def resolve_document_path(document_name: str) -> str:
    candidate = Path(document_name)

    if candidate.exists():
        return candidate.as_posix()

    fallback = Path("app/data") / document_name

    if fallback.exists():
        return fallback.as_posix()

    raise FileNotFoundError(f"Could not find document: {document_name}")


def infer_document_type(document_path: str) -> str:
    suffix = Path(document_path).suffix.lower()

    document_types = {
        ".csv": "csv",
        ".docx": "docx",
        ".md": "md",
        ".pdf": "pdf",
        ".txt": "txt",
    }

    try:
        return document_types[suffix]
    except KeyError as exc:
        raise ValueError(f"Unsupported document type: {document_path}") from exc


def extract_document_text(document_name: str, document_type: str | None = None) -> str:
    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)

    if selected_type == "pdf":
        return extract_text_from_pdf(resolved_path)

    if selected_type == "txt":
        return read_txt(resolved_path)

    if selected_type == "md":
        return read_markdown(resolved_path)

    if selected_type == "csv":
        from app.parsing.csv_parser import CSVParser

        return CSVParser().parse(resolved_path)

    if selected_type == "docx":
        from app.parsing.docx_parser import DOCXParser

        return DOCXParser().parse(resolved_path)

    raise ValueError(f"Unsupported document type: {selected_type}")


def _text_to_fitz_document(text: str) -> fitz.Document:
    story_html = (
        '<body style="font-family: sans-serif; font-size: 11pt; line-height: 1.45;">'
        f'<pre style="white-space: pre-wrap; margin: 0;">{html.escape(text)}</pre>'
        "</body>"
    )
    story = fitz.Story(html=story_html)
    mediabox = fitz.paper_rect("letter")
    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    more = 1

    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(mediabox)
        story.draw(device)
        writer.end_page()

    writer.close()
    return fitz.open(stream=buffer.getvalue(), filetype="pdf")


def _open_preview_document(resolved_path: str, document_type: str) -> fitz.Document:
    cache_key = (resolved_path, document_type)
    cached = _PREVIEW_DOC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if document_type == "pdf":
        document = fitz.open(resolved_path)
    else:
        text = extract_document_text(resolved_path, document_type)
        document = _text_to_fitz_document(text)

    _PREVIEW_DOC_CACHE[cache_key] = document
    return document


def get_preview_text(document_name: str, document_type: str | None = None) -> dict[str, Any]:
    """Return full document text and PDF page boundaries for the reader.

    Chunks are attached separately in main.preview_text via get_pipeline_chunks().
    """
    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)
    text = extract_document_text(document_name, selected_type)
    page_sections = _build_page_sections(resolved_path, selected_type)

    return {
        "document": document_name,
        "document_type": selected_type,
        "text": text,
        "length": len(text),
        "page_sections": page_sections,
    }


def _build_page_sections(resolved_path: str, document_type: str) -> list[dict[str, int]]:
    """Map each PDF page to a character range in the extracted text."""
    if document_type != "pdf":
        return []

    document = fitz.open(resolved_path)
    page_sections: list[dict[str, int]] = []
    offset = 0

    for page_index in range(document.page_count):
        page_text = document[page_index].get_text()
        page_sections.append(
            {
                "page": page_index + 1,
                "start": offset,
                "end": offset + len(page_text),
            }
        )
        offset += len(page_text)

    return page_sections


def locate_text_span(
    document_name: str,
    document_type: str | None,
    start: int | None,
    end: int | None,
) -> dict[str, Any] | None:
    """Map a search hit's character offsets to text and page numbers.

    Returned as result.location in GET /search — used by the frontend to
    scroll to and highlight the matching passage.
    """
    if start is None or end is None or start < 0 or end <= start:
        return None

    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)
    text = extract_document_text(document_name, selected_type)

    if end > len(text):
        end = len(text)
    if start >= len(text):
        return None

    page_sections = _build_page_sections(resolved_path, selected_type)
    overlapping_pages = [
        section["page"]
        for section in page_sections
        if section["end"] > start and section["start"] < end
    ]

    return {
        "start": start,
        "end": end,
        "text": text[start:end],
        "primary_page": overlapping_pages[0] if overlapping_pages else None,
        "pages": overlapping_pages,
    }


def get_preview_meta(document_name: str, document_type: str | None = None) -> dict[str, Any]:
    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)
    document = _open_preview_document(resolved_path, selected_type)
    return {
        "document": document_name,
        "document_type": selected_type,
        "page_count": document.page_count,
    }


def render_preview_page_png(
    document_name: str,
    page: int,
    *,
    document_type: str | None = None,
    scale: float = 1.0,
) -> bytes:
    if page < 1:
        raise ValueError("page must be >= 1")

    if scale <= 0 or scale > 4:
        raise ValueError("scale must be between 0 and 4")

    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)
    document = _open_preview_document(resolved_path, selected_type)

    if page > document.page_count:
        raise ValueError(f"page {page} is out of range for {document.page_count} pages")

    fitz_page = document[page - 1]
    matrix = fitz.Matrix(_BASE_SCALE * scale, _BASE_SCALE * scale)
    pixmap = fitz_page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")
