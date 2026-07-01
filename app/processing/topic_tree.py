"""Document topic tree built with top-down outline planning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.config import OUTLINE_MAX_SECTIONS, OUTLINE_MIN_SECTION_CHUNKS
from app.processing.outline_planner import (
    OutlineSection,
    build_outline_sections,
    label_outline_sections,
)

_EXTRA_STOPWORDS = {
    "student",
    "students",
    "course",
    "class",
    "section",
    "document",
    "university",
    "college",
    "syllabus",
    "information",
    "chapter",
    "page",
    "also",
    "may",
    "must",
    "will",
    "shall",
}


@dataclass
class _CorpusTfidf:
    feature_names: np.ndarray
    matrix: Any


@dataclass
class TopicNode:
    chunk_indices: list[int]
    children: list[TopicNode] = field(default_factory=list)
    title: str = ""

    def span(self, chunks: list[dict[str, Any]]) -> tuple[int | None, int | None]:
        starts: list[int] = []
        ends: list[int] = []
        for index in self.chunk_indices:
            chunk = chunks[index]
            start = chunk.get("start")
            end = chunk.get("end")
            if start is not None and end is not None:
                starts.append(start)
                ends.append(end)
        if not starts:
            return None, None
        return min(starts), max(ends)


def normalize_outline_param(value: int, name: str, *, minimum: int, maximum: int) -> int:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _build_document_tfidf(chunks: list[dict[str, Any]]) -> _CorpusTfidf | None:
    texts = [chunk.get("text", "").strip() for chunk in chunks]
    if not any(texts):
        return None

    stop_words = list(ENGLISH_STOP_WORDS | _EXTRA_STOPWORDS)
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        min_df=1,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return None

    return _CorpusTfidf(
        feature_names=vectorizer.get_feature_names_out(),
        matrix=matrix,
    )


def title_from_tfidf_keywords(
    chunk_indices: list[int],
    corpus_tfidf: _CorpusTfidf,
    *,
    top_k: int = 3,
    max_length: int = 80,
) -> str:
    if not chunk_indices:
        return ""

    row_indices = sorted(set(chunk_indices))
    subset = corpus_tfidf.matrix[row_indices].sum(axis=0)
    scores = np.asarray(subset).ravel()
    if scores.size == 0 or float(scores.max()) <= 0.0:
        return ""

    keywords: list[str] = []
    for idx in np.argsort(scores)[::-1]:
        if scores[idx] <= 0 or len(keywords) >= top_k:
            break
        keywords.append(str(corpus_tfidf.feature_names[int(idx)]))

    if not keywords:
        return ""

    title = ", ".join(keywords)
    if len(title) <= max_length:
        return title
    return title[: max_length - 3].rstrip() + "..."


def title_from_centroid_chunk(
    chunk_indices: list[int],
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    *,
    max_length: int = 80,
) -> str:
    if not chunk_indices:
        return "Untitled section"

    subset = embeddings[chunk_indices]
    centroid = subset.mean(axis=0)
    distances = np.linalg.norm(subset - centroid, axis=1)
    best_index = chunk_indices[int(np.argmin(distances))]
    text = chunks[best_index].get("text", "").strip()
    if not text:
        return "Untitled section"

    line = ""
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if line:
            break
    if not line:
        return "Untitled section"
    if len(line) <= max_length:
        return line
    return line[: max_length - 3].rstrip() + "..."


def title_for_cluster(
    chunk_indices: list[int],
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    corpus_tfidf: _CorpusTfidf | None,
) -> str:
    if len(chunk_indices) == 1:
        return title_from_centroid_chunk(chunk_indices, embeddings, chunks)

    if corpus_tfidf is not None:
        title = title_from_tfidf_keywords(chunk_indices, corpus_tfidf)
        if title:
            return title

    return title_from_centroid_chunk(chunk_indices, embeddings, chunks)


def _section_to_node(section: OutlineSection) -> TopicNode:
    return TopicNode(
        chunk_indices=list(section.chunk_indices),
        title=section.title,
        children=[_section_to_node(child) for child in section.children],
    )


def build_topic_tree(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    branch_factor: int = 10,
    max_depth: int = 3,
    max_sections: int | None = None,
    min_section_chunks: int | None = None,
) -> dict[str, Any] | None:
    """Build a top-down topic tree from chunk embeddings."""
    if not chunks:
        return None

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    branch_factor = normalize_outline_param(branch_factor, "branch_factor", minimum=1, maximum=50)
    max_depth = normalize_outline_param(max_depth, "max_depth", minimum=1, maximum=10)
    resolved_max_sections = normalize_outline_param(
        max_sections if max_sections is not None else OUTLINE_MAX_SECTIONS,
        "max_sections",
        minimum=2,
        maximum=20,
    )
    resolved_min_section_chunks = normalize_outline_param(
        min_section_chunks if min_section_chunks is not None else OUTLINE_MIN_SECTION_CHUNKS,
        "min_section_chunks",
        minimum=1,
        maximum=10,
    )

    corpus_tfidf = _build_document_tfidf(chunks)

    if len(chunks) == 1:
        root = TopicNode(
            chunk_indices=[0],
            children=[],
            title=title_for_cluster([0], embeddings, chunks, corpus_tfidf),
        )
        return serialize_topic_tree(root, chunks, max_depth=max_depth)

    document_title, sections = build_outline_sections(
        chunks,
        embeddings,
        max_sections=resolved_max_sections,
        branch_factor=branch_factor,
        max_depth=max_depth,
        min_section_chunks=resolved_min_section_chunks,
    )

    def offline_title(chunk_indices: list[int]) -> str:
        return title_for_cluster(chunk_indices, embeddings, chunks, corpus_tfidf)

    label_outline_sections(sections, chunks, embeddings, offline_title)

    if not document_title:
        document_title = title_for_cluster(
            list(range(len(chunks))), embeddings, chunks, corpus_tfidf
        )

    root = TopicNode(
        chunk_indices=list(range(len(chunks))),
        title=document_title,
        children=[_section_to_node(section) for section in sections],
    )
    return serialize_topic_tree(root, chunks, max_depth=max_depth)


def serialize_topic_tree(
    node: TopicNode,
    chunks: list[dict[str, Any]],
    *,
    max_depth: int,
    current_depth: int = 1,
) -> dict[str, Any]:
    """Convert a TopicNode tree to a JSON-serializable dict."""
    start, end = node.span(chunks)
    payload: dict[str, Any] = {
        "title": node.title or "Untitled section",
        "chunk_count": len(node.chunk_indices),
        "start": start,
        "end": end,
    }
    if node.children and current_depth < max_depth:
        payload["children"] = [
            serialize_topic_tree(
                child,
                chunks,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )
            for child in node.children
        ]
    return payload
