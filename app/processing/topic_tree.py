import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


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
class _TreeNode:
    chunk_indices: List[int]
    children: List["_TreeNode"] = field(default_factory=list)
    title: str = ""

    def span(self, chunks: List[Dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
        starts = []
        ends = []
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


def _build_document_tfidf(chunks: List[Dict[str, Any]]) -> Optional[_CorpusTfidf]:
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
    chunk_indices: List[int],
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

    keywords: List[str] = []
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


def title_for_cluster(
    chunk_indices: List[int],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    corpus_tfidf: Optional[_CorpusTfidf],
) -> str:
    if len(chunk_indices) == 1:
        return title_from_centroid_chunk(chunk_indices, embeddings, chunks)

    if corpus_tfidf is not None:
        title = title_from_tfidf_keywords(chunk_indices, corpus_tfidf)
        if title:
            return title

    return title_from_centroid_chunk(chunk_indices, embeddings, chunks)


def title_from_centroid_chunk(
    chunk_indices: List[int],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
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


def _node_centroid(node: _TreeNode, embeddings: np.ndarray) -> np.ndarray:
    return embeddings[node.chunk_indices].mean(axis=0)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _merge_nodes(
    nodes: List[_TreeNode],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    corpus_tfidf: Optional[_CorpusTfidf],
) -> _TreeNode:
    chunk_indices: List[int] = []
    for node in nodes:
        chunk_indices.extend(node.chunk_indices)

    merged = _TreeNode(
        chunk_indices=sorted(set(chunk_indices)),
        children=nodes,
    )
    merged.title = title_for_cluster(merged.chunk_indices, embeddings, chunks, corpus_tfidf)
    return merged


def _cluster_nodes(
    nodes: List[_TreeNode],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    branch_factor: int,
    corpus_tfidf: Optional[_CorpusTfidf],
) -> List[_TreeNode]:
    """Merge adjacent nodes bottom-up until the target count, preferring semantic similarity."""
    if len(nodes) <= 1:
        return nodes

    cluster_count = max(1, math.ceil(len(nodes) / branch_factor))
    cluster_count = min(cluster_count, len(nodes))

    if cluster_count == len(nodes):
        return sorted(nodes, key=lambda node: node.chunk_indices[0])

    ordered_nodes = sorted(nodes, key=lambda node: node.chunk_indices[0])

    if cluster_count == 1:
        merged = ordered_nodes[0]
        for next_node in ordered_nodes[1:]:
            merged = _merge_nodes([merged, next_node], embeddings, chunks, corpus_tfidf)
        return [merged]

    while len(ordered_nodes) > cluster_count:
        best_index = 0
        best_similarity = -float("inf")

        for index in range(len(ordered_nodes) - 1):
            left = _node_centroid(ordered_nodes[index], embeddings)
            right = _node_centroid(ordered_nodes[index + 1], embeddings)
            similarity = _cosine_similarity(left, right)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        merged = _merge_nodes(
            [ordered_nodes[best_index], ordered_nodes[best_index + 1]],
            embeddings,
            chunks,
            corpus_tfidf,
        )
        ordered_nodes = (
            ordered_nodes[:best_index] + [merged] + ordered_nodes[best_index + 2 :]
        )

    return ordered_nodes


def build_topic_tree(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    *,
    branch_factor: int = 10,
    max_depth: int = 3,
) -> Optional[Dict[str, Any]]:
    """Build a bottom-up topic tree from adjacent, semantically similar chunk groups."""
    if not chunks:
        return None

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    branch_factor = normalize_outline_param(branch_factor, "branch_factor", minimum=1, maximum=50)
    max_depth = normalize_outline_param(max_depth, "max_depth", minimum=1, maximum=10)
    corpus_tfidf = _build_document_tfidf(chunks)

    if len(chunks) == 1:
        root = _TreeNode(chunk_indices=[0], children=[])
        root.title = title_for_cluster(root.chunk_indices, embeddings, chunks, corpus_tfidf)
        return serialize_topic_tree(root, chunks, max_depth=max_depth)

    nodes = [_TreeNode(chunk_indices=[index], children=[]) for index in range(len(chunks))]
    merge_levels = max_depth - 1

    while len(nodes) > 1 and merge_levels > 0:
        next_nodes = _cluster_nodes(
            nodes,
            embeddings,
            chunks,
            branch_factor,
            corpus_tfidf,
        )
        if len(next_nodes) == len(nodes):
            break
        nodes = next_nodes
        merge_levels -= 1

    if len(nodes) > 1:
        ordered_nodes = sorted(nodes, key=lambda node: node.chunk_indices[0])
        root = ordered_nodes[0]
        for next_node in ordered_nodes[1:]:
            root = _merge_nodes([root, next_node], embeddings, chunks, corpus_tfidf)
    else:
        root = nodes[0]
        if not root.title:
            root.title = title_for_cluster(
                root.chunk_indices,
                embeddings,
                chunks,
                corpus_tfidf,
            )

    return serialize_topic_tree(root, chunks, max_depth=max_depth)


def serialize_topic_tree(
    node: _TreeNode,
    chunks: List[Dict[str, Any]],
    *,
    max_depth: int,
    current_depth: int = 1,
) -> Dict[str, Any]:
    start, end = node.span(chunks)
    payload: Dict[str, Any] = {
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
