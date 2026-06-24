import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans


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


def _merge_nodes(
    nodes: List[_TreeNode],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
) -> _TreeNode:
    chunk_indices: List[int] = []
    for node in nodes:
        chunk_indices.extend(node.chunk_indices)

    merged = _TreeNode(
        chunk_indices=sorted(set(chunk_indices)),
        children=nodes,
    )
    merged.title = title_from_centroid_chunk(merged.chunk_indices, embeddings, chunks)
    return merged


def _cluster_nodes(
    nodes: List[_TreeNode],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    branch_factor: int,
) -> List[_TreeNode]:
    if len(nodes) <= 1:
        return nodes

    cluster_count = max(1, math.ceil(len(nodes) / branch_factor))
    cluster_count = min(cluster_count, len(nodes))

    if cluster_count == 1:
        return [_merge_nodes(nodes, embeddings, chunks)]

    vectors = np.array([_node_centroid(node, embeddings) for node in nodes], dtype=np.float32)
    labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=0).fit_predict(vectors)

    grouped: Dict[int, List[_TreeNode]] = {}
    for label, node in zip(labels, nodes):
        grouped.setdefault(int(label), []).append(node)

    merged_nodes = [
        _merge_nodes(group, embeddings, chunks)
        for group in grouped.values()
    ]
    merged_nodes.sort(key=lambda node: node.chunk_indices[0])
    return merged_nodes


def build_topic_tree(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    *,
    branch_factor: int = 10,
    max_depth: int = 3,
) -> Optional[Dict[str, Any]]:
    """Build a bottom-up topic tree using chunk embeddings and centroid chunk titles."""
    if not chunks:
        return None

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    branch_factor = normalize_outline_param(branch_factor, "branch_factor", minimum=1, maximum=50)
    max_depth = normalize_outline_param(max_depth, "max_depth", minimum=1, maximum=10)

    if len(chunks) == 1:
        root = _TreeNode(chunk_indices=[0], children=[])
        root.title = title_from_centroid_chunk(root.chunk_indices, embeddings, chunks)
        return serialize_topic_tree(root, chunks, max_depth=max_depth)

    nodes = [_TreeNode(chunk_indices=[index], children=[]) for index in range(len(chunks))]
    merge_levels = max_depth - 1

    while len(nodes) > 1 and merge_levels > 0:
        next_nodes = _cluster_nodes(nodes, embeddings, chunks, branch_factor)
        if len(next_nodes) == len(nodes):
            break
        nodes = next_nodes
        merge_levels -= 1

    if len(nodes) > 1:
        root = _merge_nodes(nodes, embeddings, chunks)
    else:
        root = nodes[0]
        if not root.title:
            root.title = title_from_centroid_chunk(root.chunk_indices, embeddings, chunks)

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
