import re
from typing import Any, Dict, List

import numpy as np

from app.processing import cloud_llm

def _first_line(text: str, *, max_length: int) -> str:
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if line:
            if len(line) <= max_length:
                return line
            return line[: max_length - 3].rstrip() + "..."
    return ""


def centroid_snippets(
    chunk_indices: List[int],
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    *,
    max_snippets: int = 3,
    max_chars: int = 120,
) -> List[str]:
    if not chunk_indices:
        return []

    subset = embeddings[chunk_indices]
    centroid = subset.mean(axis=0)
    distances = np.linalg.norm(subset - centroid, axis=1)
    ranked_indices = [
        chunk_indices[index]
        for index in np.argsort(distances)
    ]

    snippets: List[str] = []
    seen = set()
    for chunk_index in ranked_indices:
        text = chunks[chunk_index].get("text", "").strip()
        snippet = _first_line(text, max_length=max_chars)
        if not snippet:
            continue
        key = snippet.casefold()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break

    return snippets


def llm_title_from_snippets(snippets: List[str]) -> str:
    if not snippets:
        return ""

    excerpt_block = "\n".join(f"- {snippet}" for snippet in snippets)
    prompt = (
        "Write a short topic title for this document section using 3 to 6 words. "
        "Return only the title, no quotes or punctuation at the end.\n\n"
        f"Section excerpts:\n{excerpt_block}\n\n"
        "Title:"
    )

    try:
        title = cloud_llm.complete(
            prompt,
            max_tokens=30,
            temperature=0,
            metric_label="outline label",
        )
    except RuntimeError:
        return ""

    title = title.strip().strip('"').strip("'")
    title = re.sub(r"\s+", " ", title)
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title
