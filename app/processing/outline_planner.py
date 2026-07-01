"""Top-down document outline planning (LLM online, embedding boundaries offline)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import ONLINE
from app.processing import cloud_llm
from app.processing.topic_labeler import llm_title_from_snippets

_PREVIEW_CHARS = 100


@dataclass
class OutlineSection:
    title: str
    chunk_indices: list[int]
    children: list[OutlineSection] = field(default_factory=list)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _first_line_preview(text: str, *, max_chars: int = _PREVIEW_CHARS) -> str:
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        if len(line) <= max_chars:
            return line
        return line[: max_chars - 3].rstrip() + "..."
    return ""


def build_chunk_index(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks):
        preview = _first_line_preview(chunk.get("text", ""))
        lines.append(f"{index}: {preview}")
    return "\n".join(lines)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _adjacent_similarities(embeddings: np.ndarray) -> list[float]:
    if len(embeddings) < 2:
        return []
    return [
        _cosine(embeddings[index], embeddings[index + 1])
        for index in range(len(embeddings) - 1)
    ]


def _contiguous_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(set(indices))
    ranges: list[list[int]] = []
    current = [ordered[0]]
    for index in ordered[1:]:
        if index == current[-1] + 1:
            current.append(index)
        else:
            ranges.append(current)
            current = [index]
    ranges.append(current)
    return ranges


def _partition_contiguous(
    start: int,
    end: int,
    section_count: int,
    embeddings: np.ndarray,
) -> list[list[int]]:
    """Partition inclusive chunk range [start, end] into contiguous sections."""
    length = end - start + 1
    if section_count <= 1 or length <= 1:
        return [list(range(start, end + 1))]

    section_count = min(section_count, length)
    local_embeddings = embeddings[start : end + 1]
    sims = _adjacent_similarities(local_embeddings)

    # dp[i][j] = best score partitioning first i local chunks into j sections
    dp = [[-1.0] * (section_count + 1) for _ in range(length + 1)]
    back: list[list[int | None]] = [[None] * (section_count + 1) for _ in range(length + 1)]
    dp[0][0] = 0.0

    for size in range(1, length + 1):
        for parts in range(1, min(size, section_count) + 1):
            best_score = -1.0
            best_split: int | None = None
            for split in range(parts - 1, size):
                previous = dp[split][parts - 1]
                if previous < 0:
                    continue
                boundary_score = 0.0 if split == 0 else 1.0 - sims[split - 1]
                total = previous + boundary_score
                if total > best_score:
                    best_score = total
                    best_split = split
            dp[size][parts] = best_score
            back[size][parts] = best_split

    if dp[length][section_count] < 0:
        step = length / section_count
        ranges = []
        cursor = start
        for part in range(section_count):
            next_end = start + int(round(step * (part + 1))) - 1
            if part == section_count - 1:
                next_end = end
            ranges.append(list(range(cursor, next_end + 1)))
            cursor = next_end + 1
        return [chunk_ids for chunk_ids in ranges if chunk_ids]

    splits: list[int] = []
    size = length
    parts = section_count
    while parts > 0:
        split = back[size][parts]
        if split is None:
            break
        if split > 0:
            splits.append(start + split)
        size = split if split is not None else 0
        parts -= 1

    boundaries = sorted(set(splits))
    ranges: list[list[int]] = []
    range_start = start
    for boundary in boundaries:
        ranges.append(list(range(range_start, boundary)))
        range_start = boundary
    ranges.append(list(range(range_start, end + 1)))
    return [chunk_ids for chunk_ids in ranges if chunk_ids]


def _target_section_count(chunk_count: int, *, max_sections: int, branch_factor: int) -> int:
    desired = math.ceil(chunk_count / branch_factor)
    return max(1, min(max_sections, desired, chunk_count))


def _merge_small_sections(
    sections: list[OutlineSection],
    *,
    min_chunks: int,
) -> list[OutlineSection]:
    if min_chunks <= 1 or len(sections) <= 1:
        return sections

    merged = list(sections)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, section in enumerate(merged):
            if len(section.chunk_indices) >= min_chunks:
                continue
            if index == 0:
                neighbor_index = 1
            elif index == len(merged) - 1:
                neighbor_index = index - 1
            else:
                left_size = len(merged[index - 1].chunk_indices)
                right_size = len(merged[index + 1].chunk_indices)
                neighbor_index = index - 1 if left_size <= right_size else index + 1

            neighbor = merged[neighbor_index]
            combined_indices = sorted(set(neighbor.chunk_indices + section.chunk_indices))
            combined_title = neighbor.title if len(neighbor.chunk_indices) >= len(section.chunk_indices) else section.title
            merged[neighbor_index] = OutlineSection(
                title=combined_title,
                chunk_indices=combined_indices,
                children=neighbor.children or section.children,
            )
            merged.pop(index)
            changed = True
            break

    return merged


def _cap_section_count(
    sections: list[OutlineSection],
    *,
    max_sections: int,
) -> list[OutlineSection]:
    merged = list(sections)
    while len(merged) > max_sections and len(merged) > 1:
        smallest_index = min(
            range(len(merged)),
            key=lambda index: len(merged[index].chunk_indices),
        )
        if smallest_index == 0:
            neighbor_index = 1
        elif smallest_index == len(merged) - 1:
            neighbor_index = smallest_index - 1
        else:
            left_size = len(merged[smallest_index - 1].chunk_indices)
            right_size = len(merged[smallest_index + 1].chunk_indices)
            neighbor_index = (
                smallest_index - 1 if left_size <= right_size else smallest_index + 1
            )

        first_index, second_index = sorted([smallest_index, neighbor_index])
        section = merged.pop(second_index)
        neighbor = merged[first_index]
        merged[first_index] = OutlineSection(
            title=neighbor.title,
            chunk_indices=sorted(set(neighbor.chunk_indices + section.chunk_indices)),
            children=neighbor.children,
        )

    return merged


def _offline_sections_for_range(
    chunk_indices: list[int],
    embeddings: np.ndarray,
    *,
    max_sections: int,
    branch_factor: int,
) -> list[list[int]]:
    if not chunk_indices:
        return []

    start = min(chunk_indices)
    end = max(chunk_indices)
    section_count = _target_section_count(
        len(chunk_indices),
        max_sections=max_sections,
        branch_factor=branch_factor,
    )
    return _partition_contiguous(start, end, section_count, embeddings)


def plan_outline_offline(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    max_sections: int,
    branch_factor: int,
    min_section_chunks: int,
) -> tuple[str, list[OutlineSection]]:
    chunk_count = len(chunks)
    section_count = _target_section_count(
        chunk_count,
        max_sections=max_sections,
        branch_factor=branch_factor,
    )
    ranges = _partition_contiguous(0, chunk_count - 1, section_count, embeddings)
    sections = [
        OutlineSection(title="", chunk_indices=chunk_ids)
        for chunk_ids in ranges
    ]
    sections = _merge_small_sections(sections, min_chunks=min_section_chunks)
    sections = _cap_section_count(sections, max_sections=max_sections)
    return "", sections


def _build_top_level_prompt(
    chunk_index: str,
    *,
    chunk_count: int,
    max_sections: int,
    min_section_chunks: int,
) -> str:
    min_sections = max(1, min(max_sections, math.ceil(chunk_count / max(max_sections, 1))))
    return (
        "Build a concise document outline from numbered chunk previews.\n"
        f"Create between {min_sections} and {max_sections} top-level sections.\n"
        "Rules:\n"
        "- Every chunk id from 0 to "
        f"{chunk_count - 1} must appear exactly once\n"
        "- Keep chunk ids contiguous within each section when possible\n"
        f"- Avoid sections with fewer than {min_section_chunks} chunks unless unavoidable\n"
        "- Use short section titles (3 to 8 words)\n"
        '- Return ONLY JSON: {"title":"Document title","sections":[{"title":"...","chunk_ids":[0,1]}]}\n\n'
        f"Chunks:\n{chunk_index}"
    )


def _parse_llm_outline(
    raw_output: str,
    chunk_count: int,
) -> tuple[str, list[OutlineSection]] | None:
    cleaned = _strip_markdown_fences(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    document_title = parsed.get("title", "")
    if not isinstance(document_title, str):
        document_title = ""
    document_title = document_title.strip()

    raw_sections = parsed.get("sections", [])
    if not isinstance(raw_sections, list) or not raw_sections:
        return None

    assigned: set[int] = set()
    sections: list[OutlineSection] = []

    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        chunk_ids = item.get("chunk_ids", [])
        if not isinstance(title, str) or not isinstance(chunk_ids, list):
            continue

        indices: list[int] = []
        for value in chunk_ids:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                index = value
            elif isinstance(value, float) and value.is_integer():
                index = int(value)
            else:
                continue
            if 0 <= index < chunk_count and index not in assigned:
                indices.append(index)
                assigned.add(index)

        if not indices:
            continue

        for run in _contiguous_ranges(indices):
            sections.append(
                OutlineSection(
                    title=title.strip() or "Untitled section",
                    chunk_indices=run,
                )
            )

    missing = [index for index in range(chunk_count) if index not in assigned]
    if missing:
        if sections:
            sections[-1].chunk_indices = sorted(set(sections[-1].chunk_indices + missing))
        else:
            sections.append(OutlineSection(title="Untitled section", chunk_indices=missing))

    if not sections:
        return None

    return document_title, sections


def plan_outline_online(
    chunks: list[dict[str, Any]],
    *,
    max_sections: int,
    min_section_chunks: int,
) -> tuple[str, list[OutlineSection]] | None:
    chunk_count = len(chunks)
    if chunk_count == 0:
        return None

    prompt = _build_top_level_prompt(
        build_chunk_index(chunks),
        chunk_count=chunk_count,
        max_sections=max_sections,
        min_section_chunks=min_section_chunks,
    )

    try:
        raw_output = cloud_llm.complete(
            prompt,
            max_tokens=800,
            temperature=0,
            metric_label="outline plan",
        )
    except RuntimeError:
        return None

    parsed = _parse_llm_outline(raw_output, chunk_count)
    if parsed is None:
        return None

    document_title, sections = parsed
    sections = _merge_small_sections(sections, min_chunks=min_section_chunks)
    sections = _cap_section_count(sections, max_sections=max_sections)
    return document_title, sections


def _subdivide_section(
    section: OutlineSection,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    max_sections: int,
    branch_factor: int,
    min_section_chunks: int,
) -> list[OutlineSection]:
    if len(section.chunk_indices) <= branch_factor:
        return [section]

    if ONLINE:
        local_index = build_chunk_index([chunks[index] for index in section.chunk_indices])
        prompt = _build_top_level_prompt(
            local_index,
            chunk_count=len(section.chunk_indices),
            max_sections=min(max_sections, math.ceil(len(section.chunk_indices) / branch_factor)),
            min_section_chunks=min_section_chunks,
        )
        try:
            raw_output = cloud_llm.complete(
                prompt,
                max_tokens=500,
                temperature=0,
                metric_label="outline subsection",
            )
        except RuntimeError:
            raw_output = ""

        parsed = _parse_llm_outline(raw_output, len(section.chunk_indices)) if raw_output else None
        if parsed is not None:
            _, local_sections = parsed
            return [
                OutlineSection(
                    title=local.title,
                    chunk_indices=[section.chunk_indices[local_index] for local_index in local.chunk_indices],
                )
                for local in local_sections
            ]

    ranges = _offline_sections_for_range(
        section.chunk_indices,
        embeddings,
        max_sections=max_sections,
        branch_factor=branch_factor,
    )
    return [
        OutlineSection(title="", chunk_indices=chunk_ids)
        for chunk_ids in ranges
    ]


def build_outline_sections(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    max_sections: int,
    branch_factor: int,
    max_depth: int,
    min_section_chunks: int,
) -> tuple[str, list[OutlineSection]]:
    if ONLINE:
        planned = plan_outline_online(
            chunks,
            max_sections=max_sections,
            min_section_chunks=min_section_chunks,
        )
        if planned is None:
            document_title, sections = plan_outline_offline(
                chunks,
                embeddings,
                max_sections=max_sections,
                branch_factor=branch_factor,
                min_section_chunks=min_section_chunks,
            )
        else:
            document_title, sections = planned
    else:
        document_title, sections = plan_outline_offline(
            chunks,
            embeddings,
            max_sections=max_sections,
            branch_factor=branch_factor,
            min_section_chunks=min_section_chunks,
        )

    if max_depth <= 1:
        return document_title, sections

    expanded: list[OutlineSection] = []
    for section in sections:
        if len(section.chunk_indices) > branch_factor:
            children = _subdivide_section(
                section,
                chunks,
                embeddings,
                max_sections=max_sections,
                branch_factor=branch_factor,
                min_section_chunks=min_section_chunks,
            )
            section = OutlineSection(
                title=section.title,
                chunk_indices=section.chunk_indices,
                children=children,
            )
        expanded.append(section)

    return document_title, expanded


def label_outline_sections(
    sections: list[OutlineSection],
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    title_fn,
) -> None:
    for section in sections:
        if not section.title:
            if ONLINE:
                from app.processing.topic_labeler import centroid_snippets

                snippets = centroid_snippets(section.chunk_indices, embeddings, chunks)
                section.title = llm_title_from_snippets(snippets) or title_fn(section.chunk_indices)
            else:
                section.title = title_fn(section.chunk_indices)

        if section.children:
            label_outline_sections(section.children, chunks, embeddings, title_fn)
