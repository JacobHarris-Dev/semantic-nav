import json
import re
from typing import List

from app.config import ONLINE
from app.processing import cloud_llm


def _build_split_prompt(simplified_query: str) -> str:
    return (
        "Extract distinct search topics from this query for semantic document retrieval. "
        "Return ONLY a JSON array of short topic strings. "
        "If there is one topic, return a one-element array. "
        "Do not include navigation words or filler.\n\n"
        'Example input: "food and navigation"\n'
        'Example output: ["food", "navigation"]\n\n'
        f"Query: {simplified_query}"
    )


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_subqueries(raw_output: str, fallback_query: str) -> List[str]:
    """Parse model output into a deduplicated list of subqueries."""
    cleaned = _strip_markdown_fences(raw_output)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return [fallback_query]

    if not isinstance(parsed, list) or not parsed:
        return [fallback_query]

    subqueries: List[str] = []
    seen = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        topic = item.strip()
        if not topic:
            continue
        key = topic.casefold()
        if key in seen:
            continue
        seen.add(key)
        subqueries.append(topic)

    if not subqueries:
        return [fallback_query]

    return subqueries


def split_query(simplified_query: str) -> List[str]:
    """Split a simplified query into distinct subqueries (online mode only)."""
    if not ONLINE:
        return [simplified_query]

    prompt = _build_split_prompt(simplified_query)

    try:
        raw_output = cloud_llm.complete(prompt, max_tokens=150)
    except RuntimeError:
        return [simplified_query]

    return parse_subqueries(raw_output, simplified_query)
