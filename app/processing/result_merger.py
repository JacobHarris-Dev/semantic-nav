from typing import Any, Dict, List, Optional, Tuple


def _chunk_key(result: Dict[str, Any]) -> Tuple[Any, ...]:
    start = result.get("start")
    end = result.get("end")
    if start is not None and end is not None:
        return ("span", start, end)
    return ("text", result.get("text", ""))


def merge_subquery_results(
    subquery_hits: List[Tuple[str, List[Dict[str, Any]]]],
    *,
    max_results: Optional[int],
) -> List[Dict[str, Any]]:
    """Merge per-subquery hits: reserve top-1 from each, then fill by score."""
    reserved: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    seen_keys = set()

    for subquery, hits in subquery_hits:
        for hit in hits:
            candidate = dict(hit)
            candidate["matched_subquery"] = subquery
            candidate_key = _chunk_key(candidate)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            if not any(item["matched_subquery"] == subquery for item in reserved):
                reserved.append(candidate)
            else:
                remaining.append(candidate)

    reserved.sort(key=lambda item: item["score"], reverse=True)
    remaining.sort(key=lambda item: item["score"], reverse=True)

    merged = list(reserved)
    for hit in remaining:
        if max_results is not None and len(merged) >= max_results:
            break
        merged.append(hit)

    if max_results is not None:
        merged = merged[:max_results]

    return merged
