from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def debug_clusters_path(
    resolved_path: str,
    chunker_type: str,
    branch_factor: int,
    max_depth: int,
    debug_dir: Path,
) -> Path:
    stem = Path(resolved_path).stem
    return (
        debug_dir
        / f"{stem}.{chunker_type}.bf{branch_factor}.d{max_depth}.clusters.txt"
    )


def _write_cluster_node(
    handle,
    node: Dict[str, Any],
    *,
    depth: int = 0,
) -> None:
    indent = "  " * depth
    title = node.get("title", "")
    chunk_count = node.get("chunk_count", 0)
    start = node.get("start")
    end = node.get("end")

    if start is not None and end is not None:
        header = f"{indent}--- Cluster (chars {start}-{end}, {chunk_count} chunks) ---\n"
    else:
        header = f"{indent}--- Cluster ({chunk_count} chunks) ---\n"

    handle.write(header)
    handle.write(f"{indent}Title: {title}\n\n")

    for child in node.get("children", []):
        _write_cluster_node(handle, child, depth=depth + 1)


def write_clusters_debug_file(
    resolved_path: str,
    chunker_type: str,
    *,
    branch_factor: int,
    max_depth: int,
    chunk_count: int,
    root: Optional[Dict[str, Any]],
    debug_dir: Path,
) -> Path:
    """Stream the topic-tree clusters to a plain-text debug file for developer inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_clusters_path(
        resolved_path,
        chunker_type,
        branch_factor,
        max_depth,
        debug_dir,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Document: {resolved_path}\n")
        handle.write(f"Chunker: {chunker_type}\n")
        handle.write(f"Branch factor: {branch_factor}\n")
        handle.write(f"Max depth: {max_depth}\n")
        handle.write(f"Total chunks: {chunk_count}\n")
        handle.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

        if root is None:
            handle.write("No clusters generated (document has no chunks).\n")
        else:
            _write_cluster_node(handle, root)

    return output_path
