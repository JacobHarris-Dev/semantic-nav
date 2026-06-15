from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Union

Chunk = Union[dict[str, Any], str]


def debug_chunks_path(resolved_path: str, chunker_type: str, debug_dir: Path) -> Path:
    stem = Path(resolved_path).stem
    return debug_dir / f"{stem}.{chunker_type}.chunks.txt"


def write_chunks_debug_file(
    resolved_path: str,
    chunker_type: str,
    chunks: List[Chunk],
    *,
    debug_dir: Path,
) -> Path:
    """Stream all chunks to a plain-text debug file for developer inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_chunks_path(resolved_path, chunker_type, debug_dir)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Document: {resolved_path}\n")
        handle.write(f"Chunker: {chunker_type}\n")
        handle.write(f"Total chunks: {len(chunks)}\n")
        handle.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

        for index, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                start = chunk.get("start")
                end = chunk.get("end")
                if start is not None and end is not None:
                    header = f"--- Chunk {index} (chars {start}-{end}) ---\n"
                else:
                    header = f"--- Chunk {index} ---\n"
            else:
                text = str(chunk)
                header = f"--- Chunk {index} ---\n"

            handle.write(header)
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.write("\n")

    return output_path
