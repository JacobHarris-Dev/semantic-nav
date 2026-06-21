import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

MODE = os.environ.get("SN_MODE", "offline").strip().lower()
MODEL_NAME = os.environ.get("SN_MODEL_NAME", "all-MiniLM-L6-v2").strip()
MODEL_CACHE_DIR = os.environ.get("SN_MODEL_CACHE_DIR", str(Path.cwd() / "models")).strip()

# If true, allow model downloads even in offline mode.
# By default offline rejects downloads and online allows them.
ALLOW_DOWNLOAD = os.environ.get("SN_ALLOW_DOWNLOAD", "false").strip().lower() in ("1", "true", "yes")

OFFLINE = MODE == "offline"
ONLINE = MODE == "online"

# When true, write a developer-only chunk dump after the first chunk of each document/chunker pair.
DEBUG_CHUNKS = os.environ.get("SN_DEBUG_CHUNKS", "false").strip().lower() in ("1", "true", "yes")
DEBUG_CHUNKS_DIR = Path(os.environ.get("SN_DEBUG_CHUNKS_DIR", str(Path.cwd() / "cache" / "debug")))

VALID_QUERY_SPLITTERS = ("groq", "gemini", "cerebras")
QUERY_SPLITTER = os.environ.get("SN_QUERY_SPLITTER", "groq").strip().lower()
if QUERY_SPLITTER not in VALID_QUERY_SPLITTERS:
    raise ValueError(
        f"SN_QUERY_SPLITTER must be one of {VALID_QUERY_SPLITTERS}, got {QUERY_SPLITTER!r}"
    )
QUERY_SPLITTER_MODEL = os.environ.get("SN_QUERY_SPLITTER_MODEL", "").strip()

_subquery_max_results_raw = os.environ.get("SN_SUBQUERY_MAX_RESULTS", "").strip()
SUBQUERY_MAX_RESULTS: int | None = (
    int(_subquery_max_results_raw) if _subquery_max_results_raw else None
)

_subquery_confidence_threshold_raw = os.environ.get("SN_SUBQUERY_CONFIDENCE_THRESHOLD", "").strip()
SUBQUERY_CONFIDENCE_THRESHOLD: float | None = (
    float(_subquery_confidence_threshold_raw) if _subquery_confidence_threshold_raw else None
)
