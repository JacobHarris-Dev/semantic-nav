import os
from pathlib import Path

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
