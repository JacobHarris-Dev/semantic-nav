import re
from typing import List


def normalize_query(text: str) -> str:
    """Normalize and strip common boilerplate/intention from user queries.

    - Lowercases
    - Removes common intent phrases (e.g., "take me to", "where they talk about")
    - Removes punctuation and extra whitespace
    - Returns a compacted query ready for embedding
    """
    if not text:
        return ""

    t = text.lower()

    # common boilerplate/intent phrases to remove
    # NOTE: longer/more-specific patterns must come BEFORE shorter ones
    # to avoid breaking multi-word matches (e.g., "where they talk about" before "where")
    patterns: List[str] = [
        # Longest, most-specific phrases first
        r"where they (?:talk about|discuss|mention)",
        r"where they mention",
        r"where can i find",
        r"where in the document",
        r"of the document",
        r"in the document",
        r"in this document",
        r"in this section",
        r"in this file",
        r"show me the part",
        r"show me the section",
        r"show me where",
        r"the part where",
        r"the section where",
        r"it talks about",
        r"is mentioned in",
        r"is discussed in",
        r"are mentioned in",
        r"are discussed in",
        r"is talked about",
        r"where it says",
        r"go back to",
        r"go forward to",
        r"navigate to",
        r"take me to",
        r"bring me to",
        r"point me to",
        r"tell me where",
        
        # Medium-length phrases
        r"search for",
        r"look for",
        r"could you",
        r"would you",
        r"can i find",
        r"can i",
        r"is mentioned",
        r"is discussed",
        r"are mentioned",
        r"are discussed",
        r"show me",
        r"the section",
        r"the part",
        
        # Single-word and shorter
        r"take me",
        r"go to",
        r"jump to",
        r"skip to",
        r"scroll to",
        r"which part",
        r"which section",
        r"please",
        r"find",
        r"locate",
        r"about",
        r"where is",
        r"where",
    ]

    for p in patterns:
        t = re.sub(p, " ", t)

    # remove punctuation but keep alphanumerics and spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t
