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
    patterns: List[str] = [
        # direct navigation intents
        r"take me to",
        r"take me",
        r"go to",
        r"go back to",
        r"go forward to",
        r"navigate to",
        r"jump to",
        r"skip to",
        r"scroll to",
        r"bring me to",
        r"point me to",

        # user phrasing for locating content
        r"show me the part",
        r"show me where",
        r"show me the section",
        r"which part",
        r"which section",
        r"the part where",
        r"the section where",
        r"where they mention",
        r"where they talk about",
        r"where it says",
        r"where can i find",
        r"where in the document",
        r"in the document",
        r"in this document",
        r"in this section",
        r"in this file",

        # conversational padding / politeness
        r"please",
        r"could you",
        r"would you",

        # generic locate/search verbs (still useful to strip for navigation intent)
        r"find",
        r"locate",
        r"search for",
        r"look for",
        r"tell me where",
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
