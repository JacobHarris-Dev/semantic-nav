import os
from typing import Any

from app.config import ONLINE


def rewrite_query(raw_query: str) -> str:
    """Rewrite a user query into a concise retrieval-friendly phrase.

    This is only invoked in online mode. In offline mode, the original query
    is returned unchanged.
    """
    if not ONLINE:
        return raw_query

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for query rewriting in online mode."
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The groq package is required for query rewriting in online mode. "
            "Install it with `pip install groq`."
        ) from exc

    client = Groq(api_key=api_key)

    prompt = (
        "Rewrite this search query into a concise phrase for semantic retrieval. "
        "Keep the meaning exact, preserve all topics, remove navigation words and filler, "
        "and return only the rewritten query.\n\n"
        f"User query: {raw_query}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    rewritten = response.choices[0].message.content.strip()

    if not rewritten or not rewritten.strip():
        raise RuntimeError("The query rewriter returned an empty response.")

    return rewritten.strip()


def _extract_text_from_response(output: Any) -> str:
    if not output:
        return ""

    if isinstance(output, str):
        return output

    if isinstance(output, list) and len(output) > 0:
        first = output[0]
        if isinstance(first, dict):
            parts = []
            for segment in first.get("content", []):
                if isinstance(segment, dict) and segment.get("type") == "output_text":
                    parts.append(segment.get("text", ""))
            return "".join(parts)

    return ""
