import os
from typing import Any

from app.config import ONLINE
from app.processing.metrics import estimate_tokens


def rewrite_query(raw_query: str) -> str:
    """Extract short topic keywords from a user query for the query splitter.

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
        "Extract distinct search topics from the user query for semantic document retrieval. "
        "Return ONLY a comma-separated list of short topic keywords (1-3 words each). "
        "Remove question words, navigation phrases, and filler. "
        "Preserve every distinct topic; do not merge unrelated ideas. "
        "Do not write full sentences or questions.\n\n"
        "Examples:\n"
        'User: "Where can I find grading policy and calculator rules for tests?"\n'
        "Output: grading policy, calculators, tests\n\n"
        'User: "food and navigation"\n'
        "Output: food, navigation\n\n"
        f'User query: "{raw_query}"\n'
        "Output:"
    )

    prompt_tokens = estimate_tokens(prompt)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0,
    )

    rewritten = response.choices[0].message.content.strip()
    response_tokens = estimate_tokens(rewritten)

    print(f"[Groq metrics] query rewrite prompt_tokens={prompt_tokens} response_tokens={response_tokens}")

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
