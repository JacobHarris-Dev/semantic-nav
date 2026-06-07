import os
from typing import List, Dict, Any

from app.config import ONLINE


def rerank_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-rank search results using Groq for improved relevance.
    
    This is only invoked in online mode. In offline mode, results are returned
    as-is from FAISS ranking.
    """
    if not ONLINE or not results:
        return results

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for result re-ranking in online mode."
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The groq package is required for result re-ranking in online mode. "
            "Install it with `pip install groq`."
        ) from exc

    client = Groq(api_key=api_key)

    # Format results for the prompt
    formatted_results = []
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        # Truncate long texts to avoid token bloat
        if len(text) > 500:
            text = text[:500] + "..."
        formatted_results.append(f"{i}. {text}")

    results_text = "\n\n".join(formatted_results)

    prompt = (
        f"Given this search query: \"{query}\"\n\n"
        "Re-rank these results by relevance to the query (most relevant first). "
        "Return ONLY the ranking as a numbered list (1, 2, 3, etc.), "
        "nothing else. Example output: 3, 1, 2\n\n"
        f"Results to rank:\n{results_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    ranking_text = response.choices[0].message.content.strip()

    # Parse the ranking
    try:
        ranking = [int(x.strip()) - 1 for x in ranking_text.split(",")]
        # Validate ranking
        if set(ranking) != set(range(len(results))):
            raise ValueError("Invalid ranking returned")
    except (ValueError, IndexError):
        # If parsing fails, return original results
        return results

    # Re-order results based on ranking
    reranked = [results[i] for i in ranking if i < len(results)]
    
    return reranked
