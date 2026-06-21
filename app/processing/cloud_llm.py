import os
from typing import Callable

from app.config import QUERY_SPLITTER, QUERY_SPLITTER_MODEL
from app.processing.metrics import estimate_tokens

DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.0-flash-lite",
    "cerebras": "llama3.1-8b",
}

API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}


def _resolve_model() -> str:
    if QUERY_SPLITTER_MODEL:
        return QUERY_SPLITTER_MODEL
    return DEFAULT_MODELS[QUERY_SPLITTER]


def _complete_groq(prompt: str, *, max_tokens: int) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for the groq query splitter.")

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The groq package is required for the groq query splitter. "
            "Install it with `pip install groq`."
        ) from exc

    client = Groq(api_key=api_key)
    model = _resolve_model()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _complete_gemini(prompt: str, *, max_tokens: int) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the gemini query splitter.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "The google-genai package is required for the gemini query splitter. "
            "Install it with `pip install google-genai`."
        ) from exc

    client = genai.Client(api_key=api_key)
    model = _resolve_model()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"max_output_tokens": max_tokens},
    )
    return response.text.strip()


def _complete_cerebras(prompt: str, *, max_tokens: int) -> str:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required for the cerebras query splitter.")

    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as exc:
        raise RuntimeError(
            "The cerebras-cloud-sdk package is required for the cerebras query splitter. "
            "Install it with `pip install cerebras-cloud-sdk`."
        ) from exc

    client = Cerebras(api_key=api_key)
    model = _resolve_model()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


_PROVIDERS: dict[str, Callable[..., str]] = {
    "groq": _complete_groq,
    "gemini": _complete_gemini,
    "cerebras": _complete_cerebras,
}


def complete(prompt: str, *, max_tokens: int = 150) -> str:
    """Run a completion against the configured query-splitter cloud provider."""
    provider = _PROVIDERS.get(QUERY_SPLITTER)
    if provider is None:
        raise RuntimeError(f"Unsupported query splitter provider: {QUERY_SPLITTER}")

    prompt_tokens = estimate_tokens(prompt)
    result = provider(prompt, max_tokens=max_tokens)
    response_tokens = estimate_tokens(result)

    print(
        f"[{QUERY_SPLITTER} metrics] query split prompt_tokens={prompt_tokens} "
        f"response_tokens={response_tokens}"
    )

    if not result or not result.strip():
        raise RuntimeError("The query splitter returned an empty response.")

    return result.strip()
