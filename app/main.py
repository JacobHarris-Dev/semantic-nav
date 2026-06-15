from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException

from app.config import MODE, OFFLINE, ONLINE, MODEL_CACHE_DIR, MODEL_NAME, ALLOW_DOWNLOAD
from app.parsing.md_parser import read_markdown
from app.parsing.pdf_parser import extract_text_from_pdf
from app.parsing.txt_parser import read_txt
from app.chunking.semanticchunker import SemanticChunker
from app.chunking.fixedchunker import FixedChunker
from app.embedding.faiss_embedder import FAISSEmbedder
from app.embedding.retrieval import Retrieval
from app.processing.query_normalizer import normalize_query
from app.processing.query_rewriter import rewrite_query
from app.processing.result_reranker import rerank_results
from app.processing.metrics import (
    snapshot,
    estimate_tokens,
    print_stage_metrics,
    print_total_metrics,
)
import time

app = FastAPI()

DEFAULT_DOCUMENT = "pdf_example.pdf"
DEFAULT_DOCUMENT_TYPE = "pdf"
DEFAULT_CHUNKER = "semantic"
CURRENT_DOCUMENT = {
    "document": DEFAULT_DOCUMENT,
    "document_type": DEFAULT_DOCUMENT_TYPE,
    "chunker": DEFAULT_CHUNKER,
}
query_engine = None
PIPELINE_CACHE = {}


def pipeline_cache_key(resolved_path: str, chunker_type: str) -> Tuple[str, str]:
    return (resolved_path, chunker_type)


def pipeline_needs_switch(
    target_document: str,
    target_chunker: str,
    *,
    has_engine: bool,
    current_document: str,
    current_chunker: str,
) -> bool:
    if not has_engine:
        return True

    target_path = resolve_document_path(target_document)
    current_path = resolve_document_path(current_document)
    if target_path != current_path:
        return True

    if target_chunker != current_chunker:
        return True

    return False


def infer_document_type(document_path: str) -> str:
    suffix = Path(document_path).suffix.lower()

    document_types = {
        ".csv": "csv",
        ".docx": "docx",
        ".md": "md",
        ".pdf": "pdf",
        ".txt": "txt",
    }

    try:
        return document_types[suffix]
    except KeyError as exc:
        raise ValueError(f"Unsupported document type: {document_path}") from exc


def resolve_document_path(document_name: str) -> str:
    candidate = Path(document_name)

    if candidate.exists():
        return candidate.as_posix()

    fallback = Path("app/data") / document_name

    if fallback.exists():
        return fallback.as_posix()

    raise FileNotFoundError(f"Could not find document: {document_name}")


def extract_document_text(document_name: str, document_type: Optional[str] = None) -> str:
    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)

    if selected_type == "pdf":
        return extract_text_from_pdf(resolved_path)

    if selected_type == "txt":
        return read_txt(resolved_path)

    if selected_type == "md":
        return read_markdown(resolved_path)

    if selected_type == "csv":
        from app.parsing.csv_parser import CSVParser

        return CSVParser().parse(resolved_path)

    if selected_type == "docx":
        from app.parsing.docx_parser import DOCXParser

        return DOCXParser().parse(resolved_path)

    raise ValueError(f"Unsupported document type: {selected_type}")


def load_pipeline(document_name: str, document_type: Optional[str] = None, chunker_type: str = "semantic"):
    """Load document, parse, chunk, embed, and build retrieval index.

    Reuses cached pipeline state when the same resolved document and chunker
    were processed earlier in this server session.

    Prints timing information for parsing, chunking, embedding, and total time.
    """
    global query_engine

    resolved_path = resolve_document_path(document_name)
    selected_type = document_type or infer_document_type(resolved_path)
    cache_key = pipeline_cache_key(resolved_path, chunker_type)

    cached_engine = PIPELINE_CACHE.get(cache_key)
    if cached_engine is not None:
        query_engine = cached_engine
        CURRENT_DOCUMENT["document"] = document_name
        CURRENT_DOCUMENT["document_type"] = selected_type
        CURRENT_DOCUMENT["chunker"] = chunker_type
        print(f"[Pipeline cache hit] document={resolved_path} chunker={chunker_type}")
        return

    pipeline_start = time.perf_counter()

    # Parsing
    parse_start = time.perf_counter()
    text = extract_document_text(document_name, selected_type)
    parse_end = time.perf_counter()

    # Select chunker based on type
    if chunker_type == "fixed":
        chunker = FixedChunker(chunk_size=500, overlap=0)
    elif chunker_type == "semantic":
        chunker = SemanticChunker(similarity_threshold=0.60)
    else:
        raise ValueError(f"Unsupported chunker type: {chunker_type}. Use 'fixed' or 'semantic'.")

    # Chunking
    chunk_before = snapshot()
    chunk_start = time.perf_counter()
    chunks = chunker.chunk_document(text)
    chunk_end = time.perf_counter()
    chunk_after = snapshot()
    print_stage_metrics(
        "chunking",
        chunk_before,
        chunk_after,
        {"stage_tokens": estimate_tokens(text[:1000])},
    )

    # Embedding
    embed_before = snapshot()
    embed_start = time.perf_counter()
    store = FAISSEmbedder(
        model_name=MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        allow_download=not OFFLINE or ALLOW_DOWNLOAD,
    )
    store.add_chunks(chunks)
    embed_end = time.perf_counter()
    embed_after = snapshot()
    chunk_texts = [chunk.get("text", str(chunk)) if isinstance(chunk, dict) else chunk for chunk in chunks[:5]]
    print_stage_metrics(
        "embedding",
        embed_before,
        embed_after,
        {"stage_tokens": estimate_tokens(' '.join(chunk_texts))},
    )

    query_engine = Retrieval(store)
    PIPELINE_CACHE[cache_key] = query_engine
    CURRENT_DOCUMENT["document"] = document_name
    CURRENT_DOCUMENT["document_type"] = selected_type
    CURRENT_DOCUMENT["chunker"] = chunker_type

    pipeline_end = time.perf_counter()

    parse_time = parse_end - parse_start
    chunk_time = chunk_end - chunk_start
    embed_time = embed_end - embed_start
    total_time = pipeline_end - pipeline_start

    print(f"[Pipeline timings] document={document_name} chunker={chunker_type}")
    print(f"  parsing: {parse_time:.3f}s")
    print(f"  chunking: {chunk_time:.3f}s")
    print(f"  embedding: {embed_time:.3f}s")
    print(f"  total: {total_time:.3f}s")


@app.on_event("startup")
async def startup_event():
    print(f"Initializing semantic pipeline in {MODE} mode...")
    load_pipeline(DEFAULT_DOCUMENT, DEFAULT_DOCUMENT_TYPE)
    print("Pipeline ready.")


@app.get("/")
def root():
    return {"message": "Semantic navigation API running"}


@app.get("/load")
def load_document(document: str = DEFAULT_DOCUMENT, document_type: Optional[str] = None, chunker_type: str = "semantic"):
    try:
        load_pipeline(document, document_type, chunker_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "loaded",
        "document": CURRENT_DOCUMENT["document"],
        "document_type": CURRENT_DOCUMENT["document_type"],
        "chunker": CURRENT_DOCUMENT["chunker"],
        "mode": MODE,
    }


@app.get("/status")
def status():
    return {
        "document": CURRENT_DOCUMENT["document"],
        "document_type": CURRENT_DOCUMENT["document_type"],
        "chunker": CURRENT_DOCUMENT["chunker"],
        "mode": MODE,
    }


def normalize_confidence_threshold(value: float) -> float:
    """Normalize a user-provided confidence threshold.

    Accept both decimal values (0.0-1.0) and percent-style values (1-100).
    """
    if value > 1.0:
        value /= 100.0

    if value < 0.0 or value > 1.0:
        raise ValueError("confidence_threshold must be between 0 and 100, or 0.0 and 1.0")

    return value


@app.get("/search")
def search(
    q: str,
    document: Optional[str] = None,
    document_type: Optional[str] = None,
    chunker_type: Optional[str] = None,
    k: int = 3,
    confidence_threshold: float = 0.0,
    max_results: Optional[int] = None,
):
    try:
        target_document = document or CURRENT_DOCUMENT["document"]
        target_chunker = chunker_type or CURRENT_DOCUMENT["chunker"]
        target_document_type = document_type or CURRENT_DOCUMENT["document_type"]

        if pipeline_needs_switch(
            target_document,
            target_chunker,
            has_engine=query_engine is not None,
            current_document=CURRENT_DOCUMENT["document"],
            current_chunker=CURRENT_DOCUMENT["chunker"],
        ):
            load_pipeline(target_document, target_document_type, target_chunker)

        search_before = snapshot()

        if ONLINE:
            rewrite_before = snapshot()
            rewritten_query = rewrite_query(q)
            rewrite_after = snapshot()
            print_stage_metrics(
                "query rewrite",
                rewrite_before,
                rewrite_after,
                {
                    "stage_tokens": estimate_tokens(rewritten_query),
                },
            )
            # In online mode, let the model handle all query cleaning
            cleaned_q = rewritten_query
        else:
            rewritten_query = q
            # In offline mode, normalize query to remove boilerplate/intent phrases before embedding
            cleaned_q = normalize_query(rewritten_query)

        confidence_threshold = normalize_confidence_threshold(confidence_threshold)
        query_before = snapshot()
        results = query_engine.query(
            cleaned_q,
            k=k,
            confidence_threshold=confidence_threshold,
            max_results=max_results,
        )
        query_after = snapshot()
        print_stage_metrics(
            "FAISS retrieval",
            query_before,
            query_after,
            {
                "embedding_tokens": estimate_tokens(cleaned_q),
            },
        )

        # Re-rank results using Groq in online mode
        if ONLINE:
            rerank_before = snapshot()
            results = rerank_results(q, results)
            rerank_after = snapshot()
            print_stage_metrics(
                "result re-ranking",
                rerank_before,
                rerank_after,
            )

        search_after = snapshot()
        print_total_metrics(search_before, search_after)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "query": q,
        "rewritten_query": rewritten_query,
        "cleaned_query": cleaned_q,
        "confidence_threshold": confidence_threshold,
        "max_results": max_results,
        "results": results,
        "document": CURRENT_DOCUMENT["document"],
        "document_type": CURRENT_DOCUMENT["document_type"],
        "chunker": CURRENT_DOCUMENT["chunker"],
    }