from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.parsing.md_parser import read_markdown
from app.parsing.pdf_parser import extract_text_from_pdf
from app.parsing.txt_parser import read_txt
from app.chunking.semanticchunker import SemanticChunker
from app.embedding.faiss_embedder import FAISSEmbedder
from app.embedding.retrieval import Retrieval

app = FastAPI()

DEFAULT_DOCUMENT = "pdf_example.pdf"
DEFAULT_DOCUMENT_TYPE = "pdf"
CURRENT_DOCUMENT = {
    "document": DEFAULT_DOCUMENT,
    "document_type": DEFAULT_DOCUMENT_TYPE,
}
query_engine = None


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


def load_pipeline(document_name: str, document_type: Optional[str] = None):
    global query_engine

    selected_type = document_type or infer_document_type(resolve_document_path(document_name))
    text = extract_document_text(document_name, selected_type)

    chunker = SemanticChunker(similarity_threshold=0.60)
    chunks = chunker.chunk_document(text)

    store = FAISSEmbedder()
    store.add_chunks(chunks)

    query_engine = Retrieval(store)
    CURRENT_DOCUMENT["document"] = document_name
    CURRENT_DOCUMENT["document_type"] = selected_type


@app.on_event("startup")
async def startup_event():
    print("Initializing semantic pipeline...")
    load_pipeline(DEFAULT_DOCUMENT, DEFAULT_DOCUMENT_TYPE)
    print("Pipeline ready.")


@app.get("/")
def root():
    return {"message": "Semantic navigation API running"}


@app.get("/load")
def load_document(document: str = DEFAULT_DOCUMENT, document_type: Optional[str] = None):
    try:
        load_pipeline(document, document_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "loaded",
        "document": CURRENT_DOCUMENT["document"],
        "document_type": CURRENT_DOCUMENT["document_type"],
    }


@app.get("/status")
def status():
    return {
        "document": CURRENT_DOCUMENT["document"],
        "document_type": CURRENT_DOCUMENT["document_type"],
    }


@app.get("/search")
def search(q: str, document: Optional[str] = None, document_type: Optional[str] = None, k: int = 3):
    try:
        if document is not None or document_type is not None:
            load_pipeline(document or CURRENT_DOCUMENT["document"], document_type)
        elif query_engine is None:
            load_pipeline(DEFAULT_DOCUMENT, DEFAULT_DOCUMENT_TYPE)

        results = query_engine.query(q, k=k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "query": q,
        "results": results,
        "document": CURRENT_DOCUMENT["document"],
        "document_type": CURRENT_DOCUMENT["document_type"],
    }