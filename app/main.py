from fastapi import FastAPI

from parsing.pdf_parser import extract_text_from_pdf
from chunking.semanticchunker import SemanticChunker
from embedding.faiss_embedder import FAISSStore
from embedding.retrieval import QueryEngine

app = FastAPI()


# Build pipeline on startup

text = extract_text_from_pdf("app/data/math_syllabus.pdf")

chunker = SemanticChunker(similarity_threshold=0.60)
chunks = chunker.chunk_document(text)

print("Chunks created:", len(chunks))



# Vector store
store = FAISSStore()
store.add_chunks(chunks)

# Query engine uses the store
query_engine = QueryEngine(store)


# Api endpoints

@app.get("/")
def root():
    return {"message": "Semantic navigation API running"}


@app.get("/search")
def search(q: str):
    results = query_engine.query(q, k=3)

    return {
        "query": q,
        "results": results
    }