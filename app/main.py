from fastapi import FastAPI

# import fastapi framework and nessecary functions 
from app.parsing.pdf_parser import extract_text_from_pdf
from app.chunking.semanticchunker import SemanticChunker
from app.embedding.faiss_embedder import FAISSEmbedder
from app.embedding.retrieval import Retrieval

# create fastapi application object attatched to routes and startup events
app = FastAPI()

query_engine = None

# define behavior for startup event
@app.on_event("startup")
async def startup_event():
    global query_engine

    print("Initializing semantic pipeline...")

    # derive text from hard-coded document using extraction function
    text = extract_text_from_pdf("app/data/math_syllabus.pdf")

    # create chunking object, if neighboring sentences are less than 60% create new chunk
    chunker = SemanticChunker(similarity_threshold=0.60)

    # use chunking object to create chunks
    chunks = chunker.chunk_document(text)

    print("Chunks created:", len(chunks))

    # create vector database
    store = FAISSEmbedder()

    # store embedded chunks in database with all-MiniLM-L6-v2
    store.add_chunks(chunks)

    # embeds user query, searches FAISS, retrieves matches
    query_engine = Retrieval(store)

    print("Pipeline ready.")

# Get endpoint. Used to check if server is running
@app.get("/")
def root():
    return {"message": "Semantic navigation API running"}

# Semantic search endpoint. Pass user input as query into retrieval pipeline
@app.get("/search")
def search(q: str):
    results = query_engine.query(q, k=3)

    return {
        "query": q,
        "results": results
    }