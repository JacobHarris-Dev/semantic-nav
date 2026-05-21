# Semantic Navigation API (MVP)

### A lightweight semantic document search system built with FastAPI, sentence embeddings, and FAISS.

### It allows users to query a document and retrieve the most semantically relevant sections.

## What It Does

### Turns a document into searchable meaning-based chunks:
#### Document → Text Extraction → Chunking → Embeddings → FAISS → Search API

### Users can query the document like:

#### "What is calculus?"
### and receive the most relevant parts of the text.

## Core Features
### PDF/text parsing using PyMuPDF (fitz)
### Fixed + semantic chunking (spaCy-based sentence splitting)
### Embeddings using all-MiniLM-L6-v2 (sentence-transformers via Hugging Face)
### Vector search with FAISS
### FastAPI endpoint for querying chunks

## API
### Search
### GET /search?q=your+query

### Example:

#### curl "http://127.0.0.1:8000/search?q=what%20is%20calculus"

### Response:

#### {
####  "query": "what is calculus",
#### "results": [...]
#### }
### Run Locally
#### uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

### Then open:

### API: http://127.0.0.1:8000
### Docs: http://127.0.0.1:8000/docs
## Tech Stack
### FastAPI
### spaCy
### sentence-transformers (MiniLM)
### FAISS
### PyMuPDF
## MVP Goal

### Enable basic semantic navigation of a document using lightweight embeddings and vector search.