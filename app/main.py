from fastapi import FastAPI

# Create FastAPI application. App represents backend server
app = FastAPI()

# Establish endpoint as / that returns JSON message
@app.get("/")
def root():
    return {"message": "Semantic navigation API running"}

# Import text extraction function to pull text from pdf
from parsing.pdf_parser import extract_text_from_pdf
text = extract_text_from_pdf("app/data/math_syllabus.pdf")

# Test fixed chunking 
from chunking.fixedchunker import fixed_chunk_text
chunks = fixed_chunk_text(text, chunk_size=250, overlap=0)
print("Number of chunks:", len(chunks))
print("First chunk: ", chunks[0])


# Test semantic chunking
from chunking.semanticchunker import SemanticChunker

# Create chunker instance
chunker = SemanticChunker(
    similarity_threshold=0.60
)

# Generate semantic chunks
chunks = chunker.chunk_document(text)

# Test output
print("Number of semantic chunks:", len(chunks))
print("\nFirst semantic chunk:\n")
print(chunks[0])