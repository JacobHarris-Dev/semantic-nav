import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.dimension = 384  # MiniLM output size
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store original chunks (FAISS only stores vectors)
        self.chunks = []

    def add_chunks(self, chunks):
        """
        Convert chunks from array or chunk objects to embeddings and store in FAISS
        """

        texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]

        # 1. Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # 2. Ensure correct type
        embeddings = np.array(embeddings).astype("float32")

        # 3. Add to FAISS index
        self.index.add(embeddings)

        # 4. Store original chunks separately
        self.chunks.extend(chunks)

  