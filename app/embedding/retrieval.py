from sentence_transformers import SentenceTransformer
import numpy as np


class Retrieval:
    def __init__(self, faiss_embedder):
        self.store = faiss_embedder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def query(self, question: str, k: int = 3):
        """
        Convert query → embedding → FAISS search → results
        """

        # 1. Embed query
        query_vector = self.model.encode([question], convert_to_numpy=True)
        query_vector = np.array(query_vector).astype("float32")

        # 2. Search FAISS index
        distances, indices = self.store.index.search(query_vector, k)

        # 3. Map results back to chunks
        results = []
        for idx in indices[0]:
            if idx < len(self.store.chunks):
                results.append(self.store.chunks[idx])

        return results