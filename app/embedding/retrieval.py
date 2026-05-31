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

        # 3. Map results back to chunks and compute confidence scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.store.chunks):
                continue

            chunk = self.store.chunks[idx]
            score = float(1.0 / (1.0 + float(distance)))

            if isinstance(chunk, dict):
                result = {
                    **chunk,
                    "score": score,
                }
            else:
                result = {
                    "text": chunk,
                    "score": score,
                }

            results.append(result)

        # 4. Highest confidence first
        results.sort(key=lambda item: item["score"], reverse=True)

        return results