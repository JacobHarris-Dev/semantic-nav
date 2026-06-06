import numpy as np
from typing import Optional


class Retrieval:
    def __init__(self, faiss_embedder):
        self.store = faiss_embedder
        self.model = self.store.model

    def query(
        self,
        question: str,
        k: int = 3,
        confidence_threshold: float = 0.0,
        max_results: Optional[int] = None,
    ):
        """
        Convert query → embedding → FAISS search → results
        """

        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        if max_results is not None and max_results <= 0:
            raise ValueError("max_results must be greater than 0")

        # 1. Embed query
        query_vector = self.model.encode([question], convert_to_numpy=True, show_progress_bar=False)
        query_vector = np.array(query_vector).astype("float32")

        # 2. Search FAISS index
        search_k = max(k, max_results) if max_results is not None else k
        distances, indices = self.store.index.search(query_vector, search_k)

        # 3. Map results back to chunks and compute confidence scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.store.chunks):
                continue

            chunk = self.store.chunks[idx]
            score = float(1.0 / (1.0 + float(distance)))
            if score <= confidence_threshold:
                continue

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

        # 4. Highest confidence first and cap results
        results.sort(key=lambda item: item["score"], reverse=True)
        if max_results is not None:
            results = results[:max_results]

        return results