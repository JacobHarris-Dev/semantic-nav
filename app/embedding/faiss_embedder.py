import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSEmbedder:
    """Embed document chunks and store vectors in a FAISS index.

    FAISS holds only numeric vectors; original chunk text and character offsets
    are kept in self.chunks for retrieval and frontend highlighting.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="models", allow_download=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir),
                local_files_only=not allow_download,
            )
        except OSError as exc:
            raise RuntimeError(
                "Could not load the model locally. "
                "If you are in offline mode, make sure the model is downloaded to the cache folder. "
                "If you want downloads enabled, set SN_MODE=online or SN_ALLOW_DOWNLOAD=true."
            ) from exc

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store original chunks (FAISS only stores vectors)
        self.chunks = []
        self.embeddings = np.empty((0, self.dimension), dtype="float32")

    def add_chunks(self, chunks):
        """
        Convert chunks from array or chunk objects to embeddings and store in FAISS
        """

        texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]

        # 1. Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # 2. Ensure correct type
        embeddings = np.array(embeddings).astype("float32")

        # 3. Add to FAISS index
        self.index.add(embeddings)

        # 4. Store original chunks and embeddings separately
        self.chunks.extend(chunks)
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

  