import spacy
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunker:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.65,
        min_chunk_sentences=2
    ):
        # Load spaCy sentence tokenizer
        self.nlp = spacy.load("en_core_web_sm")

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences

    def split_sentences(self, text):
        """
        Split document into sentences using spaCy.
        """

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def embed_sentences(self, sentences):
        """
        Generate embeddings for each sentence.
        """

        # numpy library handles vector transformations
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True
        )

        return embeddings
    
    def compute_similarities(self, embeddings):
        """
        Compute cosine similarity between adjacent sentences
        """
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]

            similarities.append(sim)

        return similarities

    def create_chunks(self, sentences, similarities):
        """
        Build chunks based on semantic shifts.
        """
        chunks = []
        current_chunk = [sentences[0]]

        for i, sim in enumerate(similarities):

            # If semantic shift detected
            if sim < self.similarity_threshold:

                # Avoid tiny chunks
                if len(current_chunk) >= self.min_chunk_sentences:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []

            current_chunk.append(sentences[i + 1])

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_document(self, text):
        """
        Full pipeline.
        """
        sentences = self.split_sentences(text)

        if len(sentences) <= 1:
            return sentences

        embeddings = self.embed_sentences(sentences)

        similarities = self.compute_similarities(embeddings)

        chunks = self.create_chunks(
            sentences,
            similarities
        )

        return chunks
