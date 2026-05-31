import spacy
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunker:
    # initalize chunking object
    def __init__(
        self,

        # define model, threshold (how similaer do sentences need to be), and min chunk size
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
        Split document into sentences using spaCy and preserve character offsets.
        """

        doc = self.nlp(text)
        return [
            {
                "text": text[sent.start_char:sent.end_char],
                "start": sent.start_char,
                "end": sent.end_char,
            }
            for sent in doc.sents
        ]
    
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

    def create_chunks(self, sentences, similarities, full_text):
        """
        Build chunks based on semantic shifts.
        """
        chunks = []
        current_chunk = [sentences[0]]

        for i, sim in enumerate(similarities):

            # If semantic shift detected
            if sim < self.similarity_threshold:

                # Avoid tiny chunks with min sentence value
                if len(current_chunk) >= self.min_chunk_sentences:
                    start = current_chunk[0]["start"]
                    end = current_chunk[-1]["end"]
                    chunks.append({
                        "text": full_text[start:end],
                        "start": start,
                        "end": end,
                    })
                    current_chunk = []
            
            # append sentence to current chunk
            current_chunk.append(sentences[i + 1])

        # Add remaining chunk
        if current_chunk:
            start = current_chunk[0]["start"]
            end = current_chunk[-1]["end"]
            chunks.append({
                "text": full_text[start:end],
                "start": start,
                "end": end,
            })

        return chunks

    def chunk_document(self, text):
        """
        Full pipeline.
        """

        # Split text into individual sentences 
        sentences = self.split_sentences(text)

        # edge case: document only has 1 or 0 sentences 
        if len(sentences) <= 1:
            if len(sentences) == 1:
                return [{"text": sentences[0]["text"], "start": sentences[0]["start"], "end": sentences[0]["end"]}]
            return []

        # generate embeddings from sentences
        embeddings = self.embed_sentences([sentence["text"] for sentence in sentences])

        # compare similarities of embedded sentences 
        similarities = self.compute_similarities(embeddings)

        # create chunks based on similarity 
        chunks = self.create_chunks(
            sentences,
            similarities,
            text,
        )

        return chunks
