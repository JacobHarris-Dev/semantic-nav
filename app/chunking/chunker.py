import spacy
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def fixed_chunk_text(text: str, chunk_size: int, overlap: int = 0):
    """
    Splits text into fixed-size character chunks.

    Args:
        text: input string
        chunk_size: number of characters per chunk
        overlap: number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def semantic_chunk_text(text: str):
    print("hello")

def split_sentences(text:str):
    
    