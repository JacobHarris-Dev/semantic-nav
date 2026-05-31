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
        end = min(start + chunk_size, len(text))
        chunks.append({
            "text": text[start:end],
            "start": start,
            "end": end,
        })
        start += chunk_size - overlap

    return chunks


class FixedChunker:
    """
    Wrapper class for fixed chunking to match SemanticChunker interface.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text: str):
        """
        Chunk document into fixed-size pieces.
        """
        return fixed_chunk_text(text, self.chunk_size, self.overlap)

