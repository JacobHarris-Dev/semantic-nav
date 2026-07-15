export function findChunkIndex(chunks, position) {
  if (!chunks.length) {
    return 0;
  }

  const directIndex = chunks.findIndex(
    (chunk) => position >= chunk.start && position < chunk.end,
  );
  if (directIndex >= 0) {
    return directIndex;
  }

  const nextIndex = chunks.findIndex((chunk) => chunk.start >= position);
  return nextIndex >= 0 ? nextIndex : chunks.length - 1;
}

export function chunkToHighlight(chunk, pageSections = []) {
  if (!chunk) {
    return null;
  }

  const page = pageSections.find(
    (section) => chunk.start >= section.start && chunk.start < section.end,
  )?.page;

  return {
    start: chunk.start,
    end: chunk.end,
    text: chunk.text,
    primary_page: page ?? null,
    pages: page ? [page] : [],
  };
}
