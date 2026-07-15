export default function ReaderChunkNav({
  chunkIndex,
  chunkCount,
  onPrevious,
  onNext,
  disabled,
}) {
  const atStart = chunkIndex <= 0;
  const atEnd = chunkCount === 0 || chunkIndex >= chunkCount - 1;

  return (
    <div className="reader-chunk-nav">
      <button type="button" onClick={onPrevious} disabled={disabled || atStart}>
        Prev
      </button>
      <span className="reader-chunk-nav-label">
        {chunkCount > 0 ? `${chunkIndex + 1} / ${chunkCount}` : "—"}
      </span>
      <button type="button" onClick={onNext} disabled={disabled || atEnd}>
        Next
      </button>
    </div>
  );
}
