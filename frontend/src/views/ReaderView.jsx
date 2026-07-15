import { useCallback, useEffect, useRef, useState } from "react";
import { loadDocument } from "../api/endpoints";
import DocumentMenu from "../components/reader/DocumentMenu";
import DocumentPreview from "../components/reader/DocumentPreview";
import ReaderChunkNav from "../components/reader/ReaderChunkNav";
import ReaderSearch from "../components/reader/ReaderSearch";
import ApiDebugDrawer from "../components/dev/ApiDebugDrawer";
import { CHUNKERS, DOCUMENTS } from "../constants/documents";
import { useApiCall } from "../hooks/useApiCall";
import { findChunkIndex } from "../utils/readerPosition";

function resolveSearchHighlight(result) {
  if (!result) {
    return null;
  }

  if (result.location) {
    return result.location;
  }

  if (result.start == null || result.end == null) {
    return null;
  }

  return {
    start: result.start,
    end: result.end,
    text: result.text ?? "",
    primary_page: null,
    pages: [],
  };
}

export default function ReaderView() {
  const { loading, run } = useApiCall();
  const previewRef = useRef(null);
  const [loadedDocument, setLoadedDocument] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [pageSections, setPageSections] = useState([]);
  const [activeChunkIndex, setActiveChunkIndex] = useState(0);
  const [chunkerType] = useState(CHUNKERS[0]);

  const handleDocumentReady = useCallback(({ chunks: nextChunks, pageSections: nextPageSections }) => {
    setChunks(nextChunks);
    setPageSections(nextPageSections);
    setActiveChunkIndex(0);
  }, []);

  const handleSelectDocument = async (doc) => {
    setLoadError(null);
    setChunks([]);
    setPageSections([]);
    setActiveChunkIndex(0);

    const result = await run("GET /load", () =>
      loadDocument({
        document: doc.name,
        document_type: doc.type,
        chunker_type: chunkerType,
      }),
    );

    if (result.error) {
      setLoadError(result.error);
      return;
    }

    setLoadedDocument(doc);
  };

  const goToChunk = useCallback(
    (index) => {
      if (index < 0 || index >= chunks.length) {
        return;
      }

      previewRef.current?.seekToChunk(index);
    },
    [chunks.length],
  );

  const handlePreviousChunk = useCallback(() => {
    goToChunk(activeChunkIndex - 1);
  }, [activeChunkIndex, goToChunk]);

  const handleNextChunk = useCallback(() => {
    goToChunk(activeChunkIndex + 1);
  }, [activeChunkIndex, goToChunk]);

  const handleResultSelect = useCallback(
    (result) => {
      const highlight = resolveSearchHighlight(result);
      if (!highlight || !chunks.length) {
        return;
      }

      previewRef.current?.stopSpeech();
      setActiveChunkIndex(findChunkIndex(chunks, highlight.start));
    },
    [chunks],
  );

  useEffect(() => {
    setChunks([]);
    setPageSections([]);
    setActiveChunkIndex(0);
  }, [loadedDocument?.name]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (!loadedDocument || !chunks.length) {
        return;
      }

      const target = event.target;
      const tagName = target.tagName;
      if (
        tagName === "INPUT" ||
        tagName === "TEXTAREA" ||
        tagName === "SELECT" ||
        target.isContentEditable
      ) {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        handlePreviousChunk();
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        handleNextChunk();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [loadedDocument, chunks.length, handlePreviousChunk, handleNextChunk]);

  return (
    <div className="view-shell reader-view">
      <header className="reader-header">
        <h1>Document Reader</h1>
        {loadedDocument && <p className="muted">Reading {loadedDocument.name}</p>}
      </header>

      <div className="reader-layout">
        <main className="panel reader-main">
          {!loadedDocument ? (
            <div className="reader-empty-state">
              <DocumentMenu
                triggerLabel="Load document"
                documents={DOCUMENTS}
                loading={loading}
                onSelect={handleSelectDocument}
                align="start"
              />
              {loadError && <p className="reader-error">{loadError}</p>}
            </div>
          ) : (
            <DocumentPreview
              ref={previewRef}
              key={loadedDocument.name}
              document={loadedDocument}
              chunkerType={chunkerType}
              chunks={chunks}
              pageSections={pageSections}
              activeChunkIndex={activeChunkIndex}
              onActiveChunkIndexChange={setActiveChunkIndex}
              onDocumentReady={handleDocumentReady}
            />
          )}
        </main>

        <aside className="panel reader-controls">
          {loadedDocument && (
            <div className="reader-controls-header">
              <DocumentMenu
                triggerLabel="Change document"
                documents={DOCUMENTS}
                loading={loading}
                onSelect={handleSelectDocument}
                align="end"
              />
            </div>
          )}

          <ReaderChunkNav
            chunkIndex={activeChunkIndex}
            chunkCount={chunks.length}
            onPrevious={handlePreviousChunk}
            onNext={handleNextChunk}
            disabled={!loadedDocument || !chunks.length}
          />

          <ReaderSearch
            document={loadedDocument}
            chunkerType={chunkerType}
            onResultSelect={handleResultSelect}
          />
        </aside>
      </div>

      <ApiDebugDrawer />
    </div>
  );
}
