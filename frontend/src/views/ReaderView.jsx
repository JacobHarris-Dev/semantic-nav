import { useState } from "react";
import { loadDocument } from "../api/endpoints";
import DocumentMenu from "../components/reader/DocumentMenu";
import DocumentPreview from "../components/reader/DocumentPreview";
import { CHUNKERS, DOCUMENTS } from "../constants/documents";
import { useApiCall } from "../hooks/useApiCall";

/**
 * Phase 2: simulated blind-reader user flow.
 * Build your layout here — outline traversal, text pane, TTS, keyboard nav, etc.
 */
export default function ReaderView() {
  const { loading, run } = useApiCall();
  const [loadedDocument, setLoadedDocument] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [chunkerType] = useState(CHUNKERS[0]);

  const handleSelectDocument = async (doc) => {
    setLoadError(null);

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

  return (
    <div className="view-shell reader-view">
      <header className="reader-header">
        <h1>Document Reader</h1>
        <p className="muted">
          {loadedDocument
            ? `Reading ${loadedDocument.name}`
            : "Simulated user flow — add outline navigation, text reading, and search here."}
        </p>
      </header>

      <div className="reader-layout">
        <aside className="panel reader-sidebar">
          <h2>Outline</h2>
          <p className="muted">Section list will go here.</p>
        </aside>

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
            <DocumentPreview key={loadedDocument.name} document={loadedDocument} />
          )}
        </main>

        <div className="reader-right-column">
          {loadedDocument && (
            <div className="reader-doc-bar">
              <DocumentMenu
                triggerLabel="Change document"
                documents={DOCUMENTS}
                loading={loading}
                onSelect={handleSelectDocument}
                align="end"
              />
            </div>
          )}
          <aside className="panel reader-controls">
            <h2>Controls</h2>
            <p className="muted">Search, prev/next, and read-aloud controls will go here.</p>
          </aside>
        </div>
      </div>
    </div>
  );
}
