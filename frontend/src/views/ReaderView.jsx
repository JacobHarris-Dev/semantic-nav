/**
 * Phase 2: simulated blind-reader user flow.
 * Build your layout here — outline traversal, text pane, TTS, keyboard nav, etc.
 */
export default function ReaderView() {
  return (
    <div className="view-shell reader-view">
      <header className="reader-header">
        <h1>Document Reader</h1>
        <p className="muted">
          Simulated user flow — add outline navigation, text reading, and search here.
        </p>
      </header>

      <div className="reader-layout">
        <aside className="panel reader-sidebar">
          <h2>Outline</h2>
          <p className="muted">Section list will go here.</p>
        </aside>

        <main className="panel reader-main">
          <h2>Content</h2>
          <p className="muted">Current section text and highlights will go here.</p>
        </main>

        <aside className="panel reader-controls">
          <h2>Controls</h2>
          <p className="muted">Search, prev/next, and read-aloud controls will go here.</p>
        </aside>
      </div>
    </div>
  );
}
