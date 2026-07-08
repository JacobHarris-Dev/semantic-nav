export default function PlaygroundStatusBar({ status, loading }) {
  if (!status) {
    return (
      <header className="status-bar">
        <div>
          <h1>API Playground</h1>
          <p className="muted">Loading server status…</p>
        </div>
        {loading && <span className="spinner">Working…</span>}
      </header>
    );
  }

  return (
    <header className="status-bar">
      <div>
        <h1>API Playground</h1>
        <p className="muted">
          <strong>{status.document}</strong> · {status.chunker} chunker ·{" "}
          <span className={`mode-pill mode-${status.mode}`}>{status.mode}</span>
        </p>
      </div>
      {loading && <span className="spinner">Working…</span>}
    </header>
  );
}
