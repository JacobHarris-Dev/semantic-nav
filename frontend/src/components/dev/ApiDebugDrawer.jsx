import { useEffect, useState } from "react";
import { clearApiDebugLog } from "../../api/debugLog";
import { useApiDebugLog } from "../../hooks/useApiDebugLog";

function formatPayload(data) {
  if (data === null || data === undefined) {
    return "";
  }
  if (typeof data === "string") {
    return data;
  }
  return JSON.stringify(data, null, 2);
}

function entryTitle(entry) {
  if (entry.label) {
    return entry.label;
  }

  try {
    const url = new URL(entry.url);
    return `${entry.method} ${url.pathname}`;
  } catch {
    return entry.method ?? "GET";
  }
}

export default function ApiDebugDrawer() {
  const entries = useApiDebugLog();
  const [open, setOpen] = useState(false);
  const [selectedId, setSelectedId] = useState(null);

  const selectedEntry = entries.find((entry) => entry.id === selectedId) ?? entries[0] ?? null;

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  useEffect(() => {
    if (entries.length > 0 && (selectedId === null || !entries.some((entry) => entry.id === selectedId))) {
      setSelectedId(entries[0].id);
    }
  }, [entries, selectedId]);

  return (
    <div className={`api-debug-drawer${open ? " api-debug-drawer--open" : ""}`}>
      <button
        type="button"
        className="api-debug-drawer-toggle"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls="api-debug-panel"
      >
        {open ? "Close debug" : "Debug"}
      </button>

      {open && (
        <aside id="api-debug-panel" className="api-debug-panel" aria-label="API debug panel">
          <div className="api-debug-panel-header">
            <h2>API Debug</h2>
            <div className="api-debug-panel-actions">
              <button type="button" onClick={() => clearApiDebugLog()}>
                Clear
              </button>
              <button type="button" onClick={() => setOpen(false)}>
                Close
              </button>
            </div>
          </div>

          {entries.length === 0 ? (
            <p className="muted api-debug-empty">No API calls yet.</p>
          ) : (
            <div className="api-debug-panel-body">
              <ul className="api-debug-history">
                {entries.map((entry) => (
                  <li key={entry.id}>
                    <button
                      type="button"
                      className={`api-debug-history-item${
                        selectedEntry?.id === entry.id ? " api-debug-history-item--active" : ""
                      }`}
                      onClick={() => setSelectedId(entry.id)}
                    >
                      <span className="api-debug-history-title">{entryTitle(entry)}</span>
                      <span className="api-debug-history-meta">
                        {entry.status ?? "—"}
                        {entry.elapsedMs != null ? ` · ${entry.elapsedMs}ms` : ""}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>

              {selectedEntry && (
                <div className="api-debug-detail">
                  <div className="meta-row">
                    <span className="badge">{selectedEntry.method ?? "GET"}</span>
                    {selectedEntry.status !== null && (
                      <span
                        className={`status status-${selectedEntry.status >= 400 ? "error" : "ok"}`}
                      >
                        {selectedEntry.status}
                      </span>
                    )}
                    {selectedEntry.elapsedMs !== undefined && (
                      <span className="muted">{selectedEntry.elapsedMs} ms</span>
                    )}
                  </div>

                  {selectedEntry.label && (
                    <p className="label-tag">{selectedEntry.label}</p>
                  )}

                  <h3 className="api-debug-section-title">Request</h3>
                  <pre className="code-block">{selectedEntry.url}</pre>

                  <h3 className="api-debug-section-title">Response</h3>
                  {selectedEntry.error ? (
                    <div className="error-banner">{selectedEntry.error}</div>
                  ) : (
                    <p className="muted success-text">Request completed successfully.</p>
                  )}
                  <pre className="code-block api-debug-response">
                    {formatPayload(selectedEntry.data)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
