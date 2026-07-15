import { useEffect, useRef, useState } from "react";
import { buildPreviewPageUrl, getPreviewMeta } from "../../api/endpoints";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;
const ZOOM_STEP = 0.25;

export default function DocumentPreview({ document }) {
  const [pageCount, setPageCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [zoom, setZoom] = useState(1);
  const viewportRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const loadPreview = async () => {
      setLoading(true);
      setError(null);
      setPageCount(0);
      setZoom(1);

      const result = await getPreviewMeta({
        document: document.name,
        document_type: document.type,
      });

      if (cancelled) {
        return;
      }

      if (result.error) {
        setError(result.error);
        setLoading(false);
        return;
      }

      setPageCount(result.data?.page_count ?? 0);
      setLoading(false);
    };

    loadPreview();

    return () => {
      cancelled = true;
    };
  }, [document.name, document.type]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return undefined;
    }

    const onWheel = (event) => {
      if (!event.ctrlKey) {
        return;
      }

      event.preventDefault();
      setZoom((current) => {
        const delta = event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
        return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((current + delta).toFixed(2))));
      });
    };

    viewport.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      viewport.removeEventListener("wheel", onWheel);
    };
  }, [loading, pageCount]);

  const adjustZoom = (delta) => {
    setZoom((current) =>
      Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((current + delta).toFixed(2)))),
    );
  };

  const pages = Array.from({ length: pageCount }, (_, index) => index + 1);

  return (
    <div className="document-preview">
      <div className="document-preview-toolbar">
        <span className="document-preview-title">{document.name}</span>
        <div className="document-preview-zoom">
          <button type="button" onClick={() => adjustZoom(-ZOOM_STEP)} disabled={zoom <= MIN_ZOOM}>
            −
          </button>
          <span>{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={() => adjustZoom(ZOOM_STEP)} disabled={zoom >= MAX_ZOOM}>
            +
          </button>
          <button type="button" onClick={() => setZoom(1)} disabled={zoom === 1}>
            Reset
          </button>
        </div>
      </div>

      {loading && <p className="muted document-preview-status">Loading preview…</p>}
      {error && <p className="reader-error document-preview-status">{error}</p>}

      {!loading && !error && pageCount === 0 && (
        <p className="muted document-preview-status">No pages to display.</p>
      )}

      {!loading && !error && pageCount > 0 && (
        <div className="document-preview-viewport" ref={viewportRef}>
          <div
            className="document-preview-pages"
            style={{ transform: `scale(${zoom})`, transformOrigin: "top center" }}
          >
            {pages.map((page) => (
              <figure key={page} className="document-preview-page">
                <img
                  src={buildPreviewPageUrl({
                    document: document.name,
                    document_type: document.type,
                    page,
                  })}
                  alt={`${document.name} page ${page}`}
                  loading="lazy"
                />
                <figcaption className="muted">Page {page}</figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
