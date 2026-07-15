import { useEffect, useId, useRef, useState } from "react";

export default function DocumentMenu({
  triggerLabel,
  documents,
  loading,
  onSelect,
  align = "start",
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const listId = useId();

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const onPointerDown = (event) => {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const handleSelect = (doc) => {
    setOpen(false);
    onSelect(doc);
  };

  return (
    <div className={`document-menu document-menu--${align}`} ref={rootRef}>
      <button
        type="button"
        className="document-menu-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        disabled={loading}
        onClick={() => setOpen((value) => !value)}
      >
        {loading ? "Loading…" : triggerLabel}
      </button>

      {open && (
        <ul id={listId} className="document-menu-list" role="listbox">
          {documents.map((doc) => (
            <li key={doc.name} role="presentation">
              <button type="button" role="option" onClick={() => handleSelect(doc)}>
                {doc.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
