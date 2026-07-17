import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { getPreviewText } from "../../api/endpoints";
import { chunkToHighlight } from "../../utils/readerPosition";

/**
 * Renders document text with chunk highlighting and optional text-to-speech.
 *
 * On mount: fetches GET /preview/text and passes chunks/pageSections to the parent.
 * Highlights use character offsets (start/end) shared with search result.location.
 * TTS reads one chunk at a time via the browser speechSynthesis API.
 */

function getSpeechSynthesis() {
  if (typeof window === "undefined") {
    return null;
  }
  return window.speechSynthesis ?? null;
}

function pushSegment(parts, text, from, to, highlight, highlightRef, highlightRefAssigned) {
  if (from >= to) {
    return;
  }

  const highlightStart = highlight?.start;
  const highlightEnd = highlight?.end;

  if (
    highlightStart == null ||
    highlightEnd == null ||
    to <= highlightStart ||
    from >= highlightEnd
  ) {
    parts.push(<span key={`${from}-${to}`}>{text.slice(from, to)}</span>);
    return;
  }

  const markStart = Math.max(from, highlightStart);
  const markEnd = Math.min(to, highlightEnd);
  const assignRef = !highlightRefAssigned.current;
  if (assignRef) {
    highlightRefAssigned.current = true;
  }

  if (from < markStart) {
    parts.push(<span key={`${from}-${markStart}`}>{text.slice(from, markStart)}</span>);
  }

  parts.push(
    <mark
      key={`${markStart}-${markEnd}`}
      ref={assignRef ? highlightRef : undefined}
      className="document-preview-highlight"
    >
      {text.slice(markStart, markEnd)}
    </mark>,
  );

  if (markEnd < to) {
    parts.push(<span key={`${markEnd}-${to}`}>{text.slice(markEnd, to)}</span>);
  }
}

function buildDocumentParts(text, pageSections, highlight, highlightRef) {
  const parts = [];
  const highlightRefAssigned = { current: false };

  if (!pageSections?.length) {
    pushSegment(parts, text, 0, text.length, highlight, highlightRef, highlightRefAssigned);
    return parts;
  }

  let cursor = 0;

  for (const section of pageSections) {
    if (cursor < section.start) {
      pushSegment(parts, text, cursor, section.start, highlight, highlightRef, highlightRefAssigned);
    }

    parts.push(
      <div key={`page-label-${section.page}`} className="document-preview-page-label">
        Page {section.page}
      </div>,
    );

    pushSegment(
      parts,
      text,
      section.start,
      section.end,
      highlight,
      highlightRef,
      highlightRefAssigned,
    );
    cursor = section.end;
  }

  if (cursor < text.length) {
    pushSegment(parts, text, cursor, text.length, highlight, highlightRef, highlightRefAssigned);
  }

  return parts;
}

const DocumentPreview = forwardRef(function DocumentPreview(
  {
    document,
    chunkerType,
    chunks,
    pageSections,
    activeChunkIndex,
    onActiveChunkIndexChange,
    onDocumentReady,
  },
  ref,
) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [speechState, setSpeechState] = useState("idle");
  const viewportRef = useRef(null);
  const highlightRef = useRef(null);
  const speechStateRef = useRef("idle");
  const chunksRef = useRef([]);
  const pageSectionsRef = useRef([]);
  const activeChunkIndexRef = useRef(0);
  const speechSessionRef = useRef(0);

  const speechSupported = Boolean(getSpeechSynthesis());

  const stopSpeech = useCallback(() => {
    speechSessionRef.current += 1;
    getSpeechSynthesis()?.cancel();
    speechStateRef.current = "idle";
    setSpeechState("idle");
  }, []);

  const speakChunkAt = useCallback(
    (index) => {
      const synthesis = getSpeechSynthesis();
      const activeChunks = chunksRef.current;
      const session = speechSessionRef.current;

      if (!synthesis || !activeChunks[index] || speechStateRef.current !== "playing") {
        speechStateRef.current = "idle";
        setSpeechState("idle");
        return;
      }

      onActiveChunkIndexChange?.(index);

      const chunk = activeChunks[index];
      const utterance = new SpeechSynthesisUtterance(chunk.text);
      utterance.onend = () => {
        if (session !== speechSessionRef.current) {
          return;
        }

        if (speechStateRef.current !== "playing") {
          return;
        }

        const nextIndex = index + 1;
        if (nextIndex >= activeChunks.length) {
          speechStateRef.current = "idle";
          setSpeechState("idle");
          return;
        }

        speakChunkAt(nextIndex);
      };
      utterance.onerror = (event) => {
        if (session !== speechSessionRef.current) {
          return;
        }

        if (event.error === "interrupted" || event.error === "canceled") {
          return;
        }

        speechStateRef.current = "idle";
        setSpeechState("idle");
      };

      synthesis.speak(utterance);
    },
    [onActiveChunkIndexChange],
  );

  const seekToChunk = useCallback(
    (index) => {
      const synthesis = getSpeechSynthesis();
      const activeChunks = chunksRef.current;

      if (!activeChunks[index]) {
        return;
      }

      const shouldContinue =
        speechStateRef.current === "playing" || speechStateRef.current === "paused";

      speechSessionRef.current += 1;
      synthesis?.cancel();
      onActiveChunkIndexChange?.(index);

      if (shouldContinue && synthesis) {
        speechStateRef.current = "playing";
        setSpeechState("playing");
        speakChunkAt(index);
      }
    },
    [onActiveChunkIndexChange, speakChunkAt],
  );

  useImperativeHandle(ref, () => ({ stopSpeech, seekToChunk }), [stopSpeech, seekToChunk]);

  useEffect(() => {
    chunksRef.current = chunks;
  }, [chunks]);

  useEffect(() => {
    pageSectionsRef.current = pageSections;
  }, [pageSections]);

  useEffect(() => {
    activeChunkIndexRef.current = activeChunkIndex;
  }, [activeChunkIndex]);

  useEffect(() => {
    let cancelled = false;

    const loadPreview = async () => {
      setLoading(true);
      setError(null);
      setText("");

      const result = await getPreviewText({
        document: document.name,
        document_type: document.type,
        chunker_type: chunkerType,
      });

      if (cancelled) {
        return;
      }

      if (result.error) {
        setError(result.error);
        setLoading(false);
        return;
      }

      const nextText = result.data?.text ?? "";
      const nextPageSections = result.data?.page_sections ?? [];
      const nextChunks = result.data?.chunks ?? [];

      setText(nextText);
      onDocumentReady?.({
        chunks: nextChunks,
        pageSections: nextPageSections,
      });
      setLoading(false);
    };

    loadPreview();

    return () => {
      cancelled = true;
    };
  }, [document.name, document.type, chunkerType, onDocumentReady]);

  useEffect(() => {
    stopSpeech();
  }, [document.name, document.type, chunkerType, stopSpeech]);

  useEffect(() => {
    return () => {
      speechSessionRef.current += 1;
      getSpeechSynthesis()?.cancel();
    };
  }, []);

  const readingHighlight = useMemo(
    () => chunkToHighlight(chunks[activeChunkIndex], pageSections),
    [chunks, pageSections, activeChunkIndex],
  );

  useEffect(() => {
    if (!readingHighlight || !highlightRef.current) {
      return;
    }

    highlightRef.current.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [readingHighlight, text]);

  const handlePlay = () => {
    const synthesis = getSpeechSynthesis();
    if (!synthesis || !chunks.length) {
      return;
    }

    if (speechState === "paused") {
      synthesis.resume();
      speechStateRef.current = "playing";
      setSpeechState("playing");
      return;
    }

    speechSessionRef.current += 1;
    synthesis.cancel();
    speechStateRef.current = "playing";
    setSpeechState("playing");
    speakChunkAt(activeChunkIndexRef.current);
  };

  const handlePause = () => {
    const synthesis = getSpeechSynthesis();
    if (!synthesis || speechStateRef.current !== "playing") {
      return;
    }

    synthesis.pause();
    speechStateRef.current = "paused";
    setSpeechState("paused");
  };

  const renderedText = useMemo(
    () => buildDocumentParts(text, pageSections, readingHighlight, highlightRef),
    [text, pageSections, readingHighlight],
  );

  const positionLabel = readingHighlight?.primary_page
    ? `Section · page ${readingHighlight.primary_page}`
    : readingHighlight
      ? "Current section"
      : null;

  const canSpeak = speechSupported && chunks.length > 0;

  return (
    <div className="document-preview">
      <div className="document-preview-toolbar">
        <span className="document-preview-title">{document.name}</span>
        <div className="document-preview-toolbar-actions">
          {positionLabel && (
            <span className="document-preview-match-label">{positionLabel}</span>
          )}
          {canSpeak && (
            <div className="document-preview-speech">
              <button
                type="button"
                onClick={handlePlay}
                disabled={speechState === "playing"}
                aria-label={speechState === "paused" ? "Resume reading" : "Play reading"}
              >
                Play
              </button>
              <button
                type="button"
                onClick={handlePause}
                disabled={speechState !== "playing"}
                aria-label="Pause reading"
              >
                Pause
              </button>
            </div>
          )}
        </div>
      </div>

      {loading && <p className="muted document-preview-status">Loading document…</p>}
      {error && <p className="reader-error document-preview-status">{error}</p>}

      {!loading && !error && !text && (
        <p className="muted document-preview-status">No text to display.</p>
      )}

      {!loading && !error && text && (
        <div className="document-preview-viewport" ref={viewportRef}>
          <pre className="document-preview-text">{renderedText}</pre>
        </div>
      )}
    </div>
  );
});

export default DocumentPreview;
