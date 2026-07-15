import { useEffect, useRef, useState } from "react";
import { search } from "../../api/endpoints";
import { useApiCall } from "../../hooks/useApiCall";

const DEFAULT_RESULT_COUNT = 3;
const DEFAULT_CONFIDENCE = 30;
const RESULT_PREVIEW_LENGTH = 72;

function getSpeechRecognition() {
  if (typeof window === "undefined") {
    return null;
  }
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function truncateText(text, maxLength = RESULT_PREVIEW_LENGTH) {
  const trimmed = (text ?? "").replace(/\s+/g, " ").trim();
  if (trimmed.length <= maxLength) {
    return trimmed;
  }
  return `${trimmed.slice(0, maxLength)}…`;
}

function buildResultSummary(item, index) {
  const parts = [`${index + 1}`];

  if (item.location?.primary_page) {
    parts.push(`p.${item.location.primary_page}`);
  }

  if (item.score != null) {
    parts.push(`${Math.round(item.score * 100)}%`);
  }

  return parts.join(" · ");
}

export default function ReaderSearch({ document, chunkerType, onResultSelect }) {
  const { loading, run } = useApiCall();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [listening, setListening] = useState(false);
  const [statusError, setStatusError] = useState(null);
  const recognitionRef = useRef(null);

  const speechSupported = Boolean(getSpeechRecognition());

  const stopListening = () => {
    recognitionRef.current?.stop();
    setListening(false);
  };

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    setResults([]);
    setSelectedIndex(null);
    onResultSelect?.(null);
  }, [document?.name, onResultSelect]);

  const selectResult = (index, nextResults = results) => {
    const result = nextResults[index];
    if (!result) {
      return;
    }

    setSelectedIndex(index);
    onResultSelect?.(result);
  };

  const startListening = () => {
    const SpeechRecognitionCtor = getSpeechRecognition();
    if (!SpeechRecognitionCtor) {
      setStatusError("Speech recognition is not supported in this browser.");
      return;
    }

    setStatusError(null);

    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join("")
        .trim();

      if (transcript) {
        setQuery(transcript);
      }
    };

    recognition.onerror = (event) => {
      if (event.error === "not-allowed") {
        setStatusError("Microphone access denied.");
      } else if (event.error !== "aborted") {
        setStatusError("Speech recognition failed.");
      }
      setListening(false);
    };

    recognition.onend = () => {
      setListening(false);
    };

    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  };

  const handleMicClick = () => {
    if (listening) {
      stopListening();
      return;
    }
    startListening();
  };

  const handleSearch = async () => {
    const trimmed = query.trim();
    if (!trimmed || !document) {
      return;
    }

    setStatusError(null);

    const result = await run("GET /search", () =>
      search({
        q: trimmed,
        document: document.name,
        document_type: document.type,
        chunker_type: chunkerType,
        k: DEFAULT_RESULT_COUNT,
        confidence_threshold: DEFAULT_CONFIDENCE,
      }),
    );

    if (result.error) {
      setStatusError(result.error);
      setResults([]);
      setSelectedIndex(null);
      onResultSelect?.(null);
      return;
    }

    const nextResults = result.data?.results ?? [];
    if (nextResults.length === 0) {
      setStatusError("No results found.");
      setResults([]);
      setSelectedIndex(null);
      onResultSelect?.(null);
      return;
    }

    setResults(nextResults);
    selectResult(0, nextResults);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    handleSearch();
  };

  return (
    <form className="reader-search" onSubmit={handleSubmit}>
      <div className="reader-search-input-row">
        <textarea
          id="reader-search-query"
          rows={2}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search document…"
          disabled={!document || loading}
          aria-label="Search query"
        />
        <button
          type="button"
          className="reader-search-mic"
          onClick={handleMicClick}
          disabled={!document || loading || !speechSupported}
          aria-pressed={listening}
          aria-label={listening ? "Stop speech recognition" : "Speech to text"}
          title={
            speechSupported
              ? listening
                ? "Stop listening"
                : "Speech to text"
              : "Speech recognition unavailable"
          }
        >
          {listening ? "Stop" : "Mic"}
        </button>
      </div>

      <button type="submit" className="reader-search-submit" disabled={!document || loading || !query.trim()}>
        {loading ? "Searching…" : "Search"}
      </button>

      {statusError && <p className="reader-search-status reader-error">{statusError}</p>}

      {results.length > 0 && (
        <ul className="reader-search-results-list">
          {results.map((item, index) => (
            <li key={`${item.start}-${item.end}-${index}`}>
              <button
                type="button"
                className={`reader-search-result${selectedIndex === index ? " reader-search-result--active" : ""}`}
                onClick={() => selectResult(index)}
                aria-pressed={selectedIndex === index}
                title={truncateText(item.location?.text ?? item.text, 240)}
              >
                <span className="reader-search-result-summary">{buildResultSummary(item, index)}</span>
                <span className="reader-search-result-text">
                  {truncateText(item.location?.text ?? item.text)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </form>
  );
}
