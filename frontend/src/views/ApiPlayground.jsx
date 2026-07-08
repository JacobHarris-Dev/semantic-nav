import { useEffect, useState } from "react";
import { getOutline, getStatus, loadDocument, search } from "../api/endpoints";
import { CHUNKERS, DOCUMENTS } from "../constants/documents";
import { useApiCall } from "../hooks/useApiCall";
import PlaygroundStatusBar from "../components/dev/PlaygroundStatusBar";
import RequestPanel from "../components/dev/RequestPanel";
import ResponsePanel from "../components/dev/ResponsePanel";

export default function ApiPlayground() {
  const { loading, lastRequest, lastResponse, run } = useApiCall();
  const [status, setStatus] = useState(null);

  const [document, setDocument] = useState(DOCUMENTS[0].name);
  const [chunkerType, setChunkerType] = useState("semantic");
  const [maxSections, setMaxSections] = useState(8);
  const [maxDepth, setMaxDepth] = useState(2);
  const [branchFactor, setBranchFactor] = useState(6);

  const [query, setQuery] = useState("grading policy");
  const [resultCount, setResultCount] = useState(5);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);

  const refreshStatus = async () => {
    const result = await run("GET /status", getStatus);
    if (!result.error && result.data) {
      setStatus(result.data);
    }
    return result;
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  const selectedDoc = DOCUMENTS.find((item) => item.name === document) ?? DOCUMENTS[0];

  const handleLoad = async () => {
    const result = await run("GET /load", () =>
      loadDocument({
        document,
        document_type: selectedDoc.type,
        chunker_type: chunkerType,
      }),
    );
    if (!result.error && result.data) {
      setStatus(result.data);
    }
  };

  const handleOutline = () =>
    run("GET /outline", () =>
      getOutline({
        branch_factor: branchFactor,
        max_depth: maxDepth,
        max_sections: maxSections,
      }),
    );

  const handleSearch = () =>
    run("GET /search", () =>
      search({
        q: query,
        k: resultCount,
        confidence_threshold: confidenceThreshold,
      }),
    );

  return (
    <div className="view-shell">
      <PlaygroundStatusBar status={status} loading={loading} />

      <div className="controls-grid">
        <section className="panel control-panel">
          <h2>1. Status</h2>
          <p className="muted">Read the active document, chunker, and server mode.</p>
          <button type="button" onClick={refreshStatus} disabled={loading}>
            Refresh status
          </button>
        </section>

        <section className="panel control-panel">
          <h2>2. Load document</h2>
          <label>
            Document
            <select value={document} onChange={(event) => setDocument(event.target.value)}>
              {DOCUMENTS.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Chunker
            <select
              value={chunkerType}
              onChange={(event) => setChunkerType(event.target.value)}
            >
              {CHUNKERS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={handleLoad} disabled={loading}>
            Load document
          </button>
        </section>

        <section className="panel control-panel">
          <h2>3. Outline</h2>
          <div className="inline-fields">
            <label>
              Max sections
              <input
                type="number"
                min={2}
                max={20}
                value={maxSections}
                onChange={(event) => setMaxSections(Number(event.target.value))}
              />
            </label>
            <label>
              Max depth
              <input
                type="number"
                min={1}
                max={10}
                value={maxDepth}
                onChange={(event) => setMaxDepth(Number(event.target.value))}
              />
            </label>
            <label>
              Branch factor
              <input
                type="number"
                min={1}
                max={50}
                value={branchFactor}
                onChange={(event) => setBranchFactor(Number(event.target.value))}
              />
            </label>
          </div>
          <button type="button" onClick={handleOutline} disabled={loading}>
            Fetch outline
          </button>
        </section>

        <section className="panel control-panel">
          <h2>4. Search</h2>
          <label>
            Query
            <input
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. grading policy"
            />
          </label>
          <div className="inline-fields">
            <label>
              Results (k)
              <input
                type="number"
                min={1}
                max={20}
                value={resultCount}
                onChange={(event) => setResultCount(Number(event.target.value))}
              />
            </label>
            <label>
              Confidence %
              <input
                type="number"
                min={0}
                max={100}
                value={confidenceThreshold}
                onChange={(event) => setConfidenceThreshold(Number(event.target.value))}
              />
            </label>
          </div>
          <button type="button" onClick={handleSearch} disabled={loading || !query.trim()}>
            Run search
          </button>
        </section>
      </div>

      <div className="io-grid">
        <RequestPanel request={lastRequest} response={lastResponse} />
        <ResponsePanel response={lastResponse} />
      </div>
    </div>
  );
}
