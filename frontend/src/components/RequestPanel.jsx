export default function RequestPanel({ request, response }) {
  if (!request && !response) {
    return (
      <section className="panel">
        <h2>Request</h2>
        <p className="muted">Run an API action to see the request URL.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Request</h2>
      {request?.label && <p className="label-tag">{request.label}</p>}
      {response && (
        <>
          <div className="meta-row">
            <span className="badge">{response.method ?? "GET"}</span>
            {response.status !== null && (
              <span className={`status status-${response.status >= 400 ? "error" : "ok"}`}>
                {response.status}
              </span>
            )}
            {response.elapsedMs !== undefined && (
              <span className="muted">{response.elapsedMs} ms</span>
            )}
          </div>
          <pre className="code-block">{response.url}</pre>
        </>
      )}
    </section>
  );
}
