function formatPayload(data) {
  if (data === null || data === undefined) {
    return "";
  }
  if (typeof data === "string") {
    return data;
  }
  return JSON.stringify(data, null, 2);
}

export default function ResponsePanel({ response }) {
  if (!response) {
    return (
      <section className="panel">
        <h2>Response</h2>
        <p className="muted">Response JSON will appear here.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Response</h2>
      {response.error ? (
        <div className="error-banner">{response.error}</div>
      ) : (
        <p className="muted success-text">Request completed successfully.</p>
      )}
      <pre className="code-block response-block">{formatPayload(response.data)}</pre>
    </section>
  );
}
