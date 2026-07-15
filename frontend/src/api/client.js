const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

function buildUrl(path, params = {}) {
  const url = new URL(path, API_BASE);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

export async function apiGet(path, params = {}) {
  const url = buildUrl(path, params);
  const startedAt = performance.now();

  try {
    const response = await fetch(url);
    const elapsedMs = Math.round(performance.now() - startedAt);
    let data = null;

    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      const detail =
        typeof data === "object" && data !== null && "detail" in data
          ? data.detail
          : `Request failed with status ${response.status}`;
      return {
        method: "GET",
        url,
        status: response.status,
        data,
        error: typeof detail === "string" ? detail : JSON.stringify(detail),
        elapsedMs,
      };
    }

    return {
      method: "GET",
      url,
      status: response.status,
      data,
      error: null,
      elapsedMs,
    };
  } catch (error) {
    const elapsedMs = Math.round(performance.now() - startedAt);
    return {
      method: "GET",
      url,
      status: null,
      data: null,
      error: error instanceof Error ? error.message : String(error),
      elapsedMs,
    };
  }
}

export { API_BASE, buildUrl };
