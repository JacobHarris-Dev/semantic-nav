/**
 * Low-level HTTP client for the semantic-nav backend.
 *
 * All API routes are GET requests. Base URL comes from VITE_API_BASE in frontend/.env.
 * Returns { url, status, data, error, elapsedMs } — check error before using data.
 */

import { recordApiDebugEntry } from "./debugLog";

/** Backend origin — defaults to local uvicorn server. */
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
      const result = {
        method: "GET",
        url,
        status: response.status,
        data,
        error: typeof detail === "string" ? detail : JSON.stringify(detail),
        elapsedMs,
      };
      recordApiDebugEntry(result);
      return result;
    }

    const result = {
      method: "GET",
      url,
      status: response.status,
      data,
      error: null,
      elapsedMs,
    };
    recordApiDebugEntry(result);
    return result;
  } catch (error) {
    const elapsedMs = Math.round(performance.now() - startedAt);
    const result = {
      method: "GET",
      url,
      status: null,
      data: null,
      error: error instanceof Error ? error.message : String(error),
      elapsedMs,
    };
    recordApiDebugEntry(result);
    return result;
  }
}

export { API_BASE, buildUrl };
