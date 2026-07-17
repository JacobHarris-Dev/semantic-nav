/**
 * Typed wrappers for each backend route.
 *
 * Typical reader integration order:
 *   1. loadDocument()   — index the document on the server
 *   2. getPreviewText() — fetch text, page sections, and chunks for rendering
 *   3. search()         — semantic query; use result.location for highlighting
 */

import { apiGet, buildUrl } from "./client";

export function getStatus() {
  return apiGet("/status");
}

/** Index a document for search. Call before search or preview for that document. */
export function loadDocument({ document, document_type, chunker_type }) {
  return apiGet("/load", {
    document,
    document_type,
    chunker_type,
  });
}

export function getPreviewMeta({ document, document_type } = {}) {
  return apiGet("/preview/meta", {
    document,
    document_type,
  });
}

/** Full text + chunks for the reader. Chunks include start/end character offsets. */
export function getPreviewText({ document, document_type, chunker_type } = {}) {
  return apiGet("/preview/text", {
    document,
    document_type,
    chunker_type,
  });
}

export function buildPreviewPageUrl({ document, document_type, page, scale = 1 } = {}) {
  return buildUrl("/preview/page", {
    document,
    document_type,
    page,
    scale,
  });
}

export function getOutline({
  document,
  document_type,
  chunker_type,
  branch_factor,
  max_depth,
  max_sections,
} = {}) {
  return apiGet("/outline", {
    document,
    document_type,
    chunker_type,
    branch_factor,
    max_depth,
    max_sections,
  });
}

/** Semantic search. Results may include location { start, end, text, primary_page }. */
export function search({
  q,
  document,
  document_type,
  chunker_type,
  k,
  confidence_threshold,
  max_results,
  subquery_max_results,
  subquery_confidence_threshold,
} = {}) {
  return apiGet("/search", {
    q,
    document,
    document_type,
    chunker_type,
    k,
    confidence_threshold,
    max_results,
    subquery_max_results,
    subquery_confidence_threshold,
  });
}
