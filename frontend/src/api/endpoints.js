import { apiGet, buildUrl } from "./client";

export function getStatus() {
  return apiGet("/status");
}

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
