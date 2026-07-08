import { apiGet } from "./client";

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
