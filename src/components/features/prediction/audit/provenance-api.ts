/**
 * provenance-api.ts
 *
 * The browser-side read path for the provenance panel.
 *
 * ── THE ONE RULE ────────────────────────────────────────────────────────────
 * The audit trails are multi-megabyte, continuously appended, and gitignored
 * live runtime state. This module NEVER reads a JSONL file. It cannot: it runs
 * in the browser. It calls one server endpoint that does a bounded tail read
 * and returns a small, already-trimmed payload.
 *
 * A 16.5 MB prediction trail is the current reality. Fetching it into a tab is
 * not an option, and neither is an unbounded server read that happens to work
 * until the file doubles. The endpoint contract, including the tail semantics,
 * is specified in ./README-endpoint.md.
 *
 * DEFENCES HERE, so a wrong server still cannot melt the tab:
 *  - a hard cap on the response body, enforced while streaming, so a server
 *    that forgets to bound its read cannot stream 20 MB into a render;
 *  - envelope validation, so a malformed or wrong-shaped response becomes an
 *    explicit error state rather than an empty table that reads like "no
 *    history";
 *  - an abort on unmount and a manual refresh that replaces, never merges.
 */

import type { ProvenancePayload } from "./prediction-provenance-core";

export const DEFAULT_PROVENANCE_ENDPOINT = "/api/prediction/provenance";

/**
 * Hard ceiling on the response body.
 *
 * The endpoint is specified to return roughly 40-80 KB. 1 MB is generous enough
 * that a server-side growth will not trip it and small enough that a
 * misconfigured one is caught in under a second.
 */
export const MAX_RESPONSE_BYTES = 1_000_000;

export const DEFAULT_MAX_RECORDS = 200;
export const DEFAULT_MAX_VERIFICATION_RECORDS = 500;

export const DEFAULT_TIMEOUT_MS = 12_000;

export type ProvenanceFetchFailure =
  | "network"
  | "timeout"
  | "aborted"
  | "http"
  | "too-large"
  | "malformed";

export class ProvenanceFetchError extends Error {
  readonly kind: ProvenanceFetchFailure;
  readonly status?: number;
  readonly detail?: string;

  constructor(
    kind: ProvenanceFetchFailure,
    message: string,
    options?: { status?: number; detail?: string }
  ) {
    super(message);
    this.name = "ProvenanceFetchError";
    this.kind = kind;
    this.status = options?.status;
    this.detail = options?.detail;
  }
}

export interface ProvenanceQuery {
  stationId: string;
  horizonHours?: number | null;
  recordId?: string | null;
  maxRecords?: number;
  maxVerificationRecords?: number;
  /** Bytes of trail the server should seek into. Null leaves the server default. */
  tailBytes?: number | null;
}

export function buildProvenanceUrl(
  endpoint: string = DEFAULT_PROVENANCE_ENDPOINT,
  query: ProvenanceQuery
): string {
  if (!query.stationId || query.stationId.trim().length === 0) {
    throw new Error("buildProvenanceUrl requires a stationId");
  }
  const params = new URLSearchParams();
  params.set("stationId", query.stationId);

  if (typeof query.horizonHours === "number" && Number.isFinite(query.horizonHours)) {
    params.set("horizonHours", String(query.horizonHours));
  }
  if (query.recordId && query.recordId.trim().length > 0) {
    params.set("recordId", query.recordId.trim());
  }
  params.set("maxRecords", String(query.maxRecords ?? DEFAULT_MAX_RECORDS));
  params.set(
    "maxVerificationRecords",
    String(query.maxVerificationRecords ?? DEFAULT_MAX_VERIFICATION_RECORDS)
  );
  if (typeof query.tailBytes === "number" && Number.isFinite(query.tailBytes) && query.tailBytes > 0) {
    params.set("tailBytes", String(Math.floor(query.tailBytes)));
  }

  // Relative endpoint, same-origin fetch: keep the base path so a proxy prefix
  // survives, but never let the caller smuggle in a full URL.
  const base = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}${params.toString()}`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Validate the envelope.
 *
 * Shallow on purpose: the point is to catch "wrong endpoint", "HTML error page"
 * and "truncated response", not to re-typecheck the server. Deep validation
 * happens in the core module, which is tolerant by design and reports gaps as
 * gaps rather than throwing.
 */
export function parseProvenanceEnvelope(body: unknown): ProvenancePayload {
  if (!isPlainObject(body)) {
    throw new ProvenanceFetchError("malformed", "Response body was not a JSON object.");
  }
  if (body.success !== true) {
    const message =
      typeof body.message === "string" && body.message.length > 0
        ? body.message
        : "Endpoint reported success=false.";
    throw new ProvenanceFetchError("malformed", message);
  }
  const data = body.data;
  if (!isPlainObject(data)) {
    throw new ProvenanceFetchError("malformed", "Envelope had no `data` object.");
  }
  const stationId = typeof data.stationId === "string" ? data.stationId : "";
  if (stationId.length === 0) {
    throw new ProvenanceFetchError(
      "malformed",
      "Payload had no stationId; this does not look like a provenance response."
    );
  }
  return data as unknown as ProvenancePayload;
}

export interface FetchProvenanceOptions {
  endpoint?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
  maxResponseBytes?: number;
  fetchImpl?: typeof fetch;
}

/**
 * Read the bounded payload.
 *
 * Streams the body and counts bytes as they arrive rather than trusting
 * `Content-Length`, which a proxy may compress or omit entirely.
 */
export async function fetchProvenancePayload(
  query: ProvenanceQuery,
  options: FetchProvenanceOptions = {}
): Promise<ProvenancePayload> {
  const {
    endpoint = DEFAULT_PROVENANCE_ENDPOINT,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    maxResponseBytes = MAX_RESPONSE_BYTES,
    fetchImpl,
  } = options;

  const doFetch = fetchImpl ?? (typeof fetch === "function" ? fetch.bind(globalThis) : null);
  if (!doFetch) {
    throw new ProvenanceFetchError("network", "No fetch implementation available.");
  }

  const url = buildProvenanceUrl(endpoint, query);

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const externalSignal = options.signal;
  const onExternalAbort = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) {
      clearTimeout(timer);
      throw new ProvenanceFetchError("aborted", "Request aborted before it started.");
    }
    externalSignal.addEventListener("abort", onExternalAbort, { once: true });
  }

  try {
    const response = await doFetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
      cache: "no-store",
    });

    if (!response.ok) {
      throw new ProvenanceFetchError(
        "http",
        `Provenance endpoint returned ${response.status} ${response.statusText}.`,
        { status: response.status }
      );
    }

    const text = await readBounded(response, maxResponseBytes);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      throw new ProvenanceFetchError(
        "malformed",
        `Provenance response was not valid JSON: ${(error as Error).message}`
      );
    }
    return parseProvenanceEnvelope(parsed);
  } catch (error) {
    if (error instanceof ProvenanceFetchError) throw error;
    if (timedOut) {
      throw new ProvenanceFetchError(
        "timeout",
        `Provenance request exceeded ${timeoutMs} ms. The audit trail may be mid-rotation.`
      );
    }
    if (isAbortError(error)) {
      throw new ProvenanceFetchError("aborted", "Provenance request aborted.");
    }
    throw new ProvenanceFetchError(
      "network",
      `Provenance request failed: ${(error as Error)?.message ?? String(error)}`
    );
  } finally {
    clearTimeout(timer);
    if (externalSignal) externalSignal.removeEventListener("abort", onExternalAbort);
  }
}

function isAbortError(error: unknown): boolean {
  if (!isPlainObject(error)) return false;
  const name = (error as { name?: unknown }).name;
  return name === "AbortError" || name === "TimeoutError";
}

/**
 * Read a response body, giving up as soon as it exceeds `maxBytes`.
 *
 * The reader is cancelled before throwing so the connection is released rather
 * than left streaming into a tab nobody is listening to.
 */
async function readBounded(response: Response, maxBytes: number): Promise<string> {
  const body = (response as { body?: ReadableStream<Uint8Array> | null }).body;

  if (!body || typeof body.getReader !== "function") {
    // No streaming body (some test doubles, older runtimes). Fall back to text()
    // but still enforce the cap afterwards.
    const text = await response.text();
    if (text.length > maxBytes) {
      throw new ProvenanceFetchError(
        "too-large",
        `Provenance response exceeded the ${maxBytes} byte browser cap. ` +
          `The endpoint must return a bounded tail read, not the whole trail.`
      );
    }
    return text;
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let received = 0;
  let text = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        received += value.byteLength;
        if (received > maxBytes) {
          throw new ProvenanceFetchError(
            "too-large",
            `Provenance response exceeded the ${maxBytes} byte browser cap ` +
              `(${received} bytes and still streaming). ` +
              `The endpoint must return a bounded tail read, not the whole trail.`
          );
        }
        text += decoder.decode(value, { stream: true });
      }
    }
    text += decoder.decode();
  } catch (error) {
    if (error instanceof ProvenanceFetchError) {
      try {
        await reader.cancel();
      } catch {
        /* the stream is already gone; nothing to release */
      }
      throw error;
    }
    throw error;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* already released by cancel() */
    }
  }

  return text;
}
