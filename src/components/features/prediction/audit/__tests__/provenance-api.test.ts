/**
 * Tests for the bounded read path.
 *
 * The whole point of this module is that the browser never receives a 16 MB
 * audit trail. That is only true if the client caps the body, so the cap and
 * its failure modes are the main thing under test here.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  DEFAULT_MAX_RECORDS,
  DEFAULT_PROVENANCE_ENDPOINT,
  MAX_RESPONSE_BYTES,
  ProvenanceFetchError,
  buildProvenanceUrl,
  fetchProvenancePayload,
  parseProvenanceEnvelope,
} from "../provenance-api";

const STATION = "KT-8CEE47DC5194";

function goodEnvelope(stationId = STATION): unknown {
  return {
    success: true,
    message: "ok",
    data: {
      schemaVersion: 1,
      generatedAtUtc: "2026-09-30T10:00:00Z",
      stationId,
      forecast: null,
      forecastHistory: [],
      forecastCoverage: null,
      verification: null,
      policy: null,
      bundles: {},
      expectedAccuracy: null,
      originObservation: null,
      warnings: [],
    },
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("buildProvenanceUrl", () => {
  it("always sends a stationId", () => {
    const url = buildProvenanceUrl(DEFAULT_PROVENANCE_ENDPOINT, { stationId: STATION });
    assert.match(url, /^\/api\/prediction\/provenance\?/);
    assert.match(url, /stationId=KT-8CEE47DC5194/);
  });

  it("refuses to build a URL with no station", () => {
    assert.throws(() => buildProvenanceUrl("/api/x", { stationId: "  " }), /stationId/);
  });

  it("sends bounded read parameters by default so the server cannot read wholesale", () => {
    const url = new URL(buildProvenanceUrl(DEFAULT_PROVENANCE_ENDPOINT, { stationId: STATION }), "http://x");
    assert.equal(url.searchParams.get("maxRecords"), String(DEFAULT_MAX_RECORDS));
    assert.ok(Number(url.searchParams.get("maxRecords")) > 0);
  });

  it("omits the horizon when it is null so the server can choose", () => {
    const url = new URL(buildProvenanceUrl("/api/x", { stationId: STATION, horizonHours: null }), "http://x");
    assert.equal(url.searchParams.has("horizonHours"), false);
  });

  it("includes a finite horizon and an explicit recordId", () => {
    const url = new URL(
      buildProvenanceUrl("/api/x", { stationId: STATION, horizonHours: 12, recordId: " c212a4a4 " }),
      "http://x"
    );
    assert.equal(url.searchParams.get("horizonHours"), "12");
    assert.equal(url.searchParams.get("recordId"), "c212a4a4", "recordId must be trimmed");
  });

  it("keeps same-origin and ignores an absolute endpoint", () => {
    const url = buildProvenanceUrl("api/prediction/provenance", { stationId: STATION });
    assert.ok(url.startsWith("/"), "an absolute URL must not be smuggled into a browser fetch");
  });

  it("appends to an endpoint that already carries a query", () => {
    const url = buildProvenanceUrl("/api/x?tenant=a", { stationId: STATION });
    assert.match(url, /\/api\/x\?tenant=a&stationId=/);
  });

  it("only sends tailBytes when it is a positive finite number", () => {
    assert.ok(buildProvenanceUrl("/api/x", { stationId: STATION, tailBytes: 2048 }).includes("tailBytes=2048"));
    assert.ok(!buildProvenanceUrl("/api/x", { stationId: STATION, tailBytes: 0 }).includes("tailBytes"));
    assert.ok(!buildProvenanceUrl("/api/x", { stationId: STATION, tailBytes: null }).includes("tailBytes"));
  });
});

describe("parseProvenanceEnvelope", () => {
  it("returns the payload on a well-formed envelope", () => {
    const payload = parseProvenanceEnvelope(goodEnvelope());
    assert.equal(payload.stationId, STATION);
    assert.equal(payload.schemaVersion, 1);
  });

  it("rejects a non-object body", () => {
    assert.throws(() => parseProvenanceEnvelope("nope"), ProvenanceFetchError);
    assert.throws(() => parseProvenanceEnvelope(null), ProvenanceFetchError);
    assert.throws(() => parseProvenanceEnvelope([1, 2]), ProvenanceFetchError);
  });

  it("rejects success=false and surfaces the server message", () => {
    try {
      parseProvenanceEnvelope({ success: false, message: "trail is being rotated" });
      assert.fail("should have thrown");
    } catch (error) {
      assert.ok(error instanceof ProvenanceFetchError);
      assert.equal((error as ProvenanceFetchError).kind, "malformed");
      assert.match((error as ProvenanceFetchError).message, /rotated/);
    }
  });

  it("rejects a payload with no stationId rather than rendering an empty table", () => {
    assert.throws(
      () => parseProvenanceEnvelope({ success: true, data: { schemaVersion: 1 } }),
      /stationId/
    );
  });

  it("rejects a missing data object", () => {
    assert.throws(() => parseProvenanceEnvelope({ success: true }), /data/);
  });
});

describe("fetchProvenancePayload", () => {
  it("fetches a bounded payload and returns it", async () => {
    let seen = "";
    const payload = await fetchProvenancePayload(
      { stationId: STATION, horizonHours: 12 },
      {
        fetchImpl: async (input) => {
          seen = String(input);
          return jsonResponse(goodEnvelope());
        },
      }
    );
    assert.equal(payload.stationId, STATION);
    assert.match(seen, /horizonHours=12/);
  });

  it("surfaces an HTTP error with its status", async () => {
    await assert.rejects(
      () =>
        fetchProvenancePayload(
          { stationId: STATION },
          { fetchImpl: async () => new Response("boom", { status: 503, statusText: "Unavailable" }) }
        ),
      (error: unknown) => {
        assert.ok(error instanceof ProvenanceFetchError);
        assert.equal((error as ProvenanceFetchError).kind, "http");
        assert.equal((error as ProvenanceFetchError).status, 503);
        return true;
      }
    );
  });

  it("rejects an HTML error page as malformed rather than parsing it", async () => {
    await assert.rejects(
      () =>
        fetchProvenancePayload(
          { stationId: STATION },
          {
            fetchImpl: async () =>
              new Response("<!doctype html><html>404</html>", {
                status: 200,
                headers: { "Content-Type": "text/html" },
              }),
          }
        ),
      (error: unknown) => {
        assert.ok(error instanceof ProvenanceFetchError);
        assert.equal((error as ProvenanceFetchError).kind, "malformed");
        return true;
      }
    );
  });

  /**
   * The load-bearing test for the read path. An endpoint that ignores
   * `maxRecords` and streams the whole 16.5 MB trail must be caught, not
   * rendered.
   */
  it("aborts a response that blows past the browser cap", async () => {
    const envelope = goodEnvelope() as { success: boolean; message: string; data: Record<string, unknown> };
    envelope.data.blob = "x".repeat(MAX_RESPONSE_BYTES + 10_000);
    const oversized = JSON.stringify(envelope);
    await assert.rejects(
      () =>
        fetchProvenancePayload(
          { stationId: STATION },
          { fetchImpl: async () => new Response(oversized, { status: 200 }), maxResponseBytes: 10_000 }
        ),
      (error: unknown) => {
        assert.ok(error instanceof ProvenanceFetchError);
        assert.equal((error as ProvenanceFetchError).kind, "too-large");
        assert.match((error as ProvenanceFetchError).message, /bounded tail read/);
        return true;
      }
    );
  });

  it("catches an oversized non-streaming body too", async () => {
    const fake = {
      ok: true,
      status: 200,
      statusText: "OK",
      body: null,
      text: async () => "y".repeat(50_000),
    } as unknown as Response;
    await assert.rejects(
      () => fetchProvenancePayload({ stationId: STATION }, { fetchImpl: async () => fake, maxResponseBytes: 1000 }),
      (error: unknown) => (error as ProvenanceFetchError).kind === "too-large"
    );
  });

  it("reports an abort distinctly from a timeout", async () => {
    const controller = new AbortController();
    const promise = fetchProvenancePayload(
      { stationId: STATION },
      {
        signal: controller.signal,
        fetchImpl: (_input, init) =>
          new Promise((_resolve, reject) => {
            (init as RequestInit | undefined)?.signal?.addEventListener("abort", () => {
              const err = new Error("aborted");
              err.name = "AbortError";
              reject(err);
            });
          }),
      }
    );
    controller.abort();
    await assert.rejects(promise, (error: unknown) => (error as ProvenanceFetchError).kind === "aborted");
  });

  it("rejects immediately when the caller signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      () =>
        fetchProvenancePayload(
          { stationId: STATION },
          { signal: controller.signal, fetchImpl: async () => jsonResponse(goodEnvelope()) }
        ),
      (error: unknown) => (error as ProvenanceFetchError).kind === "aborted"
    );
  });

  it("times out a hung endpoint", async () => {
    await assert.rejects(
      () =>
        fetchProvenancePayload(
          { stationId: STATION },
          {
            timeoutMs: 20,
            fetchImpl: (_input, init) =>
              new Promise((_resolve, reject) => {
                (init as RequestInit | undefined)?.signal?.addEventListener("abort", () => {
                  const err = new Error("aborted");
                  err.name = "AbortError";
                  reject(err);
                });
              }),
          }
        ),
      (error: unknown) => {
        assert.ok(error instanceof ProvenanceFetchError);
        assert.equal((error as ProvenanceFetchError).kind, "timeout");
        assert.match((error as ProvenanceFetchError).message, /rotation/);
        return true;
      }
    );
  });

  it("classifies an unexpected throw as a network failure", async () => {
    await assert.rejects(
      () =>
        fetchProvenancePayload(
          { stationId: STATION },
          {
            fetchImpl: async () => {
              throw new TypeError("Failed to fetch");
            },
          }
        ),
      (error: unknown) => (error as ProvenanceFetchError).kind === "network"
    );
  });
});
