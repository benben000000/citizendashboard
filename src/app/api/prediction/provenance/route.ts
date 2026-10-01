import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

/**
 * GET /api/prediction/provenance
 *
 * Serves the audit panel: the prediction trail, the verification trail, the
 * serving policy, the bundle manifests, and the live measured skill.
 *
 * WHY THIS ROUTE EXISTS
 *
 * The panel has pointed at this endpoint since before it existed. Every fetch
 * 404'd, so the verified-accuracy table could never render -- which means the
 * measured accuracy the system produces was invisible on the dashboard, and
 * asking "what accuracy do you actually achieve?" had no answer anywhere in the
 * product. The numbers existed; nothing served them.
 *
 * READING THE TRAILS
 *
 * Both JSONL trails are read TAIL-FIRST under a hard byte cap. They grow without
 * bound -- the prediction trail passed 13 MB and the verification trail 17 MB
 * while this was being built -- and reading either fully to answer a dashboard
 * question would block the event loop for seconds. Reading the tail also matches
 * what the panel needs: recent history, not the whole file. The cap is enforced
 * on BYTES READ, not on lines, so a single enormous line cannot slip past it.
 *
 * A truncated read starts mid-line. That first fragment is dropped rather than
 * parsed, and the condition is reported as a warning so the panel can say the
 * history is partial instead of silently presenting a window as if it were
 * everything.
 *
 * WHAT IS MEASURED vs WHAT IS MODELLED
 *
 * `liveSkill` is measured on forecasts the running system published, scored
 * against later station observations, with persistence recomputed on the same
 * rows. `expectedAccuracy` is the held-out backtest. They are returned as
 * separate fields and never merged: conflating a backtest with a live
 * measurement is the single easiest way to make a real number dishonest.
 */

const MAX_TAIL_BYTES = 2_000_000;
const DEFAULT_STATION = process.env.PROVENANCE_DEFAULT_STATION || "KT-8CEE47DC5194";

type JsonRecord = Record<string, unknown>;

function dataDir(): string {
  return path.join(process.cwd(), "prediction-model", "data");
}

function readTailLines(file: string, maxBytes: number): { lines: string[]; partialHead: boolean; bytes: number } {
  let handle: number | null = null;
  try {
    handle = fs.openSync(file, "r");
    const size = fs.fstatSync(handle).size;
    if (size === 0) return { lines: [], partialHead: false, bytes: 0 };
    const start = Math.max(0, size - maxBytes);
    const length = size - start;
    const buffer = Buffer.alloc(length);
    fs.readSync(handle, buffer, 0, length, start);
    const text = buffer.toString("utf-8");
    const all = text.split("\n");
    // A tail read almost never begins on a record boundary. Drop the first
    // fragment and say so, rather than letting a truncated JSON object fail as
    // "malformed" and implying the file is corrupt.
    const partialHead = start > 0;
    if (partialHead && all.length > 0) all.shift();
    const lines = all.filter((line) => line.trim().length > 0);
    return { lines, partialHead, bytes: size };
  } catch {
    return { lines: [], partialHead: false, bytes: 0 };
  } finally {
    if (handle !== null) {
      try { fs.closeSync(handle); } catch { /* already closed */ }
    }
  }
}

function parseLines(lines: string[]): { records: JsonRecord[]; malformed: number } {
  const records: JsonRecord[] = [];
  let malformed = 0;
  for (const line of lines) {
    try {
      const parsed: unknown = JSON.parse(line);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        records.push(parsed as JsonRecord);
      } else {
        malformed += 1;
      }
    } catch {
      malformed += 1;
    }
  }
  return { records, malformed };
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readJsonFile<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
  } catch {
    return null;
  }
}

export async function GET(request: Request): Promise<NextResponse> {
  const warnings: string[] = [];
  const dir = dataDir();

  try {
    const url = new URL(request.url);
    const station = url.searchParams.get("station") || DEFAULT_STATION;

    const predictionTail = readTailLines(path.join(dir, "prediction_audit.jsonl"), MAX_TAIL_BYTES);
    const verificationTail = readTailLines(path.join(dir, "prediction_verification.jsonl"), MAX_TAIL_BYTES);

    if (predictionTail.partialHead) {
      warnings.push("prediction_audit.jsonl was read from the tail; forecastHistory is a partial window.");
    }
    if (verificationTail.partialHead) {
      warnings.push("prediction_verification.jsonl was read from the tail; verification is a partial window.");
    }

    const predictions = parseLines(predictionTail.lines);
    const verifications = parseLines(verificationTail.lines);
    if (predictions.malformed > 0) {
      warnings.push(`${predictions.malformed} prediction line(s) could not be parsed.`);
    }
    if (verifications.malformed > 0) {
      warnings.push(`${verifications.malformed} verification line(s) could not be parsed.`);
    }

    const forecast = predictions.records
      .filter((r) => asString(r.station_id) === station)
      .slice(-1)[0] ?? null;

    const forecastHistory = predictions.records.filter(
      (r) => asString(r.station_id) === station
    );

    const verificationRecords = verifications.records.filter(
      (r) => asString(r.station_id) === station
    );

    const bundles: Record<string, unknown> = {};
    for (const horizon of [1, 3, 6, 12, 24]) {
      bundles[`h${horizon}`] = readJsonFile(
        path.join(dir, "bundles", `h${horizon}`, "bundle_manifest.json")
      );
    }

    const verificationSummary = readJsonFile(
      path.join(dir, "prediction_verification_summary.json")
    );

    // Measured live, not modelled. Kept in its own field so the panel never has
    // to guess which of the two accuracy numbers it is showing.
    const liveSkill = readJsonFile(path.join(dir, "live_skill_report.json"));

    const payload = {
      schemaVersion: 1,
      generatedAtUtc: new Date().toISOString(),
      stationId: station,
      forecast,
      forecastHistory,
      forecastCoverage: {
        file: "prediction_audit.jsonl",
        bytes: predictionTail.bytes,
        recordsInWindow: predictions.records.length,
        partialWindow: predictionTail.partialHead,
      },
      verification: verificationSummary
        ? {
            summary: verificationSummary,
            records: verificationRecords,
          }
        : null,
      policy: readJsonFile(path.join(dir, "inference_policy.json")),
      bundles,
      expectedAccuracy: readJsonFile(path.join(dir, "expected_accuracy_reference.json")),
      liveSkill,
      warnings,
    };

    return NextResponse.json({ success: true, data: payload });
  } catch (error) {
    console.error("Error building prediction provenance payload:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to build prediction provenance payload.",
        error: String(error),
      },
      { status: 500 }
    );
  }
}
