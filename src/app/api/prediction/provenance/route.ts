import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const MAX_TAIL_BYTES = Number(process.env.PROVENANCE_TAIL_BYTES || 2000000);
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
    const partialHead = start > 0;
    if (partialHead && all.length > 0) all.shift();
    return { lines: all.filter((l) => l.trim().length > 0), partialHead, bytes: size };
  } catch {
    return { lines: [], partialHead: false, bytes: 0 };
  } finally {
    if (handle !== null) { try { fs.closeSync(handle); } catch {} }
  }
}

function parseLines(lines: string[]): { records: any[]; malformed: number } {
  const records: any[] = []; let malformed = 0;
  for (const line of lines) {
    try {
      const parsed: unknown = JSON.parse(line);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) records.push(parsed);
      else malformed += 1;
    } catch { malformed += 1; }
  }
  return { records, malformed };
}

function readJsonFile(file: string): any {
  try { return JSON.parse(fs.readFileSync(file, "utf-8")); } catch { return null; }
}

export async function GET(request: Request): Promise<NextResponse> {
  const warnings: string[] = [];
  const dir = dataDir();

  try {
    const url = new URL(request.url);
    const station = url.searchParams.get("station") || DEFAULT_STATION;

    const predictionTail = readTailLines(path.join(dir, "prediction_audit.jsonl"), MAX_TAIL_BYTES);
    const verificationTail = readTailLines(path.join(dir, "prediction_verification.jsonl"), MAX_TAIL_BYTES);

    if (predictionTail.partialHead) warnings.push("prediction_audit.jsonl was read from the tail; forecastHistory is a partial window.");
    if (verificationTail.partialHead) warnings.push("prediction_verification.jsonl was read from the tail; verification is a partial window.");

    const predictions = parseLines(predictionTail.lines);
    const verifications = parseLines(verificationTail.lines);
    if (predictions.malformed > 0) warnings.push(predictions.malformed + " prediction line(s) could not be parsed.");
    if (verifications.malformed > 0) warnings.push(verifications.malformed + " verification line(s) could not be parsed.");

    const forecast = predictions.records.filter((r) => r.station_id === station).slice(-1)[0] ?? null;
    const forecastHistory = predictions.records.filter((r) => r.station_id === station);
    const verificationRecords = verifications.records.filter((r) => r.station_id === station);

    const bundles: Record<string, unknown> = {};
    for (const horizon of [1, 3, 6, 12, 24]) {
      bundles["h" + horizon] = readJsonFile(path.join(dir, "bundles", "h" + horizon, "bundle_manifest.json"));
    }

    const summary = readJsonFile(path.join(dir, "prediction_verification_summary.json"));

    const payload = {
      schemaVersion: 1,
      generatedAtUtc: new Date().toISOString(),
      stationId: station,
      forecast,
      forecastHistory,
      forecastCoverage: {
        file: "prediction_audit.jsonl",
        bytes: predictionTail.bytes,
        readWindowBytes: MAX_TAIL_BYTES,
        recordsInWindow: predictions.records.length,
        partialWindow: predictionTail.partialHead,
      },
      verification: summary ? { summary: summary, records: verificationRecords } : null,
      policy: readJsonFile(path.join(dir, "inference_policy.json")),
      bundles,
      expectedAccuracy: readJsonFile(path.join(dir, "expected_accuracy_reference.json")),
      liveSkill: readJsonFile(path.join(dir, "live_skill_report.json")),
      warnings,
    };

    return NextResponse.json({ success: true, data: payload });
  } catch (error) {
    console.error("Error building prediction provenance payload:", error);
    return NextResponse.json(
      { success: false, message: "Failed to build prediction provenance payload.", error: String(error) },
      { status: 500 }
    );
  }
}
