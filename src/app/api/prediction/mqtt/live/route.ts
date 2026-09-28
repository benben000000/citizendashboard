import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const MAX_MQTT_AGE_SECONDS = Number(process.env.MQTT_STATUS_MAX_AGE_SECONDS || 300);

function parseTimestamp(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : null;
}

/**
 * GET /api/prediction/mqtt/live
 * Real-Time MQTT Telemetry & PINN-LNN Continuous Prediction Stream API.
 * Reads the latest high-frequency MQTT state without needing any REST API keys.
 */
export async function GET() {
  try {
    const filePath = path.join(
      process.cwd(),
      "prediction-model",
      "data",
      "mqtt_live_predictions.json"
    );

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({
        success: true,
        message: "MQTT stream listener initialized and standby for incoming telemetry packets.",
        data: {
          status: "STANDBY",
          last_updated: new Date().toISOString(),
          total_active_stations: 0,
          stations: {},
        },
      });
    }

    const rawData = fs.readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(rawData);
    const lastUpdated = parsed.last_updated;
    const lastUpdatedMs = parseTimestamp(lastUpdated);
    const ageSeconds = lastUpdatedMs === null ? null : Math.max(0, Math.floor((Date.now() - lastUpdatedMs) / 1000));
    const isActive = ageSeconds !== null && ageSeconds <= MAX_MQTT_AGE_SECONDS;

    return NextResponse.json({
      success: true,
      message: isActive
        ? "Live MQTT telemetry cache is current."
        : "MQTT telemetry cache is stale; the subscriber may be offline.",
      data: {
        ...parsed,
        status: isActive ? "ACTIVE" : "STALE",
        is_active: isActive,
        cache_age_seconds: ageSeconds,
        max_cache_age_seconds: MAX_MQTT_AGE_SECONDS,
      },
    });
  } catch (error) {
    console.error("Error reading live MQTT predictions:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to read live MQTT telemetry stream.",
        error: String(error),
      },
      { status: 500 }
    );
  }
}
