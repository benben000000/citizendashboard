import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

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

    return NextResponse.json({
      success: true,
      message: "Live MQTT Telemetry & PINN-LNN predictions retrieved successfully.",
      data: parsed,
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
