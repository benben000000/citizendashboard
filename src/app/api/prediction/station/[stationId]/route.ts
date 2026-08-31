import { NextResponse } from "next/server";
import { predictionService } from "@/services/prediction.service";
import type { PredictionHorizon } from "@/types/prediction";

export const dynamic = "force-dynamic";

/**
 * GET /api/prediction/station/[stationId]?horizon=24h
 * Returns continuous-time LNN water level predictions and telemetry impacts.
 */
export async function GET(
  request: Request,
  { params }: { params: { stationId: string } | Promise<{ stationId: string }> }
) {
  try {
    const resolvedParams = await Promise.resolve(params);
    const stationId = resolvedParams?.stationId;

    if (!stationId) {
      return NextResponse.json(
        { success: false, message: "Station ID is required" },
        { status: 400 }
      );
    }

    const { searchParams } = new URL(request.url);
    const horizon = (searchParams.get("horizon") as PredictionHorizon) || "1h";

    const data = await predictionService.getPredictionForStation(stationId, horizon);

    return NextResponse.json(
      {
        success: true,
        data,
      },
      {
        headers: {
          "Cache-Control": "public, s-maxage=300, stale-while-revalidate=600",
        },
      }
    );
  } catch (error) {
    console.error("Prediction API error:", error);
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Internal server error",
      },
      { status: 500 }
    );
  }
}
