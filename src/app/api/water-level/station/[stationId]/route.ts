import { NextResponse } from "next/server";
import { CACHE_CONFIG, getCacheControlHeader } from "@/lib/config/cache.config";
import { formatErrorResponse } from "@/lib/utils/error";
import { waterLevelService } from "@/services/water-level.service";

export const dynamic = "force-dynamic";
export const revalidate = CACHE_CONFIG.apiRoutes.waterLevel;

/**
 * GET /api/water-level/station/[stationId]
 * Returns the latest water-level reading for one station.
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ stationId: string }> }
) {
  try {
    const { stationId } = await params;

    if (!stationId) {
      return NextResponse.json(
        {
          success: false,
          message: "Station ID is required",
        },
        { status: 400 }
      );
    }

    const data = await waterLevelService.getStationDashboardData(stationId);

    return NextResponse.json(
      {
        success: true,
        data,
      },
      {
        headers: {
          "Cache-Control": getCacheControlHeader(
            CACHE_CONFIG.http.waterLevel.sMaxAge,
            CACHE_CONFIG.http.waterLevel.staleWhileRevalidate
          ),
        },
      }
    );
  } catch (error) {
    console.error("Current water-level API error:", error);

    const errorResponse = formatErrorResponse(error);
    return NextResponse.json(
      {
        success: false,
        message: errorResponse.message,
      },
      { status: errorResponse.statusCode }
    );
  }
}
