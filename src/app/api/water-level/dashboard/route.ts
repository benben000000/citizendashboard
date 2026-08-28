import { NextResponse } from "next/server";
import { CACHE_CONFIG, getCacheControlHeader } from "@/lib/config/cache.config";
import { formatErrorResponse } from "@/lib/utils/error";
import { waterLevelService } from "@/services/water-level.service";

export const dynamic = "force-dynamic";
export const revalidate = CACHE_CONFIG.apiRoutes.waterLevel;

/**
 * GET /api/water-level/dashboard
 * Returns all readable water-level stations with their latest reading.
 */
export async function GET() {
  try {
    const data = await waterLevelService.getDashboardStations();

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
    console.error("Water-level dashboard API error:", error);

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
