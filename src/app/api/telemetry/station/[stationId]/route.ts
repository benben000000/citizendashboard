import { NextResponse } from "next/server";
import { telemetryService } from "@/services/telemetry.service";
import { formatErrorResponse } from "@/lib/utils/error";
import { CACHE_CONFIG, getCacheControlHeader } from "@/lib/config/cache.config";

export const dynamic = 'force-dynamic'; 
export const revalidate = CACHE_CONFIG.apiRoutes.telemetry;

/**
 * GET /api/telemetry/station/[stationId]
 * Returns dashboard data for a single station
 * Includes server-side caching and data transformation
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

    // Use telemetry service which handles caching and transformation
    const data = await telemetryService.getStationDashboardData(stationId);

    // Add cache headers to response
    return NextResponse.json(
      {
        success: true,
        data,
      },
      {
        headers: {
          'Cache-Control': getCacheControlHeader(
            CACHE_CONFIG.http.telemetry.sMaxAge,
            CACHE_CONFIG.http.telemetry.staleWhileRevalidate
          ),
        },
      }
    );
  } catch (error) {
    console.error("Station dashboard API error:", error);

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

