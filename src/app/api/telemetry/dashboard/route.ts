import { NextResponse } from "next/server";
import { CACHE_CONFIG, getCacheControlHeader } from "@/lib/config/cache.config";
import { formatErrorResponse } from "@/lib/utils/error";
import { telemetryService } from "@/services/telemetry.service";

export const dynamic = "force-dynamic";
export const revalidate = CACHE_CONFIG.apiRoutes.telemetry;

/**
 * GET /api/telemetry/dashboard
 * Returns dashboard data for all stations from one upstream /telemetry/dashboard request.
 */
export async function GET() {
  try {
    const data = await telemetryService.getDashboardStations();

    return NextResponse.json(
      {
        success: true,
        data,
      },
      {
        headers: {
          "Cache-Control": getCacheControlHeader(
            CACHE_CONFIG.http.telemetry.sMaxAge,
            CACHE_CONFIG.http.telemetry.staleWhileRevalidate
          ),
        },
      }
    );
  } catch (error) {
    console.error("Telemetry dashboard API error:", error);

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
