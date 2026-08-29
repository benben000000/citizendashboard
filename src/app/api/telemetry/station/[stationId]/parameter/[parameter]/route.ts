import { NextResponse } from "next/server";
import { telemetryService } from "@/services/telemetry.service";
import { formatErrorResponse } from "@/lib/utils/error";
import { CACHE_CONFIG, getCacheControlHeader } from "@/lib/config/cache.config";

export const dynamic = 'force-dynamic';
export const revalidate = CACHE_CONFIG.apiRoutes.telemetryParameter;

/**
 * GET /api/telemetry/station/[stationId]/parameter/[parameter]
 * Returns parameter-specific history for a station
 * Includes server-side caching and data transformation
 */
export async function GET(
  request: Request,
  context: { params: { stationId: string; parameter: string } | Promise<{ stationId: string; parameter: string }> }
) {
  try {
    const rawParams = context.params;
    const { stationId, parameter } = rawParams && typeof (rawParams as Promise<unknown>).then === "function" 
      ? await rawParams 
      : (rawParams as { stationId: string; parameter: string });
    const { searchParams } = new URL(request.url);
    const startDate = searchParams.get("startDate") || undefined;
    const endDate = searchParams.get("endDate") || undefined;
    const intervalParam = searchParams.get("interval");
    const interval = intervalParam ? Number(intervalParam) : undefined;

    if (!stationId) {
      return NextResponse.json(
        {
          success: false,
          message: "Station ID is required",
        },
        { status: 400 }
      );
    }

    if (!parameter) {
      return NextResponse.json(
        {
          success: false,
          message: "Parameter is required",
        },
        { status: 400 }
      );
    }

    // Reject invalid intervals before they reach the service cache key or upstream API.
    if (
      intervalParam &&
      (interval === undefined || !Number.isFinite(interval) || interval <= 0)
    ) {
      return NextResponse.json(
        {
          success: false,
          message: "Interval must be a positive number",
        },
        { status: 400 }
      );
    }

    // Use telemetry service which handles caching and transformation
    const data = await telemetryService.getStationParameterHistory(stationId, parameter, {
      startDate,
      endDate,
      interval,
    });

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
    console.error("Parameter history API error:", error);

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

