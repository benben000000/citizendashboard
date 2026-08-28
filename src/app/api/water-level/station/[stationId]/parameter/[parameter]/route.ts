import { NextResponse } from "next/server";
import { CACHE_CONFIG, getCacheControlHeader } from "@/lib/config/cache.config";
import { formatErrorResponse } from "@/lib/utils/error";
import { waterLevelService } from "@/services/water-level.service";

export const dynamic = "force-dynamic";
export const revalidate = CACHE_CONFIG.apiRoutes.waterLevelParameter;

/**
 * GET /api/water-level/station/[stationId]/parameter/[parameter]
 * Returns parameter-specific history for a station.
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ stationId: string; parameter: string }> }
) {
  try {
    const { stationId, parameter } = await params;
    const { searchParams } = new URL(request.url);
    const skipParam = searchParams.get("skip");
    const takeParam = searchParams.get("take");
    const intervalParam = searchParams.get("interval");
    const startDate = searchParams.get("startDate") || undefined;
    const endDate = searchParams.get("endDate") || undefined;

    const skip = parseOptionalNumber(skipParam);
    const take = parseOptionalNumber(takeParam);
    const interval = parseOptionalNumber(intervalParam);

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

    if (skipParam && (skip === undefined || skip < 0)) {
      return NextResponse.json(
        {
          success: false,
          message: "Skip must be a non-negative number",
        },
        { status: 400 }
      );
    }

    if (takeParam && (take === undefined || take <= 0)) {
      return NextResponse.json(
        {
          success: false,
          message: "Take must be a positive number",
        },
        { status: 400 }
      );
    }

    if (intervalParam && (interval === undefined || interval <= 0)) {
      return NextResponse.json(
        {
          success: false,
          message: "Interval must be a positive number",
        },
        { status: 400 }
      );
    }

    const data = await waterLevelService.getStationParameterHistory(stationId, parameter, {
      skip,
      take,
      interval,
      startDate,
      endDate,
    });

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
    console.error("Water-level parameter history API error:", error);

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

function parseOptionalNumber(value: string | null): number | undefined {
  if (!value) return undefined;

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}
