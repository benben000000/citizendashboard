import { NextResponse } from "next/server";
import {
  benchmarkExportService,
  BenchmarkIntervalType,
  BenchmarkFormatType,
} from "@/services/benchmark-export.service";
import { isPortalAuthenticated, isPortalConfigured } from "@/lib/auth/portal-auth";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Benchmark export.
 *
 * AUTHENTICATION
 * --------------
 * This endpoint previously had no session check while running 7-horizon
 * inference across up to 23 stations with a 60s budget, and it is reachable
 * from a floating button mounted in the root layout. That is an unauthenticated
 * compute-amplification surface over historical telemetry, so it now requires a
 * valid portal session.
 */
function requireAuth(): NextResponse | null {
  if (!isPortalConfigured()) {
    return NextResponse.json(
      {
        success: false,
        message:
          "Benchmark export is unavailable: portal credentials are not configured on this deployment.",
      },
      { status: 503 }
    );
  }
  if (!isPortalAuthenticated()) {
    return NextResponse.json(
      { success: false, message: "Authentication required for benchmark export." },
      { status: 401 }
    );
  }
  return null;
}

export async function GET(request: Request) {
  const denied = requireAuth();
  if (denied) return denied;

  try {
    const { searchParams } = new URL(request.url);
    const stationId = searchParams.get("stationId") || "all";
    const startDate = searchParams.get("startDate") || undefined;
    const endDate = searchParams.get("endDate") || undefined;
    const interval = (searchParams.get("interval") || "5m") as BenchmarkIntervalType;
    const format = (searchParams.get("format") || "csv") as BenchmarkFormatType;
    const isPreview = searchParams.get("preview") === "true";
    const previewLimit = isPreview ? 25 : undefined;

    const result = await benchmarkExportService.generateBenchmarkExport({
      stationId,
      startDate,
      endDate,
      interval,
      format,
      previewLimit,
    });

    if (isPreview) {
      return NextResponse.json({
        success: true,
        interval: result.effectiveInterval || interval,
        requestedInterval: interval,
        wasAutoScaled: result.wasAutoScaled || false,
        format,
        totalCount: result.totalCount,
        previewCount: result.records.length,
        records: result.records,
        filename: result.filename,
      });
    }

    const commonHeaders: Record<string, string> = {
      "X-Benchmark-Interval": result.effectiveInterval || interval,
      "X-Benchmark-AutoScaled": String(result.wasAutoScaled || false),
      "X-Benchmark-TotalRows": String(result.totalCount),
      "Cache-Control": "no-store, max-age=0",
    };

    if (format === "csv" && result.csvString !== undefined) {
      return new NextResponse(result.csvString, {
        status: 200,
        headers: {
          ...commonHeaders,
          "Content-Type": "text/csv; charset=utf-8",
          "Content-Disposition": `attachment; filename="${result.filename}"`,
        },
      });
    }

    if (format === "xlsx" && result.buffer) {
      return new NextResponse(new Uint8Array(result.buffer), {
        status: 200,
        headers: {
          ...commonHeaders,
          "Content-Type":
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "Content-Disposition": `attachment; filename="${result.filename}"`,
        },
      });
    }

    return new NextResponse(result.jsonString || JSON.stringify(result.records), {
      status: 200,
      headers: {
        ...commonHeaders,
        "Content-Type": "application/json; charset=utf-8",
        "Content-Disposition": `attachment; filename="${result.filename}"`,
      },
    });
  } catch (error) {
    console.error("Benchmark export API error:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to generate benchmark dataset",
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const denied = requireAuth();
  if (denied) return denied;

  try {
    const body = await request.json().catch(() => ({}));
    const stationId = body.stationId || "all";
    const startDate = body.startDate || undefined;
    const endDate = body.endDate || undefined;
    const interval = (body.interval || "5m") as BenchmarkIntervalType;
    const format = (body.format || "csv") as BenchmarkFormatType;
    const isPreview = body.preview === true;
    const previewLimit = isPreview ? 25 : undefined;

    const result = await benchmarkExportService.generateBenchmarkExport({
      stationId,
      startDate,
      endDate,
      interval,
      format,
      previewLimit,
    });

    if (isPreview) {
      return NextResponse.json({
        success: true,
        interval,
        format,
        totalCount: result.totalCount,
        previewCount: result.records.length,
        records: result.records,
        filename: result.filename,
      });
    }

    return NextResponse.json({
      success: true,
      filename: result.filename,
      totalCount: result.totalCount,
      downloadUrl: `/api/benchmark/export?stationId=${encodeURIComponent(
        stationId
      )}&startDate=${encodeURIComponent(startDate || "")}&endDate=${encodeURIComponent(
        endDate || ""
      )}&interval=${interval}&format=${format}`,
    });
  } catch (error) {
    console.error("Benchmark export POST error:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to process benchmark export request",
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
