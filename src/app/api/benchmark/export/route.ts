import { NextResponse } from "next/server";
import {
  benchmarkExportService,
  BenchmarkIntervalType,
  BenchmarkFormatType,
} from "@/services/benchmark-export.service";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET(request: Request) {
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
