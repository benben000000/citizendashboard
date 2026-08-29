import { NextResponse } from "next/server";
import { getPortalSession } from "@/lib/auth/portal-auth";
import { exportService, ExportStreamType, ExportIntervalType, ExportFormatType } from "@/services/export.service";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const session = getPortalSession();
    if (!session.valid) {
      return NextResponse.json(
        { success: false, message: "Unauthorized. Please log in to the portal." },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(request.url);
    const stream = (searchParams.get("stream") || "raw") as ExportStreamType;
    const interval = (searchParams.get("interval") || "1m") as ExportIntervalType;
    const format = (searchParams.get("format") || "csv") as ExportFormatType;
    const stationId = searchParams.get("stationId") || "all";
    const startDate = searchParams.get("startDate") || undefined;
    const endDate = searchParams.get("endDate") || undefined;
    const isPreview = searchParams.get("preview") === "true";
    const previewLimit = isPreview ? 25 : undefined;

    const result = await exportService.getExportData({
      stream,
      interval,
      format,
      stationId,
      startDate,
      endDate,
      previewLimit,
    });

    if (isPreview) {
      return NextResponse.json({
        success: true,
        stream,
        interval,
        format,
        totalCount: result.totalCount,
        previewCount: result.records.length,
        records: result.records,
        filename: result.filename,
      });
    }

    // Direct binary/text download attachment response
    if (format === "csv") {
      return new NextResponse(result.csvString, {
        status: 200,
        headers: {
          "Content-Type": "text/csv; charset=utf-8",
          "Content-Disposition": `attachment; filename="${result.filename}"`,
        },
      });
    }

    if (format === "xlsx" && result.buffer) {
      return new NextResponse(new Uint8Array(result.buffer), {
        status: 200,
        headers: {
          "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "Content-Disposition": `attachment; filename="${result.filename}"`,
        },
      });
    }

    return new NextResponse(result.jsonString, {
      status: 200,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Disposition": `attachment; filename="${result.filename}"`,
      },
    });
  } catch (error) {
    console.error("Portal export error:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to generate export file",
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
