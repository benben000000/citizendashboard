"use client";

import React, { useState, useEffect, useCallback } from "react";
import { PortalHeader } from "./portal-header";
import { ExportFilterPanel } from "./export-filter-panel";
import { ExportPreviewTable } from "./export-preview-table";
import type { ExportStreamType, ExportIntervalType, ExportFormatType } from "@/services/export.service";
import { Database, FileDown, CheckCircle2, AlertCircle } from "lucide-react";

interface PortalClientDashboardProps {
  username: string;
}

export const PortalClientDashboard: React.FC<PortalClientDashboardProps> = ({ username }) => {
  const [stream, setStream] = useState<ExportStreamType>("processed");
  const [stationId, setStationId] = useState<string>("all");
  const [interval, setInterval] = useState<ExportIntervalType>("1h");
  
  // Default to last 7 days
  const now = new Date();
  const d7 = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const [startDate, setStartDate] = useState<string>(d7.toISOString().slice(0, 10));
  const [endDate, setEndDate] = useState<string>(now.toISOString().slice(0, 10));
  const [format, setFormat] = useState<ExportFormatType>("csv");

  const [previewRecords, setPreviewRecords] = useState<Record<string, unknown>[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [isLoadingPreview, setIsLoadingPreview] = useState<boolean>(true);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);
  const [downloadSuccessMessage, setDownloadSuccessMessage] = useState<string | null>(null);

  // Fetch live preview records when filters change
  const fetchPreview = useCallback(async () => {
    try {
      setIsLoadingPreview(true);
      const params = new URLSearchParams({
        stream,
        interval,
        stationId,
        startDate: startDate ? `${startDate}T00:00:00.000Z` : "",
        endDate: endDate ? `${endDate}T23:59:59.999Z` : "",
        format,
        preview: "true",
      });

      const res = await fetch(`/api/portal/export?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setPreviewRecords(data.records || []);
        setTotalCount(data.totalCount || 0);
      } else {
        setPreviewRecords([]);
        setTotalCount(0);
      }
    } catch (e) {
      console.error("Preview fetch error:", e);
      setPreviewRecords([]);
      setTotalCount(0);
    } finally {
      setIsLoadingPreview(false);
    }
  }, [stream, interval, stationId, startDate, endDate, format]);

  useEffect(() => {
    fetchPreview();
  }, [fetchPreview]);

  // Execute full dataset file download
  const handleDownload = async () => {
    try {
      setIsDownloading(true);
      setDownloadSuccessMessage(null);

      const params = new URLSearchParams({
        stream,
        interval,
        stationId,
        startDate: startDate ? `${startDate}T00:00:00.000Z` : "",
        endDate: endDate ? `${endDate}T23:59:59.999Z` : "",
        format,
        preview: "false",
      });

      const downloadUrl = `/api/portal/export?${params.toString()}`;
      
      // Trigger native browser attachment download
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.setAttribute("download", "");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      setDownloadSuccessMessage(`Successfully generated ${format.toUpperCase()} export with ${totalCount} records.`);
      setTimeout(() => setDownloadSuccessMessage(null), 5000);
    } catch (e) {
      console.error("Download error:", e);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0d] text-white flex flex-col selection:bg-orange-500 selection:text-black">
      {/* Top Navigation */}
      <PortalHeader username={username} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Hero Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <Database className="w-7 h-7 text-orange-500" />
              <span>Telemetry Logs & Data Export Vault</span>
            </h1>
            <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
              Extract and download raw sensor logs, physics-calibrated processed telemetry, and continuous PINN nowcasts with custom temporal aggregations.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="px-3.5 py-2 rounded-xl bg-zinc-900 border border-zinc-800 text-xs font-mono flex items-center gap-2 text-zinc-300">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Vault Status: Online</span>
            </div>
          </div>
        </div>

        {/* Download Success Notification */}
        {downloadSuccessMessage && (
          <div className="flex items-center gap-3 p-4 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-sm shadow-lg animate-in fade-in slide-in-from-top-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <span className="font-medium">{downloadSuccessMessage}</span>
          </div>
        )}

        {/* Filter Controls Panel */}
        <ExportFilterPanel
          stream={stream}
          onStreamChange={setStream}
          stationId={stationId}
          onStationChange={setStationId}
          interval={interval}
          onIntervalChange={setInterval}
          startDate={startDate}
          onStartDateChange={setStartDate}
          endDate={endDate}
          onEndDateChange={setEndDate}
          format={format}
          onFormatChange={setFormat}
          onDownload={handleDownload}
          isDownloading={isDownloading}
          totalRecordsCount={totalCount}
        />

        {/* Live Data Preview Table */}
        <ExportPreviewTable
          stream={stream}
          records={previewRecords}
          totalCount={totalCount}
          isLoading={isLoadingPreview}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/60 py-6 bg-[#0b0b0e] text-center text-xs text-zinc-400">
        <p>© 2026 Kloudtrack Citizen Prediction Platform • Telemetry Data Export Service</p>
      </footer>
    </div>
  );
};
