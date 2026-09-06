"use client";

import React, { useState } from "react";
import {
  X,
  Download,
  Calendar as CalendarIcon,
  Database,
  Layers,
  Clock,
  Sparkles,
  FileSpreadsheet,
  FileText,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { DEFAULT_CENTRAL_LUZON_STATIONS } from "@/lib/constants/default-stations";

interface BenchmarkExportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function BenchmarkExportModal({
  isOpen,
  onClose,
}: BenchmarkExportModalProps) {
  const now = new Date();
  const past24h = new Date(now.getTime() - 24 * 60 * 60 * 1000);

  // Default end-to-end dates in local datetime-local format (YYYY-MM-DDTHH:mm)
  const formatForInput = (d: Date) => {
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
      d.getHours()
    )}:${pad(d.getMinutes())}`;
  };

  const [startDate, setStartDate] = useState(formatForInput(past24h));
  const [endDate, setEndDate] = useState(formatForInput(now));
  const [selectedStation, setSelectedStation] = useState<string>("all");
  const [interval, setInterval] = useState<"5m" | "15m" | "1h">("5m");
  const [format, setFormat] = useState<"csv" | "xlsx" | "json">("csv");
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<string | null>(null);

  if (!isOpen) return null;

  // Preset Date Handlers
  const handlePreset = (preset: "today" | "yesterday" | "3days" | "7days") => {
    const current = new Date();
    let start = new Date();

    if (preset === "today") {
      start = new Date(current.getFullYear(), current.getMonth(), current.getDate(), 0, 0, 0);
    } else if (preset === "yesterday") {
      start = new Date(current.getFullYear(), current.getMonth(), current.getDate() - 1, 0, 0, 0);
      const endYesterday = new Date(
        current.getFullYear(),
        current.getMonth(),
        current.getDate() - 1,
        23,
        59,
        59
      );
      setStartDate(formatForInput(start));
      setEndDate(formatForInput(endYesterday));
      return;
    } else if (preset === "3days") {
      start = new Date(current.getTime() - 3 * 24 * 60 * 60 * 1000);
    } else if (preset === "7days") {
      start = new Date(current.getTime() - 7 * 24 * 60 * 60 * 1000);
    }

    setStartDate(formatForInput(start));
    setEndDate(formatForInput(current));
  };

  // Calculate estimated records & serverless safety thresholds
  const startMs = new Date(startDate).getTime();
  const endMs = new Date(endDate).getTime();
  const diffMinutes = Math.max(0, Math.floor((endMs - startMs) / (60 * 1000)));
  const stepMinutes = interval === "5m" ? 5 : interval === "15m" ? 15 : 60;
  const numStations = selectedStation === "all" ? DEFAULT_CENTRAL_LUZON_STATIONS.length : 1;
  const estimatedRows = Math.floor(diffMinutes / stepMinutes) * numStations;

  const maxSafeRows = format === "xlsx" ? 3800 : 12000;
  const isLargeDataset = estimatedRows > maxSafeRows;

  let predictedInterval: string = interval;
  if (isLargeDataset) {
    const totalMins = diffMinutes * numStations;
    const rawStep = Math.ceil(totalMins / maxSafeRows);
    if (rawStep <= 10) predictedInterval = "10m";
    else if (rawStep <= 15) predictedInterval = "15m";
    else if (rawStep <= 30) predictedInterval = "30m";
    else if (rawStep <= 60) predictedInterval = "1h";
    else if (rawStep <= 120) predictedInterval = "2h";
    else if (rawStep <= 180) predictedInterval = "3h";
    else if (rawStep <= 360) predictedInterval = "6h";
    else predictedInterval = `${Math.ceil(rawStep / 60)}h`;
  }

  const handleDownload = async () => {
    try {
      setIsDownloading(true);
      setErrorMsg(null);
      setDownloadSuccess(false);
      setSuccessInfo(null);

      const parsedStart = startDate ? new Date(startDate) : null;
      const parsedEnd = endDate ? new Date(endDate) : null;

      const queryParams: Record<string, string> = {
        stationId: selectedStation,
        interval,
        format,
      };

      if (parsedStart && !isNaN(parsedStart.getTime())) {
        queryParams.startDate = parsedStart.toISOString();
      }
      if (parsedEnd && !isNaN(parsedEnd.getTime())) {
        queryParams.endDate = parsedEnd.toISOString();
      }

      const params = new URLSearchParams(queryParams);
      const downloadUrl = `/api/benchmark/export?${params.toString()}`;

      // Trigger native browser download
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        const errJson = await response.json().catch(() => null);
        throw new Error(errJson?.message || errJson?.error || `Server returned HTTP ${response.status}`);
      }

      const autoScaled = response.headers.get("X-Benchmark-AutoScaled") === "true";
      const effectiveInt = response.headers.get("X-Benchmark-Interval") || interval;
      const totalRowsHeader = response.headers.get("X-Benchmark-TotalRows");

      const blob = await response.blob();
      const contentDisposition = response.headers.get("Content-Disposition");
      let filename = `Kloudtrack_Benchmark_${interval}_${selectedStation}.${format}`;
      if (contentDisposition && contentDisposition.includes("filename=")) {
        const match = contentDisposition.match(/filename="?([^"]+)"?/);
        if (match && match[1]) filename = match[1];
      }

      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);

      const rowText = totalRowsHeader ? `${Number(totalRowsHeader).toLocaleString()} rows` : "";
      const scaleText = autoScaled
        ? ` (${effectiveInt} interval auto-optimized for serverless stability)`
        : ` (${effectiveInt} interval)`;
      setSuccessInfo(`Benchmark log generated: ${rowText}${scaleText}.`);
      setDownloadSuccess(true);
      setTimeout(() => setDownloadSuccess(false), 6000);
    } catch (err) {
      console.error("Export download failed:", err);
      setErrorMsg(err instanceof Error ? err.message : "Failed to download dataset");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-2xl bg-white/90 dark:bg-slate-900/90 border border-white/20 dark:border-slate-800 rounded-3xl shadow-2xl overflow-hidden backdrop-blur-xl transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 md:p-6 border-b border-border/40 bg-white/40 dark:bg-slate-800/40">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-2xl bg-primary/10 text-primary border border-primary/20">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg md:text-xl font-bold text-foreground flex items-center gap-2">
                Download Ground-Truth Benchmark Log
                <span className="text-[11px] font-extrabold uppercase px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                  Raw • Processed • 1h-72h Predictions
                </span>
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Logs unmanipulated Raw MQTT baseline vs. Processed Real-Time vs. PINN-LNN Multi-Horizon Predictions
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-slate-200/50 dark:hover:bg-slate-800 transition"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 md:p-6 space-y-5 max-h-[75vh] overflow-y-auto">
          {/* Quick Date Range Presets */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <CalendarIcon className="h-3.5 w-3.5 text-primary" />
                Select End-to-End Date Range
              </label>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => handlePreset("today")}
                  className="text-[11px] font-semibold px-2 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-primary/20 hover:text-primary transition"
                >
                  Today
                </button>
                <button
                  type="button"
                  onClick={() => handlePreset("yesterday")}
                  className="text-[11px] font-semibold px-2 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-primary/20 hover:text-primary transition"
                >
                  Yesterday
                </button>
                <button
                  type="button"
                  onClick={() => handlePreset("3days")}
                  className="text-[11px] font-semibold px-2 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-primary/20 hover:text-primary transition"
                >
                  Past 3 Days
                </button>
                <button
                  type="button"
                  onClick={() => handlePreset("7days")}
                  className="text-[11px] font-semibold px-2 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-primary/20 hover:text-primary transition"
                >
                  Past 7 Days
                </button>
              </div>
            </div>

            {/* Start & End Date Inputs */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1">
                <span className="text-[11px] font-medium text-muted-foreground">Start Date & Time</span>
                <input
                  type="datetime-local"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full px-3 py-2 text-sm font-semibold rounded-xl border border-border/70 bg-background/80 focus:ring-2 focus:ring-primary/40 focus:outline-none"
                />
              </div>
              <div className="space-y-1">
                <span className="text-[11px] font-medium text-muted-foreground">End Date & Time</span>
                <input
                  type="datetime-local"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full px-3 py-2 text-sm font-semibold rounded-xl border border-border/70 bg-background/80 focus:ring-2 focus:ring-primary/40 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Station Selection */}
          <div>
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5 mb-2">
              <Layers className="h-3.5 w-3.5 text-primary" />
              Target Station Node
            </label>
            <select
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
              className="w-full px-3 py-2.5 text-sm font-semibold rounded-xl border border-border/70 bg-background/80 focus:ring-2 focus:ring-primary/40 focus:outline-none cursor-pointer"
            >
              <option value="all">All Central Luzon Stations (23 IoT Nodes)</option>
              {DEFAULT_CENTRAL_LUZON_STATIONS.map((st) => (
                <option key={st.stationPublicId} value={st.stationPublicId}>
                  {st.stationName} ({st.stationPublicId}) • {st.city || st.state}
                </option>
              ))}
            </select>
          </div>

          {/* Interval & Export Format Selection */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Interval */}
            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5 mb-2">
                <Clock className="h-3.5 w-3.5 text-primary" />
                Time Resolution (Gap)
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(
                  [
                    { id: "5m", label: "5 Min", desc: "Default" },
                    { id: "15m", label: "15 Min", desc: "Standard" },
                    { id: "1h", label: "1 Hour", desc: "Hourly" },
                  ] as const
                ).map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setInterval(opt.id)}
                    className={`p-2 rounded-xl text-center border transition ${
                      interval === opt.id
                        ? "border-primary bg-primary/10 text-primary font-bold shadow-sm"
                        : "border-border/60 bg-slate-50/50 dark:bg-slate-800/40 text-muted-foreground hover:bg-slate-100 dark:hover:bg-slate-800"
                    }`}
                  >
                    <div className="text-xs font-bold">{opt.label}</div>
                    <div className="text-[10px] opacity-75">{opt.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Format */}
            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5 mb-2">
                <FileSpreadsheet className="h-3.5 w-3.5 text-primary" />
                Export Format
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(
                  [
                    { id: "csv", label: "CSV", icon: FileText },
                    { id: "xlsx", label: "Excel", icon: FileSpreadsheet },
                    { id: "json", label: "JSON", icon: FileCode },
                  ] as const
                ).map((fmt) => (
                  <button
                    key={fmt.id}
                    type="button"
                    onClick={() => setFormat(fmt.id)}
                    className={`p-2 rounded-xl text-center border transition flex flex-col items-center justify-center gap-1 ${
                      format === fmt.id
                        ? "border-primary bg-primary/10 text-primary font-bold shadow-sm"
                        : "border-border/60 bg-slate-50/50 dark:bg-slate-800/40 text-muted-foreground hover:bg-slate-100 dark:hover:bg-slate-800"
                    }`}
                  >
                    <fmt.icon className="h-4 w-4" />
                    <span className="text-xs font-bold">{fmt.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Data Correlation Matrix Summary Card */}
          <div className="p-4 rounded-2xl bg-slate-100/70 dark:bg-slate-800/60 border border-border/50 space-y-2">
            <div className="flex items-center justify-between text-xs font-bold text-foreground">
              <span className="flex items-center gap-1.5">
                <Sparkles className="h-4 w-4 text-amber-500" />
                Ground-Truth Multi-Stream Matrix
              </span>
              <span className="text-[11px] font-mono text-muted-foreground">
                Est. ~{estimatedRows.toLocaleString()} raw steps
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px] text-muted-foreground">
              <div className="p-2 rounded-xl bg-white/60 dark:bg-slate-900/60 border border-border/40">
                <strong className="text-foreground block mb-0.5">1. Raw MQTT Telemetry</strong>
                Baseline control: temp, hourly & daily precip, humidity, heat index, wind, pressure, light, UV, water stage.
              </div>
              <div className="p-2 rounded-xl bg-white/60 dark:bg-slate-900/60 border border-border/40">
                <strong className="text-foreground block mb-0.5">2. Processed Real-Time</strong>
                Weather page values: denoised physics, spatial kriging flags, continuous station state.
              </div>
              <div className="p-2 rounded-xl bg-white/60 dark:bg-slate-900/60 border border-border/40">
                <strong className="text-foreground block mb-0.5">3. Multi-Horizon Forecasts</strong>
                Prediction page PINN-LNN ODE: 1h, 3h, 6h, 12h, 24h, 48h, 72h across all 10 stats.
              </div>
            </div>

            {/* Smart Serverless Stability Notice for Large Multi-Station / Multi-Week ranges */}
            {isLargeDataset && (
              <div className="mt-2 p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-700 dark:text-amber-400 flex items-start gap-2">
                <Sparkles className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
                <div>
                  <strong className="font-semibold block">Multi-station serverless auto-optimization active:</strong>
                  Requesting ~{estimatedRows.toLocaleString()} rows would exceed serverless payload thresholds (&gt;50MB). 
                  The server will automatically organize each station into its own tab and adapt resolution to <strong>{predictedInterval}</strong> for instant 1-click download. 
                  <span className="italic opacity-80 block mt-0.5">Tip: To download 5-minute precision for a month, select an individual station from the dropdown above.</span>
                </div>
              </div>
            )}
          </div>

          {/* Feedback messages */}
          {errorMsg && (
            <div className="flex items-center gap-2 p-3 text-xs font-medium text-rose-600 bg-rose-50 dark:bg-rose-950/40 rounded-xl border border-rose-200 dark:border-rose-900">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {downloadSuccess && (
            <div className="flex items-center gap-2 p-3 text-xs font-medium text-emerald-600 bg-emerald-50 dark:bg-emerald-950/40 rounded-xl border border-emerald-200 dark:border-emerald-900">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>{successInfo || "Comparative benchmark log generated and downloaded successfully!"}</span>
            </div>
          )}
        </div>

        {/* Modal Footer / Action Button */}
        <div className="flex items-center justify-between p-5 border-t border-border/40 bg-white/40 dark:bg-slate-800/40">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2.5 text-xs font-semibold rounded-xl text-muted-foreground hover:text-foreground hover:bg-slate-200/60 dark:hover:bg-slate-800 transition"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={isDownloading}
            className="flex items-center gap-2 px-6 py-2.5 text-xs font-bold uppercase tracking-wider rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/20 transition disabled:opacity-50 cursor-pointer"
          >
            {isDownloading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Generating Log...</span>
              </>
            ) : (
              <>
                <Download className="h-4 w-4" />
                <span>Download Dataset ({format.toUpperCase()})</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
