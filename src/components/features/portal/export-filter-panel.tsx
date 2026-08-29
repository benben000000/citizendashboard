"use client";

import React from "react";
import {
  Download,
  Calendar,
  Layers,
  FileSpreadsheet,
  Clock,
  Radio,
  Sparkles,
  Activity,
  Cpu,
  Loader2,
} from "lucide-react";
import type { ExportStreamType, ExportIntervalType, ExportFormatType } from "@/services/export.service";
import hardCodedStations from "@/lib/constants/stations.json";

interface ExportFilterPanelProps {
  stream: ExportStreamType;
  onStreamChange: (stream: ExportStreamType) => void;
  stationId: string;
  onStationChange: (stationId: string) => void;
  interval: ExportIntervalType;
  onIntervalChange: (interval: ExportIntervalType) => void;
  startDate: string;
  onStartDateChange: (startDate: string) => void;
  endDate: string;
  onEndDateChange: (endDate: string) => void;
  format: ExportFormatType;
  onFormatChange: (format: ExportFormatType) => void;
  onDownload: () => void;
  isDownloading: boolean;
  totalRecordsCount: number;
}

export const ExportFilterPanel: React.FC<ExportFilterPanelProps> = ({
  stream,
  onStreamChange,
  stationId,
  onStationChange,
  interval,
  onIntervalChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  format,
  onFormatChange,
  onDownload,
  isDownloading,
  totalRecordsCount,
}) => {
  // Station list lookup
  const weatherStations = hardCodedStations.weather.stationIdToFetch;

  // Preset Date Range buttons
  const setDatePreset = (preset: "today" | "yesterday" | "7d" | "30d") => {
    const now = new Date();
    const todayStr = now.toISOString().slice(0, 10);

    if (preset === "today") {
      onStartDateChange(todayStr);
      onEndDateChange(todayStr);
    } else if (preset === "yesterday") {
      const yest = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const yestStr = yest.toISOString().slice(0, 10);
      onStartDateChange(yestStr);
      onEndDateChange(todayStr);
    } else if (preset === "7d") {
      const d7 = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      onStartDateChange(d7.toISOString().slice(0, 10));
      onEndDateChange(todayStr);
    } else if (preset === "30d") {
      const d30 = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      onStartDateChange(d30.toISOString().slice(0, 10));
      onEndDateChange(todayStr);
    }
  };

  return (
    <div className="w-full bg-[#121217] border border-zinc-800/80 rounded-2xl p-4 sm:p-6 shadow-2xl space-y-6">
      {/* Stream Tabs Selector */}
      <div>
        <label className="text-xs font-mono font-semibold uppercase tracking-wider text-orange-400/90 mb-2.5 flex items-center gap-1.5">
          <Layers className="w-4 h-4 text-orange-500" />
          <span>1. Select Data Stream</span>
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-3">
          {/* Raw Telemetry */}
          <button
            type="button"
            onClick={() => onStreamChange("raw")}
            className={`flex items-start gap-3 p-3.5 rounded-xl border text-left transition cursor-pointer ${
              stream === "raw"
                ? "bg-orange-500/15 border-orange-500 text-white shadow-lg shadow-orange-500/10"
                : "bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
            }`}
          >
            <div className={`p-2 rounded-lg ${stream === "raw" ? "bg-orange-500 text-white" : "bg-zinc-800 text-zinc-400"}`}>
              <Radio className="w-4 h-4" />
            </div>
            <div>
              <div className="font-semibold text-sm text-white">Raw Telemetry</div>
              <div className="text-xs text-zinc-400 mt-0.5 leading-snug">
                Unfiltered MQTT broadcasts from physical sensors
              </div>
            </div>
          </button>

          {/* Processed Telemetry */}
          <button
            type="button"
            onClick={() => onStreamChange("processed")}
            className={`flex items-start gap-3 p-3.5 rounded-xl border text-left transition cursor-pointer ${
              stream === "processed"
                ? "bg-amber-500/15 border-amber-500 text-white shadow-lg shadow-amber-500/10"
                : "bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
            }`}
          >
            <div className={`p-2 rounded-lg ${stream === "processed" ? "bg-amber-500 text-black font-bold" : "bg-zinc-800 text-zinc-400"}`}>
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <div className="font-semibold text-sm text-white">Processed Telemetry</div>
              <div className="text-xs text-zinc-400 mt-0.5 leading-snug">
                Denoised, calibrated, NOAA heat index & spatial estimates
              </div>
            </div>
          </button>

          {/* Prediction Nowcasts */}
          <button
            type="button"
            onClick={() => onStreamChange("prediction")}
            className={`flex items-start gap-3 p-3.5 rounded-xl border text-left transition cursor-pointer ${
              stream === "prediction"
                ? "bg-orange-600/15 border-orange-400 text-white shadow-lg shadow-orange-600/10"
                : "bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
            }`}
          >
            <div className={`p-2 rounded-lg ${stream === "prediction" ? "bg-orange-400 text-black font-bold" : "bg-zinc-800 text-zinc-400"}`}>
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="font-semibold text-sm text-white">Prediction Nowcasts</div>
              <div className="text-xs text-zinc-400 mt-0.5 leading-snug">
                PINN-LNN continuous forecasts (+1h to +72h lead time)
              </div>
            </div>
          </button>
        </div>
      </div>

      {/* Filter Row: Station & Timeline Granularity */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Station Selector */}
        <div>
          <label className="text-xs font-mono font-semibold uppercase tracking-wider text-orange-400/90 mb-2 block">
            2. Station Scope
          </label>
          <select
            value={stationId}
            onChange={(e) => onStationChange(e.target.value)}
            className="w-full h-11 px-3.5 rounded-xl bg-zinc-900 border border-zinc-700 text-white text-sm focus:outline-none focus:border-orange-500 focus:ring-1 focus:ring-orange-500 transition"
          >
            <option value="all">⚡ All Stations (Bulk Regional Archive)</option>
            <optgroup label="Central Luzon AWS Weather Stations">
              {weatherStations.map((st) => (
                <option key={st.stationId} value={st.stationId}>
                  {st.location.toUpperCase()} AWS ({st.stationId})
                </option>
              ))}
            </optgroup>
            <optgroup label="WLMS River Stage Gauges">
              <option value="O3z0j5bG">CALUMPIT WLMS Gauge (O3z0j5bG)</option>
            </optgroup>
          </select>
        </div>

        {/* Timeline Granularity */}
        <div>
          <label className="text-xs font-mono font-semibold uppercase tracking-wider text-orange-400/90 mb-2 flex items-center gap-1">
            <Clock className="w-3.5 h-3.5 text-orange-500" />
            <span>3. Timeline Resolution</span>
          </label>
          <div className="grid grid-cols-4 gap-1.5 bg-zinc-900 p-1 rounded-xl border border-zinc-800 h-11">
            {(
              [
                { id: "1m", label: "1 Min" },
                { id: "30m", label: "30 Min" },
                { id: "1h", label: "1 Hour" },
                { id: "1d", label: "1 Day" },
              ] as const
            ).map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onIntervalChange(item.id)}
                className={`text-xs font-semibold rounded-lg transition cursor-pointer ${
                  interval === item.id
                    ? "bg-orange-500 text-black shadow-md"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Date Range & Presets */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs font-mono font-semibold uppercase tracking-wider text-orange-400/90 flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5 text-orange-500" />
            <span>4. Date Range Filter</span>
          </label>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setDatePreset("today")}
              className="px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white transition cursor-pointer"
            >
              Today
            </button>
            <button
              type="button"
              onClick={() => setDatePreset("yesterday")}
              className="px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white transition cursor-pointer"
            >
              Yesterday
            </button>
            <button
              type="button"
              onClick={() => setDatePreset("7d")}
              className="px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white transition cursor-pointer"
            >
              Last 7 Days
            </button>
            <button
              type="button"
              onClick={() => setDatePreset("30d")}
              className="px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white transition cursor-pointer"
            >
              Last 30 Days
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <span className="text-[11px] text-zinc-400 mb-1 block">Start Date:</span>
            <input
              type="date"
              value={startDate}
              onChange={(e) => onStartDateChange(e.target.value)}
              className="w-full h-11 px-3.5 rounded-xl bg-zinc-900 border border-zinc-700 text-white text-sm focus:outline-none focus:border-orange-500 transition [color-scheme:dark]"
            />
          </div>
          <div>
            <span className="text-[11px] text-zinc-400 mb-1 block">End Date:</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => onEndDateChange(e.target.value)}
              className="w-full h-11 px-3.5 rounded-xl bg-zinc-900 border border-zinc-700 text-white text-sm focus:outline-none focus:border-orange-500 transition [color-scheme:dark]"
            />
          </div>
        </div>
      </div>

      {/* Format & Download Trigger */}
      <div className="pt-2 border-t border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Export Format Selector */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs font-mono font-semibold uppercase text-zinc-400">Format:</span>
          <div className="inline-flex rounded-xl bg-zinc-900 p-1 border border-zinc-800">
            {(
              [
                { id: "csv", label: "CSV (.csv)" },
                { id: "xlsx", label: "Excel (.xlsx)" },
                { id: "json", label: "JSON (.json)" },
              ] as const
            ).map((fmt) => (
              <button
                key={fmt.id}
                type="button"
                onClick={() => onFormatChange(fmt.id)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition cursor-pointer ${
                  format === fmt.id
                    ? "bg-amber-400 text-black shadow-sm font-bold"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                {fmt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Download Action CTA */}
        <button
          type="button"
          onClick={onDownload}
          disabled={isDownloading || totalRecordsCount === 0}
          className="w-full sm:w-auto min-w-56 h-12 px-6 rounded-xl font-bold text-sm bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-400 hover:to-amber-400 text-black shadow-lg shadow-orange-500/20 hover:shadow-orange-500/30 flex items-center justify-center gap-2 transition transform active:scale-[0.98] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isDownloading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-black" />
              <span>Generating Export...</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4 text-black" />
              <span>Download {format.toUpperCase()} ({totalRecordsCount} Rows)</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
