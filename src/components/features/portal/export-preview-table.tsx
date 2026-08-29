"use client";

import React from "react";
import { Table, Eye, FileText, Database, Layers } from "lucide-react";
import type { ExportStreamType } from "@/services/export.service";

interface ExportPreviewTableProps {
  stream: ExportStreamType;
  records: Record<string, unknown>[];
  totalCount: number;
  isLoading: boolean;
}

export const ExportPreviewTable: React.FC<ExportPreviewTableProps> = ({
  stream,
  records,
  totalCount,
  isLoading,
}) => {
  if (isLoading) {
    return (
      <div className="w-full bg-[#121217] border border-zinc-800/80 rounded-2xl p-8 flex flex-col items-center justify-center min-h-64 text-zinc-400">
        <div className="w-8 h-8 rounded-full border-2 border-orange-500 border-t-transparent animate-spin mb-3" />
        <p className="text-sm font-mono text-zinc-300">Loading dataset preview...</p>
      </div>
    );
  }

  if (!records || records.length === 0) {
    return (
      <div className="w-full bg-[#121217] border border-zinc-800/80 rounded-2xl p-12 flex flex-col items-center justify-center min-h-64 text-center">
        <Database className="w-10 h-10 text-zinc-600 mb-3" />
        <h3 className="text-base font-semibold text-white">No records found</h3>
        <p className="text-xs text-zinc-400 max-w-sm mt-1">
          Try expanding the date range or selecting &quot;All Stations&quot; in the filter panel above.
        </p>
      </div>
    );
  }

  const columns = Object.keys(records[0]);

  const formatColTitle = (key: string) => {
    return key
      .replace(/([A-Z])/g, " $1")
      .replace(/^./, (str) => str.toUpperCase())
      .replace(/MM/g, "(mm)")
      .replace(/DBZ/g, "(dBZ)")
      .replace(/Pct/g, "(%)")
      .replace(/Us/g, "(μs)")
      .replace(/Jkg/g, "(J/kg)")
      .trim();
  };

  const formatCellValue = (val: unknown) => {
    if (val === null || val === undefined) return <span className="text-zinc-600">--</span>;
    if (typeof val === "boolean") {
      return val ? (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-500/20 text-blue-300 border border-blue-400/30">
          TRUE
        </span>
      ) : (
        <span className="text-zinc-500 text-xs">FALSE</span>
      );
    }
    if (typeof val === "number") {
      return <span className="font-mono text-zinc-200">{val}</span>;
    }
    if (typeof val === "string" && (val === "CRITICAL" || val === "ALERT" || val === "NORMAL" || val === "VALID")) {
      const color =
        val === "CRITICAL"
          ? "bg-red-500/20 text-red-400 border-red-500/30"
          : val === "ALERT"
          ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
          : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
      return (
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${color}`}>
          {val}
        </span>
      );
    }
    return String(val);
  };

  return (
    <div className="w-full bg-[#121217] border border-zinc-800/80 rounded-2xl p-4 sm:p-6 shadow-2xl space-y-4">
      {/* Table Header Summary Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-zinc-800/80">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-orange-400" />
          <h3 className="text-sm font-semibold text-white">
            Dataset Live Preview ({stream.toUpperCase()})
          </h3>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
          <span>Showing first {Math.min(records.length, 25)} rows</span>
          <span>•</span>
          <span className="text-amber-400 font-bold">{totalCount} total rows available for download</span>
        </div>
      </div>

      {/* Scrollable Table View */}
      <div className="w-full overflow-x-auto rounded-xl border border-zinc-800/90 max-h-96 overflow-y-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="sticky top-0 bg-[#181820] text-orange-300/90 font-mono text-[11px] uppercase tracking-wider border-b border-zinc-800 z-10">
            <tr>
              {columns.map((col) => (
                <th key={col} className="py-3 px-3.5 font-semibold whitespace-nowrap">
                  {formatColTitle(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-850">
            {records.map((row, idx) => (
              <tr
                key={`preview-row-${idx}`}
                className="hover:bg-zinc-850/60 transition-colors"
              >
                {columns.map((col) => (
                  <td
                    key={`preview-cell-${idx}-${col}`}
                    className="py-2.5 px-3.5 text-zinc-300 whitespace-nowrap text-xs"
                  >
                    {formatCellValue(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
