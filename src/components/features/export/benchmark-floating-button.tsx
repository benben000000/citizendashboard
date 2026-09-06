"use client";

import React, { useState } from "react";
import { Download, Database } from "lucide-react";
import BenchmarkExportModal from "./benchmark-export-modal";

export default function BenchmarkFloatingButton() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      {/* Floating Bottom-Right Action Button */}
      <aside
        aria-label="Download Data"
        className="fixed bottom-5 right-5 z-40 flex items-center group pointer-events-auto select-none"
      >
        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-slate-900/85 hover:bg-slate-900 text-white dark:bg-white/90 dark:hover:bg-white dark:text-slate-900 border border-white/20 dark:border-slate-800 shadow-xl shadow-black/25 backdrop-blur-xl transition-all duration-300 hover:scale-105 active:scale-95 cursor-pointer"
          title="Download Comparative Logs (Raw MQTT, Processed, 1h-72h Predictions)"
          aria-label="Download Benchmark Data"
        >
          <div className="relative flex items-center justify-center">
            <Database className="h-4 w-4 text-emerald-400 dark:text-emerald-600 transition group-hover:scale-110" />
            <span className="absolute -top-1 -right-1 flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
          </div>
          <span className="text-xs font-extrabold tracking-wide uppercase">
            Download Data
          </span>
          <Download className="h-3.5 w-3.5 opacity-70 group-hover:opacity-100 group-hover:translate-y-0.5 transition-all" />
        </button>
      </aside>

      {/* Interactive Calendar & Export Modal */}
      <BenchmarkExportModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </>
  );
}
