"use client";

import { useTranslations } from "next-intl";
import type { PredictionHorizon } from "@/types/prediction";
import { cn } from "@/lib/utils/cn";

interface PredictionHorizonSelectorProps {
  selectedHorizon: PredictionHorizon;
  onSelectHorizon: (horizon: PredictionHorizon) => void;
}

const HORIZONS: Array<{ value: PredictionHorizon; label: string }> = [
  { value: "1h", label: "1h" },
  { value: "3h", label: "3h" },
  { value: "6h", label: "6h" },
  { value: "12h", label: "12h" },
  { value: "24h", label: "24h" },
  { value: "48h", label: "48h" },
  { value: "72h", label: "72h" },
];

export default function PredictionHorizonSelector({
  selectedHorizon,
  onSelectHorizon,
}: PredictionHorizonSelectorProps) {
  const t = useTranslations("prediction");

  return (
    <div className="flex items-center gap-1 overflow-x-auto scrollbar-none">
      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mr-1 shrink-0">
        {t("horizonLabel")}
      </span>
      <div className="flex items-center gap-1 p-0.5 rounded-full bg-slate-950/5">
        {HORIZONS.map((h) => {
          const isSelected = selectedHorizon === h.value;
          return (
            <button
              key={h.value}
              onClick={() => onSelectHorizon(h.value)}
              className={cn(
                "rounded-full px-2.5 py-1 text-[11px] font-semibold transition-all focus:outline-none focus:ring-1 focus:ring-sky-500",
                isSelected
                  ? "bg-slate-900 text-white shadow-xs"
                  : "text-slate-600 hover:text-slate-900 hover:bg-white/60"
              )}
            >
              {h.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
