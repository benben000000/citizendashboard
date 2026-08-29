"use client";

import { useLocale, useTranslations } from "next-intl";
import {
  Waves,
  Mountain,
  ShieldCheck,
  AlertCircle,
  ArrowUpRight,
  CloudLightning,
  CloudDrizzle,
  Sun,
  Timer,
  Radar,
  Radio,
} from "lucide-react";
import type {
  PredictionSummary,
  PredictionDataPoint,
  PredictionHorizon,
  FloodRiskLevel,
  SuddenRainBurstPrediction,
} from "@/types/prediction";
import type { StationPublicInfo } from "@/types/telemetry";
import PredictionHorizonSelector from "./prediction-horizon-selector";

interface PredictionPeakSummaryProps {
  summary: PredictionSummary;
  forecast: PredictionDataPoint[];
  station: StationPublicInfo;
  horizon?: PredictionHorizon;
  onSelectHorizon?: (horizon: PredictionHorizon) => void;
  suddenRainBurst?: SuddenRainBurstPrediction;
}

const TAGALOG_DAYS = ["Linggo", "Lunes", "Martes", "Miyerkules", "Huwebes", "Biyernes", "Sabado"];
const ENGLISH_DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export default function PredictionPeakSummary({
  summary,
  forecast,
  station,
  horizon = "24h",
  onSelectHorizon,
  suddenRainBurst,
}: PredictionPeakSummaryProps) {
  const locale = useLocale();
  const t = useTranslations("prediction.peakSection");

  const peakDate = new Date(summary.peakTime);
  const timeStr = peakDate.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  const dayIndex = peakDate.getDay();
  const dayName = locale === "fil" ? TAGALOG_DAYS[dayIndex] : ENGLISH_DAYS[dayIndex];
  const formattedPeakTime = `${dayName} ${timeStr}`;

  const thresholds = summary.thresholds;
  const peakLevel = summary.peakPredictedLevel;
  const rawCrit = thresholds.critical || 8.2;
  const rawWarn = thresholds.warning || 6.8;
  const rawAdv = thresholds.advisory || 5.0;

  const criticalThreshold = rawCrit > 20 ? rawCrit / 100 : rawCrit;
  const warningThreshold = rawWarn > 20 ? rawWarn / 100 : rawWarn;
  const advisoryThreshold = rawAdv > 20 ? rawAdv / 100 : rawAdv;

  const clearance = Math.abs(criticalThreshold - peakLevel);
  const isExceeded = peakLevel >= criticalThreshold;

  // Calculate percentage along the gauge (0 to critical * 1.15)
  const maxScale = criticalThreshold * 1.15;
  const peakPercent = Math.min(100, Math.max(0, (peakLevel / maxScale) * 100));

  // Risk stage color & badge styles
  const getRiskStyles = (level: FloodRiskLevel) => {
    switch (level) {
      case "critical":
        return {
          bg: "bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30",
          barColor: "#e11d48",
        };
      case "warning":
        return {
          bg: "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30",
          barColor: "#f97316",
        };
      case "advisory":
        return {
          bg: "bg-yellow-500/15 text-yellow-800 dark:text-yellow-300 border-yellow-500/30",
          barColor: "#eab308",
        };
      default:
        return {
          bg: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
          barColor: "#22c55e",
        };
    }
  };

  const riskStyles = getRiskStyles(summary.riskLevel);

  // Upstream mountain rainfall sum from horizon
  const maxRainMm = forecast.reduce(
    (max, p) => (p.rainfallAccumulationMm > max ? p.rainfallAccumulationMm : max),
    0
  );

  // Sudden burst resolution
  const burst = suddenRainBurst ?? summary.suddenRainBurst ?? {
    detected: false,
    burstType: "none" as const,
    title: t("burstTypes.none"),
    intensityMmHr: 0.0,
    probabilityPct: 15,
    expectedWindow: "Next 6 Hours",
    durationMinutes: 0,
    radarReflectivityDbz: 5.0,
    convectiveCloudCover: 12.0,
    advisory: t("burstLabels.stable"),
  };

  const isHeavyBurst = burst.burstType === "sudden_heavy" || burst.burstType === "short_burst_heavy";
  const isLightBurst = burst.burstType === "sudden_light" || burst.burstType === "short_burst_light";

  const burstBadgeStyles = isHeavyBurst
    ? "bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30"
    : isLightBurst
    ? "bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/30"
    : "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30";

  const burstIconTint = isHeavyBurst
    ? "bg-rose-500/10 text-rose-600 dark:text-rose-400"
    : isLightBurst
    ? "bg-sky-500/10 text-sky-600 dark:text-sky-400"
    : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400";

  return (
    <section
      id="prediction-details"
      className="w-full max-w-7xl mx-auto min-h-[90svh] flex flex-col justify-center px-5 py-10 sm:px-8 md:px-12"
    >
      {/* ── SECTION HEADER WITH SYNCHRONIZED LEAD HORIZON SELECTOR ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 md:mb-8">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-white/75">
              KloudTrack LNN Nowcast Intelligence
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl md:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight mt-1">
            {station.city || "Central Luzon"} — Continuous-Time Analytics
          </h2>
        </div>

        {/* Dynamic Horizon Selector on Screen 2 */}
        {onSelectHorizon && (
          <div className="flex items-center gap-2 rounded-full border border-white/40 dark:border-white/10 bg-white/75 dark:bg-white/10 px-3 py-1.5 shadow-2xs backdrop-blur-md self-start sm:self-auto">
            <span className="text-[11px] font-semibold text-slate-600 dark:text-white/70 uppercase tracking-wide pl-1 hidden md:inline">
              Horizon:
            </span>
            <PredictionHorizonSelector
              selectedHorizon={horizon}
              onSelectHorizon={onSelectHorizon}
            />
          </div>
        )}
      </div>

      {/* ── 3-CARD COMPREHENSIVE GRID ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-6 w-full">
        
        {/* ── CARD 1: PEAK WATER LEVEL & THRESHOLD CLEARANCE ── */}
        <div className="glass rounded-2xl p-5 md:p-6 flex flex-col justify-between border border-white/30 dark:border-white/10 shadow-xs hover:shadow-md transition-all">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2.5 rounded-xl bg-sky-500/10 text-sky-600 dark:text-sky-400">
                  <Waves className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white leading-tight">
                    {t("crestTitle")}
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-white/70 font-medium">
                    {t("crestSubtitle")}
                  </p>
                </div>
              </div>

              <div
                className={`px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider border shrink-0 ${riskStyles.bg}`}
              >
                {summary.riskLevel}
              </div>
            </div>

            {/* Peak Metric Large Display */}
            <div className="my-4 flex flex-wrap items-baseline gap-3">
              <div className="flex items-baseline gap-1 text-slate-900 dark:text-white">
                <span className="text-4xl sm:text-5xl font-black tracking-tight tabular-nums">
                  {peakLevel.toFixed(2)}
                </span>
                <span className="text-base font-semibold text-slate-600 dark:text-white/70">m</span>
              </div>

              <span className="text-xs font-semibold text-slate-700 dark:text-white/90 bg-white/50 dark:bg-white/10 border border-white/40 dark:border-white/10 px-3 py-1.5 rounded-lg shadow-2xs">
                {t("expectedAt", { time: formattedPeakTime })}
              </span>
            </div>

            {/* Minimalist Flood Threshold Bar */}
            <div className="space-y-2 my-4">
              <div className="flex justify-between items-center text-[11px] font-semibold text-slate-600 dark:text-white/75">
                <span>{t("thresholdLabels.normal")} &lt;{advisoryThreshold.toFixed(1)}m</span>
                <span>{t("thresholdLabels.critical")} {criticalThreshold.toFixed(1)}m</span>
              </div>

              {/* Clean Segmented Track */}
              <div className="relative h-2.5 w-full rounded-full bg-slate-200/80 dark:bg-white/10 overflow-hidden p-0.5 border border-white/20">
                <div
                  className="h-full rounded-full transition-all duration-500 shadow-xs"
                  style={{
                    width: `${peakPercent}%`,
                    backgroundColor: riskStyles.barColor,
                  }}
                />
              </div>
            </div>
          </div>

          {/* Clearance Notice */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 flex items-center gap-2 text-xs font-medium text-slate-700 dark:text-white/90">
            {isExceeded ? (
              <>
                <AlertCircle className="h-4 w-4 text-rose-500 shrink-0" />
                <span className="text-rose-600 dark:text-rose-400 font-semibold">
                  {t("clearanceExceeded", { clearance: clearance.toFixed(2) })}
                </span>
              </>
            ) : (
              <>
                <ShieldCheck className="h-4 w-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                <span>
                  {t("clearanceNormal", {
                    clearance: clearance.toFixed(2),
                    critical: criticalThreshold.toFixed(1),
                  })}
                </span>
              </>
            )}
          </div>
        </div>

        {/* ── CARD 2: SUDDEN RAIN & SHORT BURST DETECTION ── */}
        <div className="glass rounded-2xl p-5 md:p-6 flex flex-col justify-between border border-white/30 dark:border-white/10 shadow-xs hover:shadow-md transition-all">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2.5">
                <div className={`p-2.5 rounded-xl ${burstIconTint}`}>
                  {isHeavyBurst ? (
                    <CloudLightning className="h-5 w-5" />
                  ) : isLightBurst ? (
                    <CloudDrizzle className="h-5 w-5" />
                  ) : (
                    <Sun className="h-5 w-5" />
                  )}
                </div>
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white leading-tight">
                    {t("burstTitle")}
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-white/70 font-medium">
                    {t("burstSubtitle")}
                  </p>
                </div>
              </div>

              <div
                className={`px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider border shrink-0 ${burstBadgeStyles}`}
              >
                {burst.detected
                  ? t(`burstBadges.${burst.burstType}`)
                  : t("burstBadges.none")}
              </div>
            </div>

            {/* Peak Burst Intensity & Probability */}
            <div className="my-4 flex flex-wrap items-baseline gap-3">
              <div className="flex items-baseline gap-1 text-slate-900 dark:text-white">
                <span className="text-4xl sm:text-5xl font-black tracking-tight tabular-nums">
                  {burst.intensityMmHr.toFixed(1)}
                </span>
                <span className="text-base font-semibold text-slate-600 dark:text-white/70">mm/hr</span>
              </div>

              <div className="flex items-center gap-1.5 bg-white/50 dark:bg-white/10 border border-white/40 dark:border-white/10 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-700 dark:text-white/90 shadow-2xs">
                <Radio className="h-3 w-3 text-sky-500 animate-pulse" />
                <span>{burst.probabilityPct}% Confidence</span>
              </div>
            </div>

            {/* Clean Frosted Minimalist Telemetry Grid (No Text Truncation) */}
            <div className="grid grid-cols-2 gap-2.5 my-3">
              <div className="bg-white/40 dark:bg-white/5 rounded-xl p-2.5 sm:p-3 border border-white/50 dark:border-white/10 shadow-2xs">
                <span className="text-[10px] font-semibold text-slate-500 dark:text-white/60 uppercase tracking-wider block mb-1">
                  {t("burstLabels.onset")}
                </span>
                <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800 dark:text-white">
                  <Timer className="h-3.5 w-3.5 text-sky-500 shrink-0" />
                  <span className="leading-tight">{burst.expectedWindow}</span>
                </div>
              </div>

              <div className="bg-white/40 dark:bg-white/5 rounded-xl p-2.5 sm:p-3 border border-white/50 dark:border-white/10 shadow-2xs">
                <span className="text-[10px] font-semibold text-slate-500 dark:text-white/60 uppercase tracking-wider block mb-1">
                  {t("burstLabels.duration")}
                </span>
                <span className="text-xs font-bold text-slate-800 dark:text-white leading-tight block">
                  {burst.durationMinutes > 0
                    ? t("burstLabels.minutes", { min: burst.durationMinutes })
                    : "—"}
                </span>
              </div>

              <div className="bg-white/40 dark:bg-white/5 rounded-xl p-2.5 sm:p-3 border border-white/50 dark:border-white/10 shadow-2xs">
                <span className="text-[10px] font-semibold text-slate-500 dark:text-white/60 uppercase tracking-wider block mb-1">
                  {t("burstLabels.radar")}
                </span>
                <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800 dark:text-white">
                  <Radar className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                  <span>{burst.radarReflectivityDbz.toFixed(1)} dBZ</span>
                </div>
              </div>

              <div className="bg-white/40 dark:bg-white/5 rounded-xl p-2.5 sm:p-3 border border-white/50 dark:border-white/10 shadow-2xs">
                <span className="text-[10px] font-semibold text-slate-500 dark:text-white/60 uppercase tracking-wider block mb-1">
                  {t("burstLabels.cloud")}
                </span>
                <span className="text-xs font-bold text-slate-800 dark:text-white leading-tight block">
                  {burst.convectiveCloudCover.toFixed(0)}% (Himawari-9)
                </span>
              </div>
            </div>
          </div>

          {/* Actionable Protective Advisory */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-medium text-slate-700 dark:text-white/85 leading-relaxed flex items-start gap-2">
            <span className="h-2 w-2 rounded-full bg-sky-500 mt-1 shrink-0" />
            <span>{burst.advisory}</span>
          </div>
        </div>

        {/* ── CARD 3: UPSTREAM WATERSHED & INFLOW DYNAMICS ── */}
        <div className="glass rounded-2xl p-5 md:p-6 flex flex-col justify-between border border-white/30 dark:border-white/10 shadow-xs hover:shadow-md transition-all">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2.5 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                  <Mountain className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white leading-tight">
                    {t("watershedTitle")}
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-white/70 font-medium">
                    {t("watershedSubtitle")}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1 text-[11px] font-semibold text-slate-600 dark:text-white/75 bg-white/40 dark:bg-white/10 px-2.5 py-1 rounded-full border border-white/30 shrink-0">
                <span>{station.city || "Central Luzon"} Basin</span>
                <ArrowUpRight className="h-3 w-3" />
              </div>
            </div>

            {/* Clean Frosted Watershed Status Grid */}
            <div className="grid grid-cols-2 gap-3 my-4">
              <div className="bg-white/40 dark:bg-white/5 rounded-xl p-3 sm:p-3.5 border border-white/50 dark:border-white/10 shadow-2xs">
                <span className="text-[10px] font-semibold text-slate-500 dark:text-white/60 uppercase tracking-wider block mb-1">
                  {t("inflowStatus")}
                </span>
                <span className="text-sm sm:text-base font-bold text-slate-800 dark:text-white leading-snug block">
                  {maxRainMm > 15
                    ? t("inflowCritical")
                    : maxRainMm > 5
                      ? t("inflowElevated")
                      : t("inflowNormal")}
                </span>
              </div>

              <div className="bg-white/40 dark:bg-white/5 rounded-xl p-3 sm:p-3.5 border border-white/50 dark:border-white/10 shadow-2xs">
                <span className="text-[10px] font-semibold text-slate-500 dark:text-white/60 uppercase tracking-wider block mb-1">
                  {t("mountainRain")}
                </span>
                <div className="flex items-baseline gap-1 text-slate-800 dark:text-white">
                  <span className="text-base sm:text-lg font-black tabular-nums">
                    {maxRainMm.toFixed(1)}
                  </span>
                  <span className="text-xs text-slate-500 dark:text-white/60 font-medium">mm/hr</span>
                </div>
              </div>
            </div>
          </div>

          {/* Hydrological Physical Runoff Summary */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-normal text-slate-700 dark:text-white/85 leading-relaxed">
            {maxRainMm > 5 ? t("runoffDescriptionElevated") : t("runoffDescriptionNormal")}
          </div>
        </div>

      </div>
    </section>
  );
}
