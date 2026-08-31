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
  horizon = "1h",
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

  // Risk stage color
  const getRiskColor = (level: FloodRiskLevel) => {
    switch (level) {
      case "critical":
        return "#e11d48";
      case "warning":
        return "#f97316";
      case "advisory":
        return "#eab308";
      default:
        return "#16a34a";
    }
  };

  const riskColor = getRiskColor(summary.riskLevel);

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
  const burstBadgeColor = isHeavyBurst ? "#e11d48" : isLightBurst ? "#0284c7" : "#16a34a";

  return (
    <section
      id="prediction-details"
      className="w-full max-w-7xl mx-auto min-h-[90svh] flex flex-col justify-center px-5 py-10 sm:px-8 md:px-12"
    >
      {/* ── SECTION HEADER WITH SYNCHRONIZED LEAD HORIZON SELECTOR ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 md:mb-8">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500" />
            <span className="text-xs font-bold uppercase tracking-wider text-light/80">
              KloudTrack LNN Nowcast Intelligence
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-light tracking-tight mt-1">
            {station.city || "Central Luzon"} — Continuous-Time Analytics
          </h2>
        </div>

        {/* Dynamic Horizon Selector on Screen 2 */}
        {onSelectHorizon && (
          <div className="flex items-center rounded-full border border-slate-950/10 bg-white/70 px-3.5 py-1.5 shadow-2xs backdrop-blur-md self-start sm:self-auto">
            <PredictionHorizonSelector
              selectedHorizon={horizon}
              onSelectHorizon={onSelectHorizon}
            />
          </div>
        )}
      </div>

      {/* ── 3-CARD COMPREHENSIVE GRID (Matching Pure Glassmorphic Theme) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-6 w-full">
        
        {/* ── CARD 1: PEAK WATER LEVEL & THRESHOLD CLEARANCE ── */}
        <div className="glass flex flex-col justify-between p-5 md:p-6">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Waves className="h-5 w-5 text-light shrink-0" />
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-light leading-tight">
                    {t("crestTitle")}
                  </h3>
                  <p className="text-xs text-light/75 font-medium">
                    {t("crestSubtitle")}
                  </p>
                </div>
              </div>

              <div
                className="px-3 py-1 rounded-full text-[11px] font-bold text-white uppercase tracking-wider shrink-0 shadow-xs"
                style={{ backgroundColor: riskColor }}
              >
                {summary.riskLevel}
              </div>
            </div>

            {/* Peak Metric Large Display */}
            <div className="my-4 flex flex-wrap items-baseline gap-3">
              <div className="flex items-baseline gap-1 text-light">
                <span className="text-4xl sm:text-5xl font-black tracking-tight tabular-nums">
                  {peakLevel.toFixed(2)}
                </span>
                <span className="text-base font-semibold text-light/80">m</span>
              </div>

              <span className="text-xs font-semibold text-light/90 bg-white/30 dark:bg-white/10 px-2.5 py-1 rounded-lg border border-white/20">
                {t("expectedAt", { time: formattedPeakTime })}
              </span>
            </div>

            {/* Minimalist Flood Threshold Bar */}
            <div className="space-y-2 my-4">
              <div className="flex justify-between items-center text-[11px] font-semibold text-light/80">
                <span>{t("thresholdLabels.normal")} &lt;{advisoryThreshold.toFixed(1)}m</span>
                <span>{t("thresholdLabels.critical")} {criticalThreshold.toFixed(1)}m</span>
              </div>

              {/* Clean Adaptive Track */}
              <div className="relative h-2 w-full rounded-full bg-slate-900/10 dark:bg-white/15 overflow-hidden p-0.5 border border-white/20">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${peakPercent}%`,
                    backgroundColor: riskColor,
                  }}
                />
              </div>
            </div>
          </div>

          {/* Clearance Notice */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 flex items-center gap-2 text-xs font-medium text-light/90">
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
        <div className="glass flex flex-col justify-between p-5 md:p-6">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                {isHeavyBurst ? (
                  <CloudLightning className="h-5 w-5 text-light shrink-0" />
                ) : isLightBurst ? (
                  <CloudDrizzle className="h-5 w-5 text-light shrink-0" />
                ) : (
                  <Sun className="h-5 w-5 text-light shrink-0" />
                )}
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-light leading-tight">
                    {t("burstTitle")}
                  </h3>
                  <p className="text-xs text-light/75 font-medium">
                    {t("burstSubtitle")}
                  </p>
                </div>
              </div>

              <div
                className="px-3 py-1 rounded-full text-[11px] font-bold text-white uppercase tracking-wider shrink-0 shadow-xs"
                style={{ backgroundColor: burstBadgeColor }}
              >
                {burst.detected
                  ? t(`burstBadges.${burst.burstType}`)
                  : t("burstBadges.none")}
              </div>
            </div>

            {/* Peak Burst Intensity & Probability */}
            <div className="my-4 flex flex-wrap items-baseline gap-3">
              <div className="flex items-baseline gap-1 text-light">
                <span className="text-4xl sm:text-5xl font-black tracking-tight tabular-nums">
                  {burst.intensityMmHr.toFixed(1)}
                </span>
                <span className="text-base font-semibold text-light/80">mm/hr</span>
              </div>

              <div className="flex items-center gap-1.5 bg-white/30 dark:bg-white/10 border border-white/20 px-2.5 py-1 rounded-lg text-xs font-semibold text-light/90">
                <Radio className="h-3 w-3 text-light shrink-0" />
                <span>{burst.probabilityPct}% Confidence</span>
              </div>
            </div>

            {/* Clean Adaptive Frosted Glass Sub-Cards (No Text Truncation) */}
            <div className="grid grid-cols-2 gap-2.5 my-3">
              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-semibold text-light/75 uppercase tracking-wider block mb-1">
                  {t("burstLabels.onset")}
                </span>
                <div className="flex items-center gap-1.5 text-xs font-bold text-light">
                  <Timer className="h-3.5 w-3.5 text-light shrink-0" />
                  <span className="leading-tight">{burst.expectedWindow}</span>
                </div>
              </div>

              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-semibold text-light/75 uppercase tracking-wider block mb-1">
                  {t("burstLabels.duration")}
                </span>
                <span className="text-xs font-bold text-light leading-tight block">
                  {burst.durationMinutes > 0
                    ? t("burstLabels.minutes", { min: burst.durationMinutes })
                    : "—"}
                </span>
              </div>

              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-semibold text-light/75 uppercase tracking-wider block mb-1">
                  {t("burstLabels.radar")}
                </span>
                <div className="flex items-center gap-1.5 text-xs font-bold text-light">
                  <Radar className="h-3.5 w-3.5 text-light shrink-0" />
                  <span>{burst.radarReflectivityDbz.toFixed(1)} dBZ</span>
                </div>
              </div>

              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-semibold text-light/75 uppercase tracking-wider block mb-1">
                  {t("burstLabels.cloud")}
                </span>
                <span className="text-xs font-bold text-light leading-tight block">
                  {burst.convectiveCloudCover.toFixed(0)}% (Himawari-9)
                </span>
              </div>
            </div>
          </div>

          {/* Actionable Protective Advisory */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-medium text-light/90 leading-relaxed flex items-start gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-light mt-1.5 shrink-0" />
            <span>{burst.advisory}</span>
          </div>
        </div>

        {/* ── CARD 3: UPSTREAM WATERSHED & INFLOW DYNAMICS ── */}
        <div className="glass flex flex-col justify-between p-5 md:p-6">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Mountain className="h-5 w-5 text-light shrink-0" />
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-light leading-tight">
                    {t("watershedTitle")}
                  </h3>
                  <p className="text-xs text-light/75 font-medium">
                    {t("watershedSubtitle")}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1 text-[11px] font-semibold text-light/85 bg-white/30 dark:bg-white/10 px-2.5 py-1 rounded-full border border-white/20 shrink-0">
                <span>{station.city || "Central Luzon"} Basin</span>
                <ArrowUpRight className="h-3 w-3 text-light" />
              </div>
            </div>

            {/* Clean Frosted Watershed Status Grid */}
            <div className="grid grid-cols-2 gap-3 my-4">
              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-3 sm:p-3.5 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-semibold text-light/75 uppercase tracking-wider block mb-1">
                  {t("inflowStatus")}
                </span>
                <span className="text-sm sm:text-base font-bold text-light leading-snug block">
                  {maxRainMm > 15
                    ? t("inflowCritical")
                    : maxRainMm > 5
                      ? t("inflowElevated")
                      : t("inflowNormal")}
                </span>
              </div>

              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-3 sm:p-3.5 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-semibold text-light/75 uppercase tracking-wider block mb-1">
                  {t("mountainRain")}
                </span>
                <div className="flex items-baseline gap-1 text-light">
                  <span className="text-base sm:text-lg font-black tabular-nums">
                    {maxRainMm.toFixed(1)}
                  </span>
                  <span className="text-xs text-light/75 font-semibold">mm/hr</span>
                </div>
              </div>
            </div>
          </div>

          {/* Hydrological Physical Runoff Summary */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-normal text-light/90 leading-relaxed">
            {maxRainMm > 5 ? t("runoffDescriptionElevated") : t("runoffDescriptionNormal")}
          </div>
        </div>

      </div>
    </section>
  );
}
