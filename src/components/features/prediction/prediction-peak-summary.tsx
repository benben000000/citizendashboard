"use client";

import { useLocale, useTranslations } from "next-intl";
import {
  Waves,
  Mountain,
  ShieldCheck,
  ShieldAlert,
  AlertCircle,
  ArrowUpRight,
  CloudLightning,
  CloudDrizzle,
  Sun,
  Timer,
  Radar,
  Radio,
  Umbrella,
  Car,
} from "lucide-react";
import type {
  PredictionSummary,
  PredictionDataPoint,
  PredictionHorizon,
  FloodRiskLevel,
  SuddenBurstType,
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

  const thresholds = summary.thresholds;
  const rawCrit = thresholds.critical || 8.2;
  const rawWarn = thresholds.warning || 6.8;
  const rawAdv = thresholds.advisory || 5.0;

  const criticalThreshold = rawCrit > 20 ? rawCrit / 100 : rawCrit;
  const warningThreshold = rawWarn > 20 ? rawWarn / 100 : rawWarn;
  const advisoryThreshold = rawAdv > 20 ? rawAdv / 100 : rawAdv;

  // ── REAL-TIME HORIZON-ADAPTIVE FORECAST PARSER ──
  const now = new Date();
  const forwardPoints = forecast.filter((p) => p.isForecast);
  const rainPoints = forwardPoints.filter((p) => p.rainfallAccumulationMm > 0.05);
  const peakRainPoint = forwardPoints.reduce(
    (max, p) => (p.rainfallAccumulationMm > (max?.rainfallAccumulationMm ?? 0) ? p : max),
    null as PredictionDataPoint | null
  );

  const peakForecastPoint = forwardPoints.reduce(
    (max, p) => (p.predictedWaterLevel > (max?.predictedWaterLevel ?? 0) ? p : max),
    null as PredictionDataPoint | null
  );

  // Peak time adaptively synchronized to the selected horizon
  const activePeakDate = peakForecastPoint ? new Date(peakForecastPoint.timestamp) : new Date(summary.peakTime);
  const timeStr = activePeakDate.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  const dayIndex = activePeakDate.getDay();
  const dayName = locale === "fil" ? TAGALOG_DAYS[dayIndex] : ENGLISH_DAYS[dayIndex];
  const formattedPeakTime = `${dayName} ${timeStr}`;

  const peakLevel = peakForecastPoint ? peakForecastPoint.predictedWaterLevel : summary.peakPredictedLevel;
  const peakPercent = Math.min(100, Math.max(0, (peakLevel / (criticalThreshold * 1.15)) * 100));

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

  const maxRainMm = forecast.reduce(
    (max, p) => (p.rainfallAccumulationMm > max ? p.rainfallAccumulationMm : max),
    0
  );

  const fallbackBurst = suddenRainBurst ?? summary.suddenRainBurst ?? {
    detected: false,
    burstType: "none" as const,
    title: t("burstTypes.none"),
    intensityMmHr: 0.0,
    probabilityPct: 15,
    expectedWindow: `Next ${horizon.toUpperCase()} (Dry)`,
    durationMinutes: 0,
    radarReflectivityDbz: 5.0,
    convectiveCloudCover: 12.0,
    advisory: t("burstLabels.stable"),
  };

  // ── REAL-TIME ADAPTIVE RAIN ONSET & DURATION ACROSS SELECTED HORIZON ──
  let adaptiveBurstDetected = false;
  let adaptiveBurstType: SuddenBurstType = "none";
  let adaptiveWindow = horizon === "1h" ? "Next 1 Hour (Dry)" : horizon === "72h" ? "Next 3 Days (Dry)" : `Next ${horizon.toUpperCase()} (Dry)`;
  let adaptiveDurationStr = "0 mins (Dry)";
  let adaptiveIntensityMmHr = 0.0;

  if (rainPoints.length > 0 && peakRainPoint && peakRainPoint.rainfallAccumulationMm > 0.05) {
    adaptiveBurstDetected = true;
    adaptiveIntensityMmHr = peakRainPoint.rainfallAccumulationMm;
    adaptiveBurstType =
      adaptiveIntensityMmHr >= 5.0
        ? "sudden_heavy"
        : adaptiveIntensityMmHr >= 1.5
        ? "sudden_light"
        : "short_burst_light";

    const firstRainPoint = rainPoints[0];
    const firstRainDate = new Date(firstRainPoint.timestamp);
    const diffMins = Math.max(0, Math.round((firstRainDate.getTime() - now.getTime()) / (60 * 1000)));
    const diffHours = diffMins / 60;

    const timeFormatted = firstRainDate.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });

    if (diffMins <= 30) {
      adaptiveWindow = `In ~30m (${timeFormatted})`;
    } else if (diffMins <= 75) {
      adaptiveWindow = `In +1h (${timeFormatted})`;
    } else {
      const displayHours = diffHours >= 10 ? Math.round(diffHours) : (diffHours % 1 === 0 ? diffHours : diffHours.toFixed(1));
      adaptiveWindow = `In +${displayHours}h (${timeFormatted})`;
    }

    const durationMinutes = Math.min(
      180,
      Math.max(15, rainPoints.length * (horizon === "1h" || horizon === "3h" || horizon === "6h" ? 30 : 60))
    );
    adaptiveDurationStr = durationMinutes >= 60 ? `~${Math.round(durationMinutes / 60)} hrs` : `${durationMinutes} mins`;
  } else if (fallbackBurst.detected && fallbackBurst.burstType !== "none") {
    adaptiveBurstDetected = true;
    adaptiveBurstType = fallbackBurst.burstType;
    adaptiveWindow = fallbackBurst.expectedWindow;
    adaptiveDurationStr = fallbackBurst.durationMinutes > 0 ? `${fallbackBurst.durationMinutes} mins` : "15-20 mins";
    adaptiveIntensityMmHr = fallbackBurst.intensityMmHr;
  }

  const isHeavyBurst = adaptiveBurstType === "sudden_heavy" || adaptiveBurstType === "short_burst_heavy" || adaptiveIntensityMmHr >= 5.0;
  const isLightBurst = adaptiveBurstType === "sudden_light" || adaptiveBurstType === "short_burst_light" || adaptiveIntensityMmHr > 0.1;
  const burstBadgeColor = isHeavyBurst ? "#e11d48" : isLightBurst ? "#0284c7" : "#16a34a";

  const passabilityInfo = (() => {
    if (summary.riskLevel === "critical" || summary.riskLevel === "warning" || peakLevel >= warningThreshold) {
      return {
        badge: t("passableDanger"),
        bgColor: "#e11d48",
        textColor: "#ffffff",
        description: t("passableDangerDesc"),
        icon: ShieldAlert,
      };
    }
    if (summary.riskLevel === "advisory" || peakLevel >= advisoryThreshold) {
      return {
        badge: t("passableCaution"),
        bgColor: "#eab308",
        textColor: "#1e293b",
        description: t("passableCautionDesc"),
        icon: AlertCircle,
      };
    }
    return {
      badge: t("passableSafe"),
      bgColor: "#16a34a",
      textColor: "#ffffff",
      description: t("passableSafeDesc"),
      icon: ShieldCheck,
    };
  })();

  const cleanTimeStr = adaptiveWindow.replace(/^(Expected in|In)\s+/i, "");
  const chipTimeStr = adaptiveWindow;

  const umbrellaInfo = (() => {
    if (isHeavyBurst) {
      return {
        badge: t("umbrellaHeavy"),
        bgColor: "#e11d48",
        textColor: "#ffffff",
        description: t("umbrellaHeavyDesc", { time: cleanTimeStr }),
        icon: Umbrella,
      };
    }
    if (isLightBurst) {
      return {
        badge: t("umbrellaLight"),
        bgColor: "#0284c7",
        textColor: "#ffffff",
        description: t("umbrellaLightDesc", {
          time: cleanTimeStr,
          duration: adaptiveDurationStr,
        }),
        icon: Umbrella,
      };
    }
    return {
      badge: t("umbrellaNoRain"),
      bgColor: "#16a34a",
      textColor: "#ffffff",
      description: t("umbrellaNoRainDesc"),
      icon: Sun,
    };
  })();

  const mountainInfo = (() => {
    if (maxRainMm > 15) {
      return {
        badge: t("mountainDanger"),
        bgColor: "#e11d48",
        textColor: "#ffffff",
        description: t("mountainDangerDesc"),
      };
    }
    if (maxRainMm > 5) {
      return {
        badge: t("mountainCaution"),
        bgColor: "#eab308",
        textColor: "#1e293b",
        description: t("mountainCautionDesc"),
      };
    }
    return {
      badge: t("mountainSafe"),
      bgColor: "#16a34a",
      textColor: "#ffffff",
      description: t("mountainSafeDesc"),
    };
  })();

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
            <span className="text-xs font-bold uppercase tracking-wider text-light/80">
              KloudTrack Instant Citizen Action Intelligence
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-light tracking-tight mt-1">
            {station.city || "Central Luzon"} — Real-Time Citizen Decision Guide
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

      {/* ── 3-CARD COMPREHENSIVE CITIZEN ACTION GRID ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-6 w-full">
        
        {/* ── CARD 1: ROAD & FLOOD PASSABILITY ("Ligtas ba ang Daan o May Baha?") ── */}
        <div className="glass flex flex-col justify-between p-5 md:p-6 transition hover:scale-[1.01]">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Car className="h-5 w-5 text-light shrink-0" />
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-light leading-tight">
                    {t("crestTitle")}
                  </h3>
                  <p className="text-xs text-light/75 font-medium">
                    {t("crestSubtitle")}
                  </p>
                </div>
              </div>
            </div>

            {/* Solid Color Action Badge */}
            <div
              className="rounded-xl p-3 my-3 shadow-md flex items-center gap-2.5 transition"
              style={{ backgroundColor: passabilityInfo.bgColor, color: passabilityInfo.textColor }}
            >
              <passabilityInfo.icon className="h-5 w-5 shrink-0" style={{ color: passabilityInfo.textColor }} />
              <span className="text-xs sm:text-sm font-extrabold uppercase tracking-wide" style={{ color: passabilityInfo.textColor }}>
                {passabilityInfo.badge}
              </span>
            </div>

            {/* Peak Metric Large Display & Expected Peak Time */}
            <div className="my-3 flex flex-wrap items-baseline justify-between gap-2">
              <div className="flex items-baseline gap-1 text-light">
                <span className="text-3xl sm:text-4xl font-black tracking-tight tabular-nums">
                  {peakLevel.toFixed(2)}
                </span>
                <span className="text-sm font-semibold text-light/80">m water stage</span>
              </div>

              <span className="text-xs font-semibold text-light/90 bg-white/30 dark:bg-white/10 px-2.5 py-1 rounded-lg border border-white/20">
                {t("expectedAt", { time: formattedPeakTime })}
              </span>
            </div>

            {/* Human Flood Depth Gauge Bar */}
            <div className="space-y-1.5 my-3">
              <div className="flex justify-between items-center text-[10px] font-semibold text-light/80">
                <span>{t("thresholdLabels.normal", { val: advisoryThreshold.toFixed(1) })}</span>
                <span>{t("thresholdLabels.advisory")}</span>
                <span>{t("thresholdLabels.critical", { val: criticalThreshold.toFixed(1) })}</span>
              </div>

              {/* Clean Adaptive Track */}
              <div className="relative h-2.5 w-full rounded-full bg-slate-900/15 dark:bg-white/15 overflow-hidden p-0.5 border border-white/20">
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

          {/* Citizen Road Practical Advice */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-medium text-light/95 leading-relaxed">
            {passabilityInfo.description}
          </div>
        </div>

        {/* ── CARD 2: RAIN & UMBRELLA GUIDE ("Bubuhos ba ang Ulan? Payong & Kapote") ── */}
        <div className="glass flex flex-col justify-between p-5 md:p-6 transition hover:scale-[1.01]">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Umbrella className="h-5 w-5 text-light shrink-0" />
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-light leading-tight">
                    {t("umbrellaTitle")}
                  </h3>
                  <p className="text-xs text-light/75 font-medium">
                    {t("umbrellaSubtitle")}
                  </p>
                </div>
              </div>

              <div
                className="px-2.5 py-1 rounded-full text-[10px] sm:text-[11px] font-bold text-white uppercase tracking-wider shrink-0 shadow-xs"
                style={{ backgroundColor: burstBadgeColor }}
              >
                {adaptiveBurstDetected
                  ? t(`burstBadges.${adaptiveBurstType}`)
                  : t("burstBadges.none")}
              </div>
            </div>

            {/* Solid Color Action Badge */}
            <div
              className="rounded-xl p-3 my-3 shadow-md flex items-center gap-2.5 transition"
              style={{ backgroundColor: umbrellaInfo.bgColor, color: umbrellaInfo.textColor }}
            >
              <umbrellaInfo.icon className="h-5 w-5 shrink-0" style={{ color: umbrellaInfo.textColor }} />
              <span className="text-xs sm:text-sm font-extrabold uppercase tracking-wide" style={{ color: umbrellaInfo.textColor }}>
                {umbrellaInfo.badge}
              </span>
            </div>

            {/* 2 Clean Citizen Decision Boxes (When & How Long) */}
            <div className="grid grid-cols-2 gap-2.5 my-3">
              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-bold text-light/75 uppercase tracking-wider block mb-1">
                  {t("burstLabels.onset")}
                </span>
                <div className="flex items-center gap-1.5 text-xs sm:text-sm font-bold text-light">
                  <Timer className="h-3.5 w-3.5 text-light shrink-0" />
                  <span className="leading-tight">{chipTimeStr}</span>
                </div>
              </div>

              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-bold text-light/75 uppercase tracking-wider block mb-1">
                  {t("burstLabels.duration")}
                </span>
                <span className="text-xs sm:text-sm font-bold text-light leading-tight block">
                  {adaptiveDurationStr}
                </span>
              </div>
            </div>
          </div>

          {/* Actionable Protective Advisory */}
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-medium text-light/95 leading-relaxed">
            {umbrellaInfo.description}
          </div>
        </div>

        {/* ── CARD 3: MOUNTAIN FLASH FLOOD ALERT ("Baha Mula sa Bundok") ── */}
        <div className="glass flex flex-col justify-between p-5 md:p-6 transition hover:scale-[1.01]">
          <div>
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <Mountain className="h-5 w-5 text-light shrink-0" />
                <div>
                  <h3 className="text-base sm:text-lg font-bold text-light leading-tight">
                    {t("mountainTitle")}
                  </h3>
                  <p className="text-xs text-light/75 font-medium">
                    {t("mountainSubtitle")}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1 text-[11px] font-semibold text-light/85 bg-white/30 dark:bg-white/10 px-2.5 py-1 rounded-full border border-white/20 shrink-0">
                <span>{station.city || "Central Luzon"} Basin</span>
                <ArrowUpRight className="h-3 w-3 text-light" />
              </div>
            </div>

            {/* Solid Color Action Badge */}
            <div
              className="rounded-xl p-3 my-3 shadow-md flex items-center gap-2.5 transition"
              style={{ backgroundColor: mountainInfo.bgColor, color: mountainInfo.textColor }}
            >
              <Mountain className="h-5 w-5 shrink-0" style={{ color: mountainInfo.textColor }} />
              <span className="text-xs sm:text-sm font-extrabold uppercase tracking-wide" style={{ color: mountainInfo.textColor }}>
                {mountainInfo.badge}
              </span>
            </div>

            {/* Clean Frosted Watershed Status Grid */}
            <div className="grid grid-cols-2 gap-2.5 my-3">
              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-bold text-light/75 uppercase tracking-wider block mb-1">
                  {t("inflowStatus")}
                </span>
                <span className="text-xs sm:text-sm font-bold text-light leading-snug block">
                  {maxRainMm > 15
                    ? t("inflowCritical")
                    : maxRainMm > 5
                      ? t("inflowElevated")
                      : t("inflowNormal")}
                </span>
              </div>

              <div className="bg-white/35 dark:bg-white/10 rounded-xl p-2.5 sm:p-3 border border-white/30 dark:border-white/10">
                <span className="text-[10px] font-bold text-light/75 uppercase tracking-wider block mb-1">
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
          <div className="pt-3.5 border-t border-slate-950/10 dark:border-white/10 text-xs font-medium text-light/95 leading-relaxed">
            {mountainInfo.description}
          </div>
        </div>

      </div>
    </section>
  );
}
