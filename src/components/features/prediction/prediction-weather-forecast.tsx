"use client";

import { useMemo } from "react";
import { useLocale, useTranslations } from "next-intl";
import WeatherIcon from "@/components/shared/weather-icons";
import { StationSelectorPopover } from "@/components/shared/station-selector-popover";
import type { StationPublicInfo } from "@/types/telemetry";
import type {
  PredictionWeatherOverview,
  PredictionDataPoint,
  PredictionHorizon,
  FloodRiskLevel,
} from "@/types/prediction";
import {
  TriangleAlert,
  Flame,
  Wind,
  CloudRain,
  Waves,
} from "lucide-react";

interface PredictionWeatherForecastProps {
  station: StationPublicInfo;
  stations: StationPublicInfo[];
  weather: PredictionWeatherOverview;
  forecastPoints: PredictionDataPoint[];
  horizon: PredictionHorizon;
  thresholds: {
    advisory: number;
    warning: number;
    critical: number;
  };
  onStationSelect?: (stationId: string) => void;
  nearestStationId?: string | null;
  onDetectNearest?: () => void;
  isLocating?: boolean;
}

const TAGALOG_DAYS = ["Linggo", "Lunes", "Martes", "Miyerkules", "Huwebes", "Biyernes", "Sabado"];
const ENGLISH_DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export default function PredictionWeatherForecast({
  station,
  stations,
  weather,
  forecastPoints,
  horizon,
  thresholds,
  onStationSelect,
  nearestStationId,
  onDetectNearest,
  isLocating,
}: PredictionWeatherForecastProps) {
  const locale = useLocale();
  const t = useTranslations("prediction");

  // Extract target forecast data for the specific chosen lead horizon
  const targetData = useMemo(() => {
    const forecastOnly = forecastPoints.filter((p) => p.isForecast);
    const targetPoint = forecastOnly.length > 0 ? forecastOnly[forecastOnly.length - 1] : null;

    const now = new Date();
    const targetDate = targetPoint
      ? new Date(targetPoint.timestamp)
      : new Date(now.getTime() + 24 * 60 * 60 * 1000);

    const timeStr = targetDate.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });

    const dayIndex = targetDate.getDay();
    const weekdayStr = locale === "fil" ? TAGALOG_DAYS[dayIndex] : ENGLISH_DAYS[dayIndex];

    // Find the hourly forecast point closest to targetDate
    let hourlyMatch: typeof weather.hourly[0] | null = null;
    if (weather.hourly && weather.hourly.length > 0) {
      const targetTimeMs = targetDate.getTime();
      let minDiff = Number.POSITIVE_INFINITY;
      for (const h of weather.hourly) {
        const diff = Math.abs(new Date(h.timestamp).getTime() - targetTimeMs);
        if (diff < minDiff) {
          minDiff = diff;
          hourlyMatch = h;
        }
      }
    }

    const temp = hourlyMatch ? hourlyMatch.temp : weather.currentTemp;
    const heatIndex = hourlyMatch ? hourlyMatch.heatIndex : weather.currentHeatIndex;
    const rainProbability = hourlyMatch
      ? hourlyMatch.rainProbability
      : targetPoint
        ? Math.round(targetPoint.rainfallAccumulationMm > 0 ? 75 : 15)
        : 20;
    const precipitationMm = targetPoint
      ? targetPoint.rainfallAccumulationMm
      : hourlyMatch?.precipitationMm ?? 0.0;
    const windSpeedKmH = hourlyMatch ? hourlyMatch.windSpeedKmH : weather.windSpeed;
    const windDirection = hourlyMatch?.windDirection ?? weather.windDirection ?? "NE";
    const humidity = hourlyMatch ? hourlyMatch.humidity : weather.humidity;
    const pressure = hourlyMatch?.pressure ?? weather.pressure ?? 1008;

    const condition =
      hourlyMatch?.condition ||
      (precipitationMm > 5 ? "storm" : precipitationMm > 0.5 ? "rain" : "partly-cloudy");

    const predictedWaterLevel = targetPoint ? targetPoint.predictedWaterLevel : 3.45;

    let riskStatus: FloodRiskLevel = "normal";
    if (predictedWaterLevel >= thresholds.critical || precipitationMm >= 10.0 || rainProbability >= 85) {
      riskStatus = "critical";
    } else if (predictedWaterLevel >= thresholds.warning || precipitationMm >= 5.0 || rainProbability >= 65) {
      riskStatus = "warning";
    } else if (predictedWaterLevel >= thresholds.advisory || precipitationMm >= 1.0 || rainProbability >= 40) {
      riskStatus = "advisory";
    }

    const conformalBounds = hourlyMatch?.conformalBounds;

    return {
      timeStr,
      weekdayStr,
      temp,
      heatIndex,
      condition,
      rainProbability,
      precipitationMm,
      windSpeedKmH,
      windDirection,
      humidity,
      pressure,
      predictedWaterLevel,
      riskStatus,
      conformalBounds,
    };
  }, [forecastPoints, horizon, locale, thresholds, weather]);

  const getHeatIndexLabel = (hi: number) => {
    if (hi >= 52) return t("heatIndexCategories.extremeDanger");
    if (hi >= 42) return t("heatIndexCategories.danger");
    if (hi >= 33) return t("heatIndexCategories.extremeCaution");
    if (hi >= 27) return t("heatIndexCategories.caution");
    return t("heatIndexCategories.comfortable");
  };

  const getRiskDetails = (status: FloodRiskLevel, rainProb: number, precipMm: number) => {
    switch (status) {
      case "critical":
        return {
          label: t("riskBadges.critical"),
          waterStatusLabel: t("waterStatuses.critical"),
          color: "#e11d48",
          advice: t("advisories.critical"),
        };
      case "warning":
        return {
          label: t("riskBadges.warning"),
          waterStatusLabel: t("waterStatuses.warning"),
          color: "#f97316",
          advice: t("advisories.warning"),
        };
      case "advisory":
        return {
          label: t("riskBadges.advisory"),
          waterStatusLabel: t("waterStatuses.advisory"),
          color: "#eab308",
          advice: t("advisories.advisory"),
        };
      default:
        return {
          label: t("riskBadges.normal"),
          waterStatusLabel: t("waterStatuses.normal"),
          color: "#76e000",
          advice: rainProb >= 25 || precipMm > 0 ? t("advisories.rainNormal") : t("advisories.normal"),
        };
    }
  };

  const riskDetails = getRiskDetails(
    targetData.riskStatus,
    targetData.rainProbability,
    targetData.precipitationMm
  );

  // Rain prediction phrasing translated dynamically
  const willItRainText =
    targetData.rainProbability >= 70
      ? t("rainLikelihood.heavy")
      : targetData.rainProbability >= 40
        ? t("rainLikelihood.scattered")
        : t("rainLikelihood.low");

  return (
    <div className="w-full pt-1 pb-0">
      {/* 2-Column Hero: Left Side Hero Info + Right Side 4 Specific Glass Cards */}
      <div className="grid grid-cols-1 items-center gap-10 lg:grid-cols-12 lg:gap-14 xl:gap-20">
        {/* ── LEFT COLUMN: HERO INFORMATION ── */}
        <div className="flex flex-col items-start lg:col-span-7">
          {/* Station Location Selector */}
          <div className="w-full mb-2 md:mb-3">
            <StationSelectorPopover
              station={station}
              stations={stations}
              selectedStationId={station.stationPublicId}
              onStationSelect={onStationSelect || (() => undefined)}
              nearestStationId={nearestStationId}
              onDetectNearest={onDetectNearest}
              isLocating={isLocating}
              showAddress={false}
              stationGroup="weather"
              locationClassName="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-light"
              triggerClassName="w-full justify-start text-left p-0 border-0 bg-transparent hover:bg-transparent shadow-none"
            />
          </div>

          {/* Temperature Sub-label */}
          <p className="text-base font-medium text-light/90 text-left md:text-lg lg:text-xl mb-1">
            <strong>{t("temperature")}</strong>{" "}
            <span className="font-light">{t("todayAt")} {targetData.weekdayStr}</span>{" "}
            <span className="font-medium">{targetData.timeStr}</span>
          </p>

          {/* Giant Temperature & Animated Weather Icon */}
          <div className="flex items-center gap-3 my-3 md:my-5">
            <p className="flex gap-1 text-[clamp(4.5rem,12vw,8rem)] font-bold text-light tabular-nums tracking-tighter leading-none">
              {targetData.temp.toFixed(1)}
              <span className="mt-1.5 self-start text-2xl font-bold leading-none tracking-tight text-light md:mt-3 md:text-4xl lg:mt-4 lg:text-5xl">
                °C
              </span>
            </p>

            <WeatherIcon
              condition={targetData.condition}
              className="ml-2 h-16 w-16 shrink-0 items-center justify-center md:ml-4 md:h-22 md:w-22 lg:ml-5 lg:h-26 lg:w-26"
            />
          </div>

          {/* Warning / Risk Pill Badge */}
          <div className="mt-3 md:mt-4 flex items-center">
            <div
              className="inline-flex items-center rounded-full px-4.5 py-1 text-xs md:text-sm font-bold shadow-xs transition"
              style={{
                backgroundColor: riskDetails.color,
                color: riskDetails.label === "Normal" ? "#1e293b" : "#ffffff",
              }}
            >
              <TriangleAlert className="mr-2 h-3.5 w-3.5 shrink-0" />
              <span>{riskDetails.label}</span>
            </div>
          </div>

          {/* Citizen Advisory Statement */}
          <p className="mt-5 md:mt-6 w-full max-w-lg text-sm md:text-base font-normal text-light/95 leading-relaxed">
            {riskDetails.advice}
          </p>
        </div>

        {/* ── RIGHT COLUMN: 4 REQUESTED GLASS CARDS (2x2 Grid) ── */}
        <div className="w-full lg:col-span-5 flex justify-center lg:justify-end">
          <div className="grid grid-cols-2 gap-3.5 md:gap-4 w-full max-w-md">
            {/* 1. Heat Index Card */}
            <div className="glass flex min-h-32 md:min-h-36 flex-col justify-between p-4 md:p-4.5">
              <div className="flex items-center gap-2">
                <Flame className="h-4 w-4 md:h-4.5 md:w-4.5 text-light shrink-0" />
                <span className="text-xs md:text-sm text-light font-medium leading-tight">
                  {t("cards.heatIndex")}
                </span>
              </div>

              <div className="my-auto flex items-baseline gap-1 text-light">
                <span className="text-2xl md:text-3xl font-bold leading-none">
                  {targetData.heatIndex.toFixed(1)}
                </span>
                <span className="text-xs font-medium">°C</span>
              </div>

              <div className="text-[11px] font-semibold text-light/85 truncate">
                {getHeatIndexLabel(targetData.heatIndex)}
              </div>
            </div>

            {/* 2. Wind & Pressure Card */}
            <div className="glass flex min-h-32 md:min-h-36 flex-col justify-between p-4 md:p-4.5">
              <div className="flex items-center gap-2">
                <Wind className="h-4 w-4 md:h-4.5 md:w-4.5 text-light shrink-0" />
                <span className="text-xs md:text-sm text-light font-medium leading-tight">
                  {t("cards.windPressure")}
                </span>
              </div>

              <div className="my-auto flex items-baseline gap-1 text-light">
                <span className="text-xl sm:text-2xl md:text-3xl font-bold leading-none">
                  {targetData.windDirection} {targetData.windSpeedKmH}
                </span>
                <span className="text-xs font-medium">km/h</span>
              </div>

              <div className="text-[11px] font-semibold text-light/85 truncate">
                {targetData.pressure} hPa
              </div>
            </div>

            {/* 3. Rain Forecast on Selected Horizon Card */}
            <div className="glass flex min-h-32 md:min-h-36 flex-col justify-between p-4 md:p-4.5">
              <div className="flex items-center gap-2">
                <CloudRain className="h-4 w-4 md:h-4.5 md:w-4.5 text-light shrink-0" />
                <span className="text-xs md:text-sm text-light font-medium leading-tight">
                  {t("cards.rainChance")} ({horizon})
                </span>
              </div>

              <div className="my-auto flex items-baseline gap-1 text-light">
                <span className="text-2xl md:text-3xl font-bold leading-none">
                  {targetData.rainProbability}
                </span>
                <span className="text-xs font-medium">%</span>
                {targetData.conformalBounds && (
                  <span className="text-[10px] text-light/75 ml-1 font-mono tracking-tight bg-white/10 px-1.5 py-0.5 rounded-full">
                    ±1σ: {targetData.conformalBounds.likelyLower}–{targetData.conformalBounds.likelyUpper}%
                  </span>
                )}
              </div>

              <div className="text-[11px] font-semibold text-light/85 truncate">
                {willItRainText} {targetData.precipitationMm > 0 ? `(~${targetData.precipitationMm.toFixed(1)}mm)` : ""}
              </div>
            </div>

            {/* 4. Water Level & Flood Alert Card */}
            <div className="glass flex min-h-32 md:min-h-36 flex-col justify-between p-4 md:p-4.5">
              <div className="flex items-center gap-2">
                <Waves className="h-4 w-4 md:h-4.5 md:w-4.5 text-light shrink-0" />
                <span className="text-xs md:text-sm text-light font-medium leading-tight">
                  {t("cards.waterLevelFlood")}
                </span>
              </div>

              <div className="my-auto flex items-baseline gap-1 text-light">
                <span className="text-2xl md:text-3xl font-bold leading-none">
                  {targetData.predictedWaterLevel.toFixed(2)}
                </span>
                <span className="text-xs font-medium">m</span>
              </div>

              <div className="text-[11px] font-semibold text-light/85 truncate flex items-center gap-1">
                <span>{riskDetails.waterStatusLabel}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
