"use client";

import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";

import PublicDashboardTopBar from "@/components/shared/public-dashboard-top-bar";
import { cn } from "@/lib/utils/cn";
import {
  formatWaterLevelChange,
  formatWaterLevelNumber,
  getWaterLevelTrend,
  type WaterLevelTrend,
} from "@/lib/utils/water-level";
import type { Locale } from "@/lib/i18n/translations";
import type { WaterLevelHistoryMetricDataPoint } from "@/types/water-level";
import type { StationPublicInfo } from "@/types/telemetry";

import WaterLevelTrendChart, { type WaterLevelChartPoint } from "./water-level-trend-chart";
import WaterLevelMap from "./water-level-map";
import WaterLevelStationSelector from "./water-level-station-selector";
import { formatDate } from "@/lib/utils/date";

interface WaterLevelDashboardProps {
  station: StationPublicInfo | null;
  stations: StationPublicInfo[];
  currentValue: number | null;
  currentRecordedAt: string | null;
  todayHistory: WaterLevelHistoryMetricDataPoint[];
  yesterdayHistory: WaterLevelHistoryMetricDataPoint[];
  hasCurrentError: boolean;
  hasHistoryError: boolean;
}

const TREND_STYLES: Record<
  WaterLevelTrend,
  { wrapper: string; icon: typeof ArrowRight; label: string }
> = {
  rising: {
    wrapper: "border-emerald-950/25 bg-emerald-700/35 text-emerald-950 shadow-emerald-950/15",
    icon: ArrowUpRight,
    label: "text-[var(--water-level-trend-rising)]",
  },
  falling: {
    wrapper: "border-sky-950/25 bg-sky-700/35 text-sky-950 shadow-sky-950/15",
    icon: ArrowDownRight,
    label: "text-[var(--water-level-trend-falling)]",
  },
  stable: {
    wrapper: "border-amber-950/25 bg-amber-500/40 text-amber-950 shadow-amber-950/15",
    icon: ArrowRight,
    label: "text-[var(--water-level-trend-stable)]",
  },
  unknown: {
    wrapper: "border-slate-950/20 bg-slate-700/25 text-slate-950 shadow-slate-950/10",
    icon: ArrowRight,
    label: "text-[var(--water-level-trend-unknown)]",
  },
};

function sortHistoryAscending(history: WaterLevelHistoryMetricDataPoint[]) {
  return [...history].sort(
    (left, right) => new Date(left.recordedAt).getTime() - new Date(right.recordedAt).getTime()
  );
}

function buildTrendContext(history: WaterLevelHistoryMetricDataPoint[]) {
  const validHistory = sortHistoryAscending(history).filter((point) => typeof point.value === "number");
  const firstValue = validHistory[0]?.value ?? null;
  const latestHistoryValue = validHistory[validHistory.length - 1]?.value ?? null;

  if (firstValue === null || latestHistoryValue === null || validHistory.length < 2) {
    return {
      change: null,
      trend: "unknown" as WaterLevelTrend,
    };
  }

  const change = latestHistoryValue - firstValue;

  return {
    change,
    trend: getWaterLevelTrend(change),
  };
}

function makeChartData(history: WaterLevelHistoryMetricDataPoint[]): WaterLevelChartPoint[] {
  return sortHistoryAscending(history)
    .filter((point) => typeof point.value === "number")
    .map((point) => ({
      recordedAt: point.recordedAt,
      value: point.value,
    }));
}

export default function WaterLevelDashboard({
  station,
  stations,
  currentValue,
  currentRecordedAt,
  todayHistory,
  yesterdayHistory,
  hasCurrentError,
  hasHistoryError,
}: WaterLevelDashboardProps) {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const todayChartData = makeChartData(todayHistory);
  const yesterdayChartData = makeChartData(yesterdayHistory);
  const { change, trend } = buildTrendContext(todayHistory);
  const trendStyle = TREND_STYLES[trend];
  const showDataWarnings = hasCurrentError || hasHistoryError;
  const trendLabel = t(`waterLevel.trend.${trend}`);

  return (
    <main className="relative mx-auto flex min-h-[100svh] w-full max-w-7xl flex-1 flex-col gap-3 px-4 py-3 md:px-6 md:py-4 lg:h-[100svh] lg:overflow-hidden">
      <div className="rounded-2xl bg-[#F9F9F6]/60 backdrop-blur-lg">
        <div className="px-6 py-3">
          <PublicDashboardTopBar />
        </div>
      </div>

      <div className="mt-4 mb-2">
        <WaterLevelStationSelector station={station} stations={stations} />
      </div>

      <section className="grid min-h-0 flex-1 items-stretch gap-3 lg:grid-cols-[minmax(0,1.08fr)_minmax(390px,0.92fr)]">
        

        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          <div>
            <h2 className="mb-3 border-l-4 border-l-main pl-2 text-lg font-semibold text-light md:text-xl">
              {t("waterLevel.metrics.title")}
            </h2>
            <div className="flex flex-col gap-2">
              <div className="glass px-5 py-3">
                <div className="flex flex-col gap-2">
                  <div className="flex items-start justify-between gap-3">
                    <p className="min-w-0 text-xs font-medium uppercase tracking-widest opacity-75 md:text-[0.6875rem]">
                      {t("waterLevel.metrics.waterLevel")}
                    </p>
                    <div className="ml-auto shrink-0 text-right">
                      <p className="max-w-[9rem] text-xs leading-tight text-light md:max-w-none md:text-sm">
                        {formatDate(currentRecordedAt ?? undefined, locale).formatted}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-end gap-2">
                    <div className="flex items-end gap-1">
                      <span className="text-5xl font-semibold tracking-tight text-foreground md:text-6xl">
                        {formatWaterLevelNumber(currentValue)}
                      </span>
                      <span className="text-xl font-medium text-foreground md:text-2xl">cm</span>
                    </div>
                    {/* <div
                      className={cn(
                        "inline-flex shrink-0 items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em]",
                        trendStyle.wrapper
                      )}
                    >
                      <TrendIcon className="h-3.5 w-3.5" />
                      <span>{trendLabel}</span>
                    </div> */}
                  </div>
                </div>
              </div>

              <div className="flex gap-2 p">
                <div className="glass w-1/2 px-5 py-3">
                  <p className="text-xs font-medium uppercase tracking-widest opacity-75 md:text-[0.6875rem]">
                    {t("waterLevel.metrics.changeFromToday")}
                  </p>
                  <p className={cn("mt-2 text-xl font-semibold", trendStyle.label)}>
                    {formatWaterLevelChange(change)}
                  </p>
                </div>

                <div className="glass w-1/2 px-5 py-3">
                  <p className="text-xs font-medium uppercase tracking-widest opacity-75 md:text-[0.6875rem]">
                    {t("waterLevel.metrics.trend")}
                  </p>
                  <p className={cn("mt-2 text-xl font-semibold", trendStyle.label)}>
                    {trendLabel}
                  </p>
                </div>

              </div>
            </div>
          </div>


          <div className="min-h-0">
            <div className="mb-2">
              <h2 className="mb-3 border-l-4 border-l-main pl-2 text-lg font-semibold text-light md:text-xl">
                {t("waterLevel.chart.title")}
              </h2>
              
            </div>
            <WaterLevelTrendChart
              todayData={todayChartData}
              yesterdayData={yesterdayChartData}
              referenceThreshold={station?.referenceThreshold}
              waterLevelLabel={t("waterLevel.metrics.waterLevel")}
              notEnoughHistoryLabel={t("waterLevel.chart.notEnoughHistory")}
              bridgeReferenceLabel={t("waterLevel.chart.bridgeReference")}
            />
          </div>

          {showDataWarnings ? (
            <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-900 shadow-sm dark:text-amber-100">
              {hasCurrentError && hasHistoryError
                ? t("waterLevel.warnings.allUnavailable")
                : hasCurrentError
                  ? t("waterLevel.warnings.currentUnavailable")
                  : t("waterLevel.warnings.historyUnavailable")}
            </p>
          ) : null}
        </div>

        <div className="flex min-h-0 min-w-0 flex-col gap-3 lg:h-full">
          <div className="flex h-full min-h-[420px] flex-col lg:min-h-0">
            <h2 className="mb-3 border-l-4 border-l-main pl-2 text-lg font-semibold text-light md:text-xl">
              {t("dashboard.info.exploreYourArea")}
            </h2>
            <div className="h-[420px] overflow-hidden rounded-lg lg:h-auto lg:flex-1">
              <WaterLevelMap
                station={station}
                currentValue={currentValue}
                tokenMissingLabel={t("waterLevel.map.tokenMissing")}
                stationFallbackLabel={t("waterLevel.map.stationFallback")}
              />
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
