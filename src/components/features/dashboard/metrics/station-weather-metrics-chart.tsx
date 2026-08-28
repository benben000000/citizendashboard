"use client";
import React, { useMemo } from "react";
import StationParameterChart from "./station-parameter-chart";
import { getParameterByKey } from "@/lib/constants/parameters";
import type { ParameterConfig, ParameterType } from "@/types/parameter";
import { useStationParameterData } from "@/hooks/useStationParameterData";
import type { TelemetryMetricRaw } from "@/types/telemetry-raw";
import { useTranslations } from "next-intl";
import type { TranslationKey } from "@/lib/i18n/translations";
import { useStationPrecipitationHistory } from "@/hooks/useStationPrecipitationHistory";

interface StationWeatherMetricsChartProps {
  stationPublicId: string;
  metricKey: string;
}

const PARAMETER_LABEL_KEYS: Record<ParameterType, TranslationKey> = {
  temperature: "common.metrics.temperature",
  heatIndex: "common.metrics.heatIndex",
  humidity: "common.metrics.humidity",
  pressure: "common.metrics.pressure",
  windSpeed: "common.metrics.windSpeed",
  precipitation: "common.metrics.precipitation",
  uvIndex: "common.metrics.uvIndex",
  lightIntensity: "common.metrics.lightIntensity",
};

const StationWeatherMetricsChart: React.FC<StationWeatherMetricsChartProps> = ({
  stationPublicId,
  metricKey,
}) => {
  const t = useTranslations();
  const parameter: ParameterConfig | undefined = getParameterByKey(metricKey);
  const precipitationHistory = useStationPrecipitationHistory();

  // Compute a stable 48-hr start date (yesterday midnight) so the hook
  // dependency doesn't fire on every render.
  const dateRange = useMemo(() => {
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
    const startOfYesterday = new Date(startOfToday.getTime() - 24 * 60 * 60 * 1000);
    const endOfToday = new Date(startOfToday.getTime() + 24 * 60 * 60 * 1000);
    return {
      startDate: startOfYesterday.toISOString(),
      endDate: endOfToday.toISOString(),
      startOfTodayISO: startOfToday.toISOString(),
    };
  }, []);

  const { data, loading, error, refetch } = useStationParameterData(
    stationPublicId,
    parameter?.key,
    {
      enabled: metricKey !== "precipitation" || !precipitationHistory.enabled,
      startDate: dateRange.startDate,
      endDate: dateRange.endDate,
    }
  );

  const activeHistory =
    metricKey === "precipitation" && precipitationHistory.enabled
      ? precipitationHistory
      : { data, loading, error, refetch, startOfTodayISO: dateRange.startOfTodayISO };

  // Split the flat 48-hr array into today vs yesterday buckets — mirrors
  // exactly what HeatIndexCardContainer does with combinedHeatIndex.
  const { todayData, yesterdayData } = useMemo(() => {
    if (!activeHistory.data || activeHistory.data.length === 0) {
      return { todayData: [] as TelemetryMetricRaw[], yesterdayData: [] as TelemetryMetricRaw[] };
    }

    return {
      todayData: activeHistory.data.filter(
        (d) => d.recordedAt >= activeHistory.startOfTodayISO,
      ),
      yesterdayData: activeHistory.data.filter(
        (d) => d.recordedAt < activeHistory.startOfTodayISO,
      ),
    };
  }, [activeHistory.data, activeHistory.startOfTodayISO]);

  if (!parameter) {
    return (
      <div className="bg-white/50 border-2 border-card-border p-8 block">
        <p className="text-muted-foreground text-base font-mono uppercase tracking-wider text-center">
          {t("dashboard.chart.unavailableForMetric")}
        </p>
      </div>
    );
  }

  return (
    <StationParameterChart
      todayData={todayData}
      yesterdayData={yesterdayData}
      parameter={{
        ...parameter,
        label: t(PARAMETER_LABEL_KEYS[parameter.key]),
      }}
      loading={activeHistory.loading}
      error={activeHistory.error}
      onRetry={activeHistory.refetch}
    />
  );
};

export default StationWeatherMetricsChart;
