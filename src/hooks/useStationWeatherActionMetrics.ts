import { useMemo } from "react";
import {
  useMetricConfig,
  type WeatherMetricConfig,
} from "@/hooks/useWeatherMetric";
import type { Locale } from "@/lib/i18n/translations";
import type { StationWeatherMetricGroup } from "@/lib/config/stationWeatherMetricGroups";
import {
  getWarningStyles,
  getWhatCanYouDo,
} from "@/lib/utils/weatherWarningUtil";
import type { TelemetryMetrics } from "@/types/telemetry";

/**
 * Prepares the action section from live telemetry only. In wet season,
 * precipitation uses the existing hourly reading for warnings and safety
 * actions; the derived full-day precipitation total does not affect actions.
 */

/** Returns the action metric, warning presentation, safety steps, and two stats. */
export const useStationWeatherActionMetrics = (
  telemetryData: TelemetryMetrics | null | undefined,
  metricGroup: StationWeatherMetricGroup,
  locale: Locale,
  dayPrecipitation?: number,
) => {
  const metrics = useMetricConfig(telemetryData, { dayPrecipitation });

  return useMemo(() => {
    const requestedActionMetricKey = metricGroup.actionMetric ?? metricGroup.primary;
    const requestedActionMetric = metrics.find((metric) => metric.key === requestedActionMetricKey);
    const hasRequestedPrecipitation =
      requestedActionMetricKey === "precipitation" &&
      typeof requestedActionMetric?.value === "number" &&
      requestedActionMetric.value > 0;
    const actionMetricKey =
      requestedActionMetricKey === "precipitation" && !hasRequestedPrecipitation
        ? "temperature"
        : requestedActionMetricKey;
    const actionMetric = metrics.find((metric) => metric.key === actionMetricKey);
    const warning = actionMetric?.warning;

    const secondaryKeys =
      requestedActionMetricKey === "precipitation"
        ? hasRequestedPrecipitation
          ? ["dayPrecipitation", "humidity"]
          : ["precipitation", "dayPrecipitation"]
        : metricGroup.secondary.filter((key) => key !== actionMetricKey);
    const stats = secondaryKeys
      .map((key) => metrics.find((metric) => metric.key === key))
      .filter((metric): metric is WeatherMetricConfig => !!metric)
      .slice(0, 2);

    return {
      actionMetric,
      actionItems: getWhatCanYouDo(warning?.whatCanYouDo, locale),
      stats,
      warning,
      warningStyles: getWarningStyles(warning?.color),
    };
  }, [locale, metricGroup.actionMetric, metricGroup.primary, metricGroup.secondary, metrics]);
};
