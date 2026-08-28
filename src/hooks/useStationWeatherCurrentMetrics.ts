import { useMemo } from "react";
import {
  useMetricConfig,
  type WeatherMetricConfig,
} from "@/hooks/useWeatherMetric";
import type { StationWeatherMetricGroup } from "@/lib/config/stationWeatherMetricGroups";
import type { TelemetryMetrics } from "@/types/telemetry";

/**
 * Selects the hero and secondary metrics for the current-weather section.
 * It includes the derived daily precipitation card when supplied and can show
 * an active precipitation warning alongside the wet-season temperature hero.
 */

const hasPositiveMetricValue = (metric: WeatherMetricConfig | undefined) =>
  typeof metric?.value === "number" ? metric.value > 0 : metric?.value != null;

/** Returns the current section's hero metric, supporting metrics, and suggestion. */
export const useStationWeatherCurrentMetrics = (
  telemetryData: TelemetryMetrics | null | undefined,
  metricGroup: StationWeatherMetricGroup,
  dayPrecipitation?: number,
) => {
  const metrics = useMetricConfig(telemetryData, { dayPrecipitation });

  return useMemo(() => {
    const primaryMetric = metrics.find((metric) => metric.key === metricGroup.primary);
    const temperatureMetric = metrics.find((metric) => metric.key === "temperature");
    const precipitationMetric = metrics.find((metric) => metric.key === "precipitation");
    const hasPrecipitationValue = hasPositiveMetricValue(precipitationMetric);
    const shouldShowPrecipitationWarning =
      primaryMetric?.key === "temperature" &&
      hasPrecipitationValue &&
      !!precipitationMetric?.warning;

    const heroMetric = shouldShowPrecipitationWarning
      ? {
          ...primaryMetric,
          warning: precipitationMetric.warning,
        }
      : primaryMetric;

    const suggestedActionMetric =
      hasPrecipitationValue && precipitationMetric?.warning?.suggestionAction
        ? precipitationMetric
        : temperatureMetric;

    const otherParams = metricGroup.secondary
      .filter((key) => key !== metricGroup.primary)
      .map((key) => metrics.find((metric) => metric.key === key))
      .filter((metric): metric is WeatherMetricConfig => !!metric);

    return {
      heroMetric,
      otherParams,
      suggestedAction: suggestedActionMetric?.warning?.suggestionAction,
    };
  }, [metricGroup.primary, metricGroup.secondary, metrics]);
};
