import React, { useCallback, useMemo } from "react";
import { TelemetryMetrics } from "@/types/telemetry";
import { Sun, Thermometer, Droplets, Eye, CloudRain, Gauge, Wind } from "lucide-react";
import { getWindDirectionLabel } from "@/lib/utils/weatherUtil";
import { getReferenceWarning, getWarningStyles } from "@/lib/utils/weatherWarningUtil";
import { getParameterByKey } from "@/lib/constants/parameters";
import { useLocale, useTranslations } from "next-intl";
import type { Locale } from "@/lib/i18n/translations";

/**
 * Builds presentation-ready weather metrics from raw telemetry. This module
 * centralizes localized labels, units, icons, warning metadata, wind display,
 * and optional derived values such as today's accumulated precipitation.
 */

/** Returns warning and selection styles for an individual metric card. */
export const useCardStyles = (
  warning: { color: string; term: string } | null | undefined,
  isSelected: boolean,
  isSelectable: boolean = true,
) => {
  const hasWarning = !!warning;
  const warningStyles = getWarningStyles(warning?.color);
  const cardStyle = useMemo<React.CSSProperties>(() => {
    const style: React.CSSProperties = {
      backgroundColor: ``,
    };
    // if (isSelectable && isSelected) {
      // Use only longhand border properties to avoid conflicts
      // style.borderTop = `2px solid ${darkenColor(warningStyles.text, 0.1)}`;
      // style.borderRight = `2px solid ${darkenColor(warningStyles.text, 0.1)}`;
      // style.borderLeft = `2px solid ${darkenColor(warningStyles.text, 0.1)}`;
      // style.borderBottom = `2px solid ${darkenColor(warningStyles.text, 0.1)}`;
    // }

    if ( isSelected && isSelectable) {
      style.transform = "scale(1.025)";
      
    }

    return style;
  }, [isSelected, isSelectable]);

  return { hasWarning, warningStyles, cardStyle };
};

/** Converts telemetry into the metric configurations consumed by dashboard UI. */
export const useMetricConfig = (
  telemetryMetrics: TelemetryMetrics | null | undefined,
  derivedMetrics: { dayPrecipitation?: number } = {},
) => {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const unitFor = useCallback((key: string) => getParameterByKey(key)?.unit ?? "", []);

  return useMemo(() => {
    const precipitationValue = telemetryMetrics?.hourlyPrecip ?? telemetryMetrics?.precipitation;

    const metrics = [
      {
        key: "temperature",
        label: t("common.metrics.temperature"),
        value: telemetryMetrics?.temperature,
        unit: t("common.units.celsius"),
        icon: Thermometer,
        color: "#f87171",
        warning:
          telemetryMetrics?.temperature != null
            ? getReferenceWarning("Temperature", telemetryMetrics.temperature, true, locale)
            : null,
      },
      {
        key: "heatIndex",
        label: t("common.metrics.heatIndex"),
        value: telemetryMetrics?.heatIndex,
        unit: t("common.units.celsius"),
        icon: Thermometer,
        color: "#fb923c",
        warning:
          telemetryMetrics?.heatIndex != null ? getReferenceWarning("Heat Index", telemetryMetrics.heatIndex, true, locale) : null,
      },
      {
        key: "humidity",
        label: t("common.metrics.humidity"),
        value: telemetryMetrics?.humidity,
        unit: unitFor("humidity"),
        icon: Droplets,
        color: "#60a5fa",
        warning: telemetryMetrics?.humidity != null ? getReferenceWarning("Humidity", telemetryMetrics.humidity, true, locale) : null,
      },
      {
        key: "pressure",
        label: t("common.metrics.pressure"),
        value: telemetryMetrics?.pressure,
        unit: unitFor("pressure"),
        icon: Gauge,
        color: "#a78bfa",
        warning: telemetryMetrics?.pressure != null ? getReferenceWarning("Pressure", telemetryMetrics.pressure, true, locale) : null,
      },
      {
        key: "windSpeed",
        label: t("common.metrics.wind"),
        value: telemetryMetrics?.windSpeed,
        unit: unitFor("windSpeed"),
        displayValue:
          telemetryMetrics?.windSpeed != null
            ? `${telemetryMetrics.windDirection != null ? getWindDirectionLabel(telemetryMetrics.windDirection).direction : ""} ${(Math.round(telemetryMetrics.windSpeed * 10) / 10).toFixed(0)} `
            : "--",
        icon: Wind,
        color: "#4ade80",
        warning:
          telemetryMetrics?.windSpeed != null ? getReferenceWarning("Wind Speed", telemetryMetrics.windSpeed, true, locale) : null,
      },
      {
        key: "precipitation",
        label: t("common.metrics.precipitation"),
        value: precipitationValue,
        subValue: undefined,
        unit: unitFor("precipitation"),
        icon: CloudRain,
        color: "#06b6d4",
        warning:
          precipitationValue != null
            ? getReferenceWarning("Precipitation", precipitationValue, true, locale)
            : null,
      },
      {
        key: "uvIndex",
        label: t("common.metrics.uvIndex"),
        value: telemetryMetrics?.uvIndex,
        unit: unitFor("uvIndex"),
        icon: Eye,
        color: "#facc15",
        warning: telemetryMetrics?.uvIndex != null ? getReferenceWarning("UV Index", telemetryMetrics.uvIndex, true, locale) : null,
      },
      {
        key: "lightIntensity",
        label: t("common.metrics.lightIntensity"),
        value: telemetryMetrics?.lightIntensity,
        unit: unitFor("lightIntensity"),
        icon: Sun,
        color: "#fbbf24",
        warning:
          telemetryMetrics?.lightIntensity != null
            ? getReferenceWarning("Light", telemetryMetrics.lightIntensity, true, locale)
            : null,
      },
    ];

    if (derivedMetrics.dayPrecipitation !== undefined) {
      metrics.push({
        key: "dayPrecipitation",
        label: t("common.metrics.dayPrecipitation"),
        value: derivedMetrics.dayPrecipitation,
        unit: unitFor("precipitation"),
        icon: CloudRain,
        color: "#0891b2",
        warning: null,
      });
    }

    return metrics;
  }, [derivedMetrics.dayPrecipitation, locale, telemetryMetrics, t, unitFor]);
};

export type WeatherMetricConfig = ReturnType<typeof useMetricConfig>[number];
