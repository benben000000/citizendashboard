/**
 * Weather parameter configuration for Today Graph tabs
 */

import type { ParameterConfig, ParameterType } from '@/types/parameter';
import type { TelemetryMetrics } from '@/types/telemetry';

export const PARAMETERS: ParameterConfig[] = [
  {
    key: 'temperature',
    apiKey: 'temperature',
    label: 'Temperature',
    color: '#ef4444',
    unit: '°C',
    chartType: 'line',
  },
  {
    key: 'heatIndex',
    apiKey: 'heatIndex',
    label: 'Heat Index',
    color: '#ef4444',
    unit: '°C',
    chartType: 'line',
  },
  {
    key: 'humidity',
    apiKey: 'humidity',
    label: 'Humidity',
    color: '#3b82f6',
    unit: '%',
    chartType: 'area',
  },
  {
    key: 'pressure',
    apiKey: 'pressure',
    label: 'Pressure',
    color: '#a855f7',
    unit: 'hPa',
    chartType: 'area',
  },
  {
    key: 'windSpeed',
    apiKey: 'windSpeed',
    label: 'Wind Speed',
    color: '#06b6d4',
    unit: 'km/h',
    chartType: 'bar',
  },
  {
    key: 'precipitation',
    apiKey: 'precipitation',
    label: 'Precipitation',
    color: '#3b82f6',
    unit: 'mm',
    chartType: 'bar',
  },
  {
    key: 'uvIndex',
    apiKey: 'uvIndex',
    label: 'UV Index',
    color: '#eab308',
    unit: '',
    chartType: 'bar',
  },
  {
    key: 'lightIntensity',
    apiKey: 'lightIntensity',
    label: 'Light Intensity',
    color: '#fbbf24',
    unit: 'lx',
    chartType: 'area',
  },
];

// Single lookup table for metric units, API keys, chart type, and display metadata.
export const PARAMETER_BY_KEY = PARAMETERS.reduce(
  (acc, parameter) => {
    acc[parameter.key] = parameter;
    return acc;
  },
  {} as Record<ParameterType, ParameterConfig>
);

export function isParameterType(value: string): value is ParameterType {
  return value in PARAMETER_BY_KEY;
}

export function getParameterByKey(
  key: string | null | undefined
): ParameterConfig | undefined {
  return key && isParameterType(key) ? PARAMETER_BY_KEY[key] : undefined;
}

// Centralize metric value access so map markers and charts cannot drift on key names.
export function getTelemetryMetricValue(
  metricKey: string,
  telemetry: TelemetryMetrics | null | undefined
): number | null {
  if (!telemetry || !isParameterType(metricKey)) return null;
  if (metricKey === "precipitation") {
    return telemetry.hourlyPrecip ?? telemetry.precipitation ?? null;
  }
  return telemetry[metricKey] ?? null;
}

// Weather warning labels use "light" while telemetry uses "lightIntensity".
export function getWarningMetricKey(metricKey: string): string {
  return metricKey === "lightIntensity" ? "light" : metricKey;
}
