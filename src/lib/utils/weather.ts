// weather-utils.ts
// Utility helpers for weather station metrics

export const readingLabels: Record<string, string> = {
  temperature: "Temperature (°C)",
  humidity: "Humidity (%)",
  pressure: "Pressure (hPa)",
  heatIndex: "Heat Index (°C)",
  windDirection: "Wind Direction (°)",
  windSpeed: "Wind Speed (km/h)",
  precipitation: "Precipitation (mm)",
  uvIndex: "UV Index",
  distance: "Distance (cm)",
  lightIntensity: "Light Intensity (lx)",
};

export const readingColors: Record<string, string> = {
  temperature: "#ef4444", // Red
  humidity: "#3b82f6", // Blue
  pressure: "#a855f7", // Purple
  heatIndex: "#f97316", // Orange
  windDirection: "#10b981", // Green
  windSpeed: "#10b981", // Green
  precipitation: "#3b82f6", // Blue
  uvIndex: "#eab308", // Yellow
  distance: "#10b981", // Green
  lightIntensity: "#f59e0b", // Amber
};

/**
 * Extracts the unit of measurement from a label string, e.g. "Temperature (°C)" => "°C".
 * Returns an empty string if no unit is found.
 */
export const unitMeasurement: Record<string, string> = {
  temperature: "°C",
  humidity: "%",
  pressure: "mb",
  heatIndex: "°C",
  windDirection: "°",
  windSpeed: "km/h",
  precipitation: "mm",
  uvIndex: "",
  distance: "cm",
  lightIntensity: "lux",
};

export const WEATHER_COLORS = {
  heatIndex: "#fb923c", // orange-400
  temperature: "#f87171", // red-400
  humidity: "#60a5fa", // blue-400
  pressure: "#a78bfa", // violet-400
  wind: "#4ade80", // green-400
  windSpeed: "#4ade80", // green-400
  windDirection: "#4ade80", // green-400
  uvIndex: "#facc15", // yellow-400
  lightIntensity: "#fbbf24", // amber-400
  light: "#fbbf24", // amber-400
  precipitation: "#06b6d4", // cyan-600
} as const;

/**
 * Gets the label and color for a metric key.
 */
export function getWeatherMetricInfo(key: string) {
  return {
    label: readingLabels[key] || key,
    color: readingColors[key] || "#38bdf8",
    unit: unitMeasurement[key] || "",
  };
}

export function getWindDirection(deg: number): string {
  const directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const index = Math.round(deg / 45) % 8;
  return directions[index];
}

import { TelemetryMetrics } from "@/types/telemetry";

export type WeatherCondition = "sunny" | "hot" | "cold" | "partly-cloudy" | "cloudy" | "rain" | "storm" | "high-uv";

export function getWeatherCondition(telemetry: TelemetryMetrics): WeatherCondition {
  if (!telemetry) return "sunny";

  const { temperature, humidity, precipitation, windSpeed, uvIndex, hourlyPrecip } = telemetry;

  // 1. Measured Precipitation logic
  if ((precipitation && precipitation > 0) || (hourlyPrecip && hourlyPrecip > 0)) {
    if (windSpeed && windSpeed > 20) return "storm";
    return "rain";
  }

  // 2. Thermodynamic Convective Instability vs. Nocturnal Radiation Fog Filtering:
  // True convective precipitation requires dynamic wind shear or diurnal solar buoyancy.
  // Calm nocturnal saturated air (W <= 0.5 km/h at night) represents maritime radiation fog / low stratus.
  if (humidity != null && humidity >= 94 && temperature != null && temperature <= 26.8) {
    const now = new Date();
    const phHour = (now.getUTCHours() + 8) % 24;
    const isNocturnalCalm = (phHour >= 22 || phHour <= 6) && (!windSpeed || windSpeed <= 0.5);

    if (isNocturnalCalm) {
      return "cloudy"; // High-humidity radiation fog/stratus
    }
    if (windSpeed && windSpeed > 20) return "storm";
    return "rain"; // Active convective condensation
  }

  // 3. Temperature-based
  if (temperature != null) {
    if (temperature >= 35) return "hot"; // typical PH heat
    if (temperature <= 20) return "cold";
  }

  // 4. Humidity cloud logic
  if (humidity != null) {
    if (humidity > 85) return "cloudy";
    if (humidity > 60) return "partly-cloudy";
  }

  // 5. High UV warning
  if (uvIndex != null && uvIndex >= 8) return "high-uv";

  // Default
  return "sunny";
}
