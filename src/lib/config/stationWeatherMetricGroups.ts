import type { ParameterType } from "@/types/parameter";

export type StationWeatherDisplayMetric = ParameterType | "dayPrecipitation";

export interface StationWeatherMetricGroup {
  primary: ParameterType;
  secondary: StationWeatherDisplayMetric[];
  actionMetric?: ParameterType;
}

export const STATION_WEATHER_METRIC_GROUPS = {
  drySeason: {
    primary: "heatIndex",
    secondary: ["temperature", "humidity", "uvIndex", "lightIntensity"],
    actionMetric: "heatIndex",
  },
  wetSeason: {
    primary: "temperature",
    secondary: ["dayPrecipitation", "precipitation", "humidity", "windSpeed"],
    actionMetric: "precipitation",
  },
} as const satisfies Record<string, StationWeatherMetricGroup>;

export const getStationWeatherMetricGroupForDate = (
  date: Date = new Date(),
): StationWeatherMetricGroup => {
  const month = date.getMonth();
  const isWetSeason = month >= 5 && month <= 9;

  return isWetSeason
    ? STATION_WEATHER_METRIC_GROUPS.wetSeason
    : STATION_WEATHER_METRIC_GROUPS.drySeason;
};
