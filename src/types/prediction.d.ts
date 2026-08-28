import type { StationPublicInfo } from "./telemetry";
import type { WaterLevelHistoryMetricDataPoint } from "./water-level";
import type { WeatherCondition } from "@/lib/utils/weather";

export type PredictionHorizon = "1h" | "3h" | "6h" | "12h" | "24h" | "48h" | "72h";

export type FloodRiskLevel = "normal" | "advisory" | "warning" | "critical";

export type SuddenBurstType =
  | "sudden_heavy"
  | "short_burst_heavy"
  | "sudden_light"
  | "short_burst_light"
  | "none";

export interface SuddenRainBurstPrediction {
  detected: boolean;
  burstType: SuddenBurstType;
  title: string;
  intensityMmHr: number;
  probabilityPct: number;
  expectedWindow: string;
  durationMinutes: number;
  radarReflectivityDbz: number;
  convectiveCloudCover: number;
  advisory: string;
}

export interface PredictionDataPoint {
  timestamp: string;
  actualWaterLevel?: number | null;
  predictedWaterLevel: number;
  lowerBound: number;
  upperBound: number;
  rainfallAccumulationMm: number;
  rateOfRiseMPerHr: number;
  isForecast: boolean;
}

export interface HourlyWeatherForecast {
  time: string;
  timestamp: string;
  temp: number;
  heatIndex: number;
  condition: WeatherCondition;
  conditionText: string;
  rainProbability: number;
  precipitationMm: number;
  windSpeedKmH: number;
  humidity: number;
}

export interface DailyWeatherForecast {
  date: string;
  dayName: string;
  maxTemp: number;
  minTemp: number;
  maxHeatIndex: number;
  condition: WeatherCondition;
  conditionText: string;
  rainProbability: number;
  totalRainfallMm: number;
}

export interface PredictionWeatherOverview {
  currentTemp: number;
  currentHeatIndex: number;
  condition: WeatherCondition;
  conditionText: string;
  humidity: number;
  windSpeed: number;
  precipitationChance: number;
  summaryMessage: string;
  hourly: HourlyWeatherForecast[];
  daily: DailyWeatherForecast[];
}

export interface PredictionSummary {
  stationId: string;
  stationName: string;
  currentWaterLevel: number;
  peakPredictedLevel: number;
  peakTime: string;
  timeToPeakMinutes: number;
  riskLevel: FloodRiskLevel;
  confidenceScore: number;
  leadTimeHorizon: PredictionHorizon;
  lastRunAt: string;
  thresholds: {
    advisory: number;
    warning: number;
    critical: number;
  };
  suddenRainBurst?: SuddenRainBurstPrediction;
}

export interface PredictionPublicDTO {
  station: StationPublicInfo;
  summary: PredictionSummary;
  forecast: PredictionDataPoint[];
  history: WaterLevelHistoryMetricDataPoint[];
  weatherForecast: PredictionWeatherOverview;
  suddenRainBurst?: SuddenRainBurstPrediction;
}
