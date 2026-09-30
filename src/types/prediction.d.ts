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

export interface ConformalUncertaintyBand {
  sigma: number;
  likelyLower: number;   // 68% Confidence (±1σ)
  likelyUpper: number;   // 68% Confidence (±1σ)
  extremeLower: number;  // 95% Confidence (±2σ)
  extremeUpper: number;  // 95% Confidence (±2σ)
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
  windDirection?: string;
  humidity: number;
  pressure?: number;
  conformalBounds?: ConformalUncertaintyBand;
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

export interface BlockedTargetNotice {
  status: string;
  reason: string;
}

export interface PredictionWeatherOverview {
  currentTemp: number;
  currentHeatIndex: number;
  condition: WeatherCondition;
  conditionText: string;
  humidity: number;
  windSpeed: number;
  windDirection?: string;
  pressure?: number;
  precipitationChance: number;
  summaryMessage: string;
  hourly: HourlyWeatherForecast[];
  daily: DailyWeatherForecast[];
  /**
   * UV index availability, taken from the engine's governance metadata.
   *
   * UV is `BLOCKED_BY_SENSOR_CALIBRATION` upstream: the raw sensors report up to
   * 11.0 at midnight, and `inference.py` raises on any UV request. The UI must
   * therefore NOT synthesise a clear-sky proxy value and present it as a reading.
   * When blocked, render the status instead of a number.
   */
  uvIndex: BlockedTargetNotice;
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

/**
 * Model governance metadata.
 *
 * These fields come straight from the validated Python engine
 * (`prediction-model/src/inference.py`). They were previously discarded at the
 * service boundary, which allowed the UI to present imperative flood warnings
 * for a model explicitly flagged `not_for_life_safety: true`.
 *
 * Every forecast surface MUST render this block.
 */
export interface ModelGovernance {
  /** e.g. "RESEARCH_PROTOTYPE" | "CANDIDATE_RESEARCH". */
  modelStatus: string;
  /** Always true for the current engine. Drives the mandatory UI disclaimer. */
  notForLifeSafety: boolean;
  productName: string;
  modelVersion: string;
  bundleVersion: string;
  activeBundleHorizon: string;
  policyVersion: string;
  policyCodeCommit: string;
  modelCodeCommit: string;
  checkpointSha256: string | null;
  policySha256: string | null;
  forecastOriginTimestamp: string;
  targetTimestamp: string | null;
  /** Age of the underlying model output, seconds. */
  ageSeconds: number | null;
  freshnessBudgetSeconds: number;
}

export interface ModelInputQuality {
  /** False when the anomaly gate quarantined the input sequence. */
  learnedOutputTrusted: boolean;
  sensorQualityFlag: string;
  policy: string;
  anomalySummary: string;
  anomalyCount: number;
}

export interface ModelSourceSelection {
  temperature: string;
  humidity: string;
  pressure: string;
  windSpeed: string;
  windDirection: string;
  heatIndex: string;
}

export interface ModelForecastProvenance {
  rainProbabilitySource: string;
  rainModelWeight: number;
  rainPersistenceWeight: number;
  operationalThreshold: number;
  operationalAlert: boolean;
  /** Raw learned output vs the persistence fallback, for auditability. */
  learnedVsPersistence: Record<string, number | null>;
  weatherUncertaintyStatus: string;
  weatherUncertaintyReason: string;
  blockedTargets: {
    uvIndex: { status: string; reason: string };
    lightIntensity: { status: string; reason: string };
  };
  waterLevelBeta: {
    predictedWaterLevelM: number;
    status: string;
    notForLifeSafety: boolean;
  } | null;
}

export interface PredictionPublicDTO {
  station: StationPublicInfo;
  summary: PredictionSummary;
  forecast: PredictionDataPoint[];
  history: WaterLevelHistoryMetricDataPoint[];
  weatherForecast: PredictionWeatherOverview;
  suddenRainBurst?: SuddenRainBurstPrediction;
  /** Present when a validated model forecast backs this response. */
  governance?: ModelGovernance;
  inputQuality?: ModelInputQuality;
  sourceSelection?: ModelSourceSelection;
  modelProvenance?: ModelForecastProvenance;
  /** Explains why no model forecast is attached, when one is not. */
  forecastUnavailable?: {
    reason: string;
    message: string;
  };
}

