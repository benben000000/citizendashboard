import { StationPublicInfo } from "./telemetry";

export interface WaterLevelMetrics {
  waterLevelId: number;
  recordedAt: string;
  startTimestamp?: string | null;
  endTimestamp?: string | null;
  sampleInterval?: number | null;
  sampleCount?: number | null;
  filteredSampleCount?: number | null;
  spikeCount?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  rawMode?: number | null;
  calculatedWaterLevel?: number | null;
  median?: number | null;
  frequentRangeLow?: number | null;
  frequentRangeHigh?: number | null;
  estimatedMovAvg?: number | null;
}

export interface WaterLevelPublicDTO {
  station: StationPublicInfo;
  waterLevel: WaterLevelMetrics | null;
}

export interface WaterLevelHistoryDTO {
  station: StationPublicInfo;
  waterLevel: WaterLevelMetrics[];
}

export interface WaterLevelHistoryMetricDataPoint {
  id: number;
  recordedAt: string;
  createdAt: string;
  value: number | null;
}

export interface WaterLevelHistoryMetricDTO {
  station: StationPublicInfo;
  waterLevel: WaterLevelHistoryMetricDataPoint[];
}
