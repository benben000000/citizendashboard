import { StationRaw } from "./telemetry-raw";

export interface WaterLevelMetricRaw {
  id: number;
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

export interface WaterLevelHistoryMetricPointRaw {
  id: number;
  recordedAt: string;
  createdAt: string;
  value: number | null;
}

export interface WaterLevelDashboardSingleRaw {
  station: StationRaw;
  waterLevel: WaterLevelMetricRaw | null;
}

export type WaterLevelDashboardRaw = WaterLevelDashboardSingleRaw[];

export interface WaterLevelHistoryRaw {
  station: StationRaw;
  waterLevel: WaterLevelMetricRaw[];
}

export interface WaterLevelHistoryMetricRaw {
  station: StationRaw;
  waterLevel: WaterLevelHistoryMetricPointRaw[];
}
