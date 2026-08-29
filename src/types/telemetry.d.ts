export interface StationPublicInfo {
  stationPublicId: string; 
  stationName: string;
  stationType: string;
  address: string;
  city: string;
  state: string;
  country: string;
  location: [number, number];
  isActive: boolean;
  referenceThreshold?: number;
}

export interface TelemetryMetrics {
  telemetryId: number;
  recordedAt: string;
  temperature?: number | null;
  humidity?: number | null;
  pressure?: number | null;
  heatIndex?: number | null;
  windDirection?: number | null;
  windSpeed?: number | null;
  precipitation?: number | null;
  hourlyPrecip?: number | null;
  uvIndex?: number | null;
  distance?: number | null;
  lightIntensity?: number | null;
  isSpatialEstimate?: boolean;
  estimateSource?: string;
  confidencePct?: number;
}

export interface TelemetryPublicDTO {
  station: StationPublicInfo;
  telemetry: TelemetryMetrics | null;
}

export interface TelemetryHistoryDTO {
  station: StationPublicInfo;
  telemetry: TelemetryMetrics[];
}
