export interface StationRaw {
  stationName: string;
  stationType: string;
  location: [number, number] | { lat: number; lng: number };
  address: string;
  city: string;
  state: string;
  country: string;
  elevation: number;
  isActive: boolean;
  activatedAt: string | null;
  organizationId: number | null;
  organization: { id: number; organizationName: string } | null;
  id: string;
}


export interface TelemetryRaw {
  id?: number;
  recordedAt: string;
  heatIndex: number | null;
  temperature: number | null;
  humidity: number | null;
  pressure: number | null;
  wind: {
    direction: number | null;
    speed: number | null;
  };
  precipitation: number | null;
  uvIndex: number | null;
  distance: number | null;
  light?: number | null;
  lightIntensity?: number | null;
  hourlyPrecip?: number | null;
}

export interface TelemetryMetricRaw {
  id: number;
  recordedAt: string;
  createdAt?: string;
  value: number;
}

// History response from (/:stationId//?take=1&... | /?interval=...&startDate=...)
export interface TelemetryHistoryTakeRaw {
  station: StationRaw;
  telemetry: TelemetryRaw[];
}

// History response (/:stationId/:metric/?interval=...&startDate=...)
export interface TelemetryHistoryMetricRaw {
  station: StationRaw;
  telemetry: TelemetryMetricRaw[];
}

/** Shape of the telemetry object nested inside the dashboard response */
export interface TelemetryDashboardDataRaw {
  id: number;
  recordedAt: string;
  heatIndex: number | null;
  temperature: number | null;
  humidity: number | null;
  pressure: number | null;
  wind: {
    direction: number | null;
    speed: number | null;
  };
  precipitation: number | null;
  uvIndex: number | null;
  distance: number | null;
  lightIntensity: number | null;
}

export interface DashboardSingleRaw {
  station: StationRaw;
  telemetry: TelemetryRaw | TelemetryDashboardDataRaw | null;
}

export type DashboardRaw = DashboardSingleRaw[] | { stations: DashboardSingleRaw[] };
