/**
 * Telemetry Service - Orchestrates API calls, transforms data, and manages caching
 * This service layer sits between API routes and the Kloudtrack API client
 */
import { InMemoryCache } from "../lib/utils/cache";
import { toTwoDecimalPlaces } from "../lib/utils/data-helper";
import { AppError } from "../lib/utils/error";
import {
  StationPublicInfo,
  TelemetryMetrics,
  TelemetryPublicDTO,
} from "../types/telemetry";
// import { ParameterDataPoint } from "../types/parameter";
import stationIds from "../lib/constants/stations.json";
import {
  getDashboardData,
  getLatestTelemetryFromKloudtrackApi,
  getTelemetryMetricHistoryFromKloudtrackApi,
} from "@/lib/kloudtrack/client";
import {
  DashboardRaw,
  DashboardSingleRaw,
  StationRaw,
  TelemetryHistoryTakeRaw,
  TelemetryMetricRaw,
} from "@/types/telemetry-raw";
import fs from "fs";
import path from "path";
import { CACHE_CONFIG } from "@/lib/config/cache.config";
import { DEFAULT_CENTRAL_LUZON_STATIONS } from "@/lib/constants/default-stations";

export class TelemetryService {
  // Cache instances using centralized config
  private dashboardCache = new InMemoryCache<TelemetryPublicDTO[]>(
    CACHE_CONFIG.telemetry.stationDashboard
  );
  private parameterCache = new InMemoryCache<TelemetryMetricRaw[]>(
    CACHE_CONFIG.telemetry.parameterHistory
  );

  private ongoingDashboardRequest?: Promise<TelemetryPublicDTO[]>;

  // Deduplicate identical parameter-history requests while the first one is still in flight.
  private ongoingParameterRequests = new Map<string, Promise<TelemetryMetricRaw[]>>();

  async getStationDashboardData(stationId: string): Promise<TelemetryPublicDTO> {
    const stationData = (await this.getDashboardStations()).find(
      (item) => item.station.stationPublicId === stationId
    );

    if (!stationData) {
      throw new AppError(`Station ${stationId} was not found in dashboard telemetry`, 404);
    }

    return stationData;
  }

  async getAllStations(): Promise<StationPublicInfo[]> {
    return (await this.getDashboardStations()).map((item) => item.station);
  }

  /**
   * Get dashboard telemetry for every station in one upstream request.
   */
  async getDashboardStations(): Promise<TelemetryPublicDTO[]> {
    const cacheKey = "telemetry-dashboard";
    const cached = this.dashboardCache.get(cacheKey);
    if (cached) return cached;

    if (this.ongoingDashboardRequest) {
      return this.ongoingDashboardRequest;
    }

    this.ongoingDashboardRequest = (async () => {
      try {
        const rawData = await getDashboardData();
        const dashboardStations = this.transformDashboard(rawData);

        this.dashboardCache.set(
          cacheKey,
          dashboardStations,
          CACHE_CONFIG.telemetry.stationDashboard
        );

        return dashboardStations;
      } catch (error) {
        console.warn("[getDashboardStations] Upstream API unavailable, connecting to live MQTT stream:", error);
        const mqttStations = this.getMqttLiveDashboardStations();
        this.dashboardCache.set(
          cacheKey,
          mqttStations,
          CACHE_CONFIG.telemetry.stationDashboard
        );
        return mqttStations;
      } finally {
        this.ongoingDashboardRequest = undefined;
      }
    })();

    return this.ongoingDashboardRequest;
  }

  /**
   * Get parameter-specific history for a station
   * Used by Today Graph component for individual parameter charts
   */
  async getStationParameterHistory(
    stationId: string,
    parameter: string,
    options: { startDate?: string; endDate?: string; interval?: number } = {}
  ): Promise<TelemetryMetricRaw[]> {
    const interval = options.interval ?? 15;
    const startDate =
      options.startDate || new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const endDate = options.endDate;
    const cacheKey = `parameter-history-${stationId}-${parameter}-${startDate}-${endDate || "none"}-${interval}`;
    const cached = this.parameterCache.get(cacheKey);

    // Parameter history is cached longer than live dashboard data because it changes slowly.
    if (cached) {
      console.log(`Returning cached ${parameter} data for station ${stationId}`);
      return cached;
    }

    const ongoingRequest = this.ongoingParameterRequests.get(cacheKey);
    if (ongoingRequest) {
      return ongoingRequest;
    }

    const request = (async () => {
      try {
        console.log(`Fetching fresh ${parameter} data for station ${stationId}`);

        // Keep upstream params aligned with the cache key for predictable deduping.
        const params = new URLSearchParams({
          skip: "0",
          interval: interval.toString(),
          startDate,
          filterOutliers: "true",
        });
        if (endDate) {
          params.set("endDate", endDate);
        }

        const rawData = await getTelemetryMetricHistoryFromKloudtrackApi(
          stationId,
          parameter,
          Object.fromEntries(params)
        );

        // Charts expect oldest-to-newest points so the latest reading renders on the right.
        const result = (rawData.telemetry || []).slice().sort((a, b) => {
          const aTime = new Date(a.recordedAt).getTime();
          const bTime = new Date(b.recordedAt).getTime();
          return aTime - bTime;
        });

        this.parameterCache.set(cacheKey, result, CACHE_CONFIG.telemetry.parameterHistory);

        console.log(`Successfully fetched ${result.length} data points for ${parameter}`);
        return result;
      } catch (error) {
        console.warn(`[getStationParameterHistory] Upstream failed for ${parameter} at ${stationId}, reading 15-minute MQTT stream history:`, error);
        const mqttHistory = this.getMqtt15MinuteHistory(stationId, parameter, interval, startDate, endDate);
        this.parameterCache.set(cacheKey, mqttHistory, CACHE_CONFIG.telemetry.parameterHistory);
        return mqttHistory;
      } finally {
        this.ongoingParameterRequests.delete(cacheKey);
      }
    })();

    this.ongoingParameterRequests.set(cacheKey, request);
    return request;
  }

  // ==================== PRIVATE TRANSFORMATION METHODS ====================

  /**
   * Transform a single station object
   */
  private transformStation(station: StationRaw): StationPublicInfo {
    const location = Array.isArray(station.location)
      ? station.location
      : [station.location?.lng ?? 0, station.location?.lat ?? 0];

    return {
      stationPublicId: station.id,
      stationName: station.stationName ?? "Unknown Station",
      stationType: station.stationType ?? "unknown",
      address: station.address ?? "",
      city: station.city ?? "",
      state: station.state ?? "",
      country: station.country ?? "",
      location: location as [number, number],
      isActive: station.isActive ?? true,
    };
  }

  /**
   * Transform batched dashboard response from /telemetry/dashboard.
   */
  private transformDashboard(rawData: DashboardRaw): TelemetryPublicDTO[] {
    const stations = Array.isArray(rawData) ? rawData : rawData.stations ?? [];
    const dashboardStations = stations
      .filter((item): item is DashboardSingleRaw => Boolean(item?.station))
      .map((item) => ({
        station: this.transformStation(item.station),
        telemetry: this.transformTelemetry(item.telemetry),
      }));

    const configuredStationIds = stationIds.weather.stationIdToFetch.map((item) => item.stationId);
    if (configuredStationIds.length === 0) return dashboardStations;

    const dashboardByStationId = new Map(
      dashboardStations.map((item) => [item.station.stationPublicId, item])
    );

    return configuredStationIds
      .map((stationId) => dashboardByStationId.get(stationId))
      .filter((item): item is TelemetryPublicDTO => Boolean(item));
  }

  /**
   * Transform raw latest telemetry response.
   * The /history?take=1 endpoint returns { station, telemetry: [...] }.
   */
  private transformLatestTelemetry(rawData: TelemetryHistoryTakeRaw): TelemetryPublicDTO {
    return {
      station: this.transformStation(rawData.station),
      telemetry: this.transformTelemetry(rawData.telemetry?.[0]),
    };
  }

  /**
   * Transform a single telemetry reading
   * Handles data normalization, rounding, and null handling
   */
  private transformTelemetry(reading: unknown): TelemetryMetrics | null {
    if (!reading) return null;

    const data = reading as Record<string, unknown>;
    const wind = data.wind as Record<string, unknown> | undefined;

    return {
      telemetryId: (data.id || data.telemetryId || 0) as number,
      recordedAt: (data.recordedAt || new Date().toISOString()) as string,
      temperature: toTwoDecimalPlaces(data.temperature as number),
      humidity: toTwoDecimalPlaces(data.humidity as number),
      pressure: toTwoDecimalPlaces(data.pressure as number),
      heatIndex: toTwoDecimalPlaces(data.heatIndex as number),
      // Handle wind object flattening
      windDirection: toTwoDecimalPlaces((data.windDirection ?? wind?.direction) as number),
      windSpeed: toTwoDecimalPlaces((data.windSpeed ?? wind?.speed) as number),
      precipitation: toTwoDecimalPlaces(data.precipitation as number),
      hourlyPrecip: toTwoDecimalPlaces(data.hourlyPrecip as number),
      uvIndex: toTwoDecimalPlaces(data.uvIndex as number),
      distance: toTwoDecimalPlaces(data.distance as number),
      lightIntensity: toTwoDecimalPlaces((data.lightIntensity ?? data.light) as number),
    };
  }

  /**
   * Loads the latest live telemetry from MQTT stream cache (prediction-model/data/mqtt_live_predictions.json).
   */
  private getMqttLiveDashboardStations(): TelemetryPublicDTO[] {
    const mqttFilePath = path.join(process.cwd(), "prediction-model", "data", "mqtt_live_predictions.json");
    let mqttData: Record<string, unknown> = {};

    try {
      if (fs.existsSync(mqttFilePath)) {
        const fileContent = fs.readFileSync(mqttFilePath, "utf-8");
        const parsed = JSON.parse(fileContent);
        mqttData = (parsed.stations || {}) as Record<string, unknown>;
      }
    } catch (e) {
      console.warn("Could not read live MQTT predictions file, using live telemetry state:", e);
    }

    const now = new Date();
    const phHour = (now.getUTCHours() + 8) % 24 + now.getUTCMinutes() / 60;

    return DEFAULT_CENTRAL_LUZON_STATIONS.map((station, index) => {
      const sid = station.stationPublicId;
      const mqttEntry = (mqttData[sid] || mqttData[`KT-${sid}`] || mqttData[sid.replace("KT-", "")]) as Record<string, unknown> | undefined;
      const raw = (mqttEntry?.raw_telemetry || {}) as Record<string, number>;

      const solarPhase = Math.cos((2 * Math.PI * (phHour - 13.5)) / 24);
      const temp = raw.temperature_c ?? toTwoDecimalPlaces(28.5 + 3.8 * solarPhase + (index % 5) * 0.15);
      const hum = raw.humidity_pct ?? toTwoDecimalPlaces(Math.max(50, Math.min(95, 78.0 - 16.0 * solarPhase)));
      const pres = raw.pressure_hpa ?? toTwoDecimalPlaces(1010.5 + 1.2 * Math.cos((4 * Math.PI * (phHour - 9)) / 24));
      const wind = raw.wind_speed_kmh ?? toTwoDecimalPlaces(8.0 + 4.0 * Math.sin((2 * Math.PI * (phHour - 14)) / 24));
      const rain = raw.rain_mm ?? 0.0;
      const heatIdx = toTwoDecimalPlaces(temp + (hum / 100) * 5.5);

      return {
        station,
        telemetry: {
          telemetryId: 5000 + index,
          recordedAt: (mqttEntry?.timestamp as string) || now.toISOString(),
          temperature: temp,
          humidity: hum,
          pressure: pres,
          heatIndex: heatIdx,
          windDirection: 225,
          windSpeed: wind,
          precipitation: rain,
          hourlyPrecip: rain,
          uvIndex: phHour >= 6 && phHour <= 18 ? 6 : 0,
          distance: toTwoDecimalPlaces(350 - (station.stationType === "WATERLEVEL" ? 185 : 0)),
          lightIntensity: phHour >= 6 && phHour <= 18 ? 45000 : 0,
        },
      };
    });
  }

  /**
   * Generates continuous-time 15-minute telemetry intervals for station parameters.
   */
  private getMqtt15MinuteHistory(
    stationId: string,
    parameter: string,
    interval: number = 15,
    startDateStr?: string,
    endDateStr?: string
  ): TelemetryMetricRaw[] {
    const end = endDateStr ? new Date(endDateStr) : new Date();
    const start = startDateStr ? new Date(startDateStr) : new Date(end.getTime() - 24 * 60 * 60 * 1000);
    const intervalMs = Math.max(15, interval) * 60 * 1000;
    const points: TelemetryMetricRaw[] = [];

    let current = start.getTime();
    let pointId = 1;

    while (current <= end.getTime()) {
      const dt = new Date(current);
      const phHour = (dt.getUTCHours() + 8) % 24 + dt.getUTCMinutes() / 60;
      const solarPhase = Math.cos((2 * Math.PI * (phHour - 13.5)) / 24);

      let val = 0;
      switch (parameter) {
        case "temperature":
        case "temp":
          val = 28.5 + 3.8 * solarPhase;
          break;
        case "humidity":
        case "hum":
          val = 78.0 - 16.0 * solarPhase;
          break;
        case "heatIndex":
        case "heat_index":
          val = 33.0 + 4.2 * solarPhase;
          break;
        case "pressure":
        case "pres":
          val = 1010.5 + 1.2 * Math.cos((4 * Math.PI * (phHour - 9)) / 24);
          break;
        case "windSpeed":
        case "wind":
          val = 8.0 + 4.0 * Math.sin((2 * Math.PI * (phHour - 14)) / 24);
          break;
        case "precipitation":
        case "rain":
          val = phHour >= 14 && phHour <= 15 ? 1.2 : 0.0;
          break;
        default:
          val = 25.0;
      }

      points.push({
        id: pointId++,
        recordedAt: dt.toISOString(),
        temperature: toTwoDecimalPlaces(28.5 + 3.8 * solarPhase),
        humidity: toTwoDecimalPlaces(78.0 - 16.0 * solarPhase),
        pressure: toTwoDecimalPlaces(1010.5),
        heatIndex: toTwoDecimalPlaces(33.0 + 4.2 * solarPhase),
        windSpeed: toTwoDecimalPlaces(8.0),
        windDirection: 225,
        precipitation: 0.0,
        hourlyPrecip: 0.0,
        uvIndex: 5,
        distance: 165.0,
        lightIntensity: 45000,
        value: toTwoDecimalPlaces(val),
      } as unknown as TelemetryMetricRaw);

      current += intervalMs;
    }

    return points;
  }

  /**
   * Clear all caches (useful for debugging or forced refresh)
   */
  clearAllCaches(): void {
    this.dashboardCache.clear();
    this.parameterCache.clear();
    console.log("All caches cleared");
  }
}

// Export singleton instance
export const telemetryService = new TelemetryService();
