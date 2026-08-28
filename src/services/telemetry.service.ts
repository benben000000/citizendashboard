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
import { CACHE_CONFIG } from "@/lib/config/cache.config";

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
        console.error("[getDashboardStations] Failed to fetch dashboard telemetry:", error);
        throw new AppError("Failed to fetch dashboard telemetry", 500);
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
    const interval = options.interval ?? 60;
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
        console.error(`Failed to fetch ${parameter} history for station ${stationId}:`, error);
        throw new AppError(`Failed to fetch ${parameter} history for station ${stationId}`, 500);
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
