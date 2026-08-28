import { CACHE_CONFIG } from "@/lib/config/cache.config";
import {
  getWaterLevelDashboardFromKloudtrackApi,
  getWaterLevelMetricHistoryFromKloudtrackApi,
} from "@/lib/kloudtrack/client";
import stationIds from "@/lib/constants/stations.json";
import { InMemoryCache } from "@/lib/utils/cache";
import { toTwoDecimalPlaces } from "@/lib/utils/data-helper";
import { AppError } from "@/lib/utils/error";
import { StationPublicInfo } from "@/types/telemetry";
import { StationRaw } from "@/types/telemetry-raw";
import {
  WaterLevelHistoryDTO,
  WaterLevelHistoryMetricDataPoint,
  WaterLevelHistoryMetricDTO,
  WaterLevelMetrics,
  WaterLevelPublicDTO,
} from "@/types/water-level";
import {
  WaterLevelDashboardRaw,
  WaterLevelHistoryMetricPointRaw,
  WaterLevelMetricRaw,
  WaterLevelHistoryRaw,
} from "@/types/water-level-raw";

const WATER_LEVEL_STATION_CONFIG_BY_ID = new Map(
  stationIds.waterLevel.stationIdToFetch.map((station) => [station.stationId, station])
);

export const WATER_LEVEL_INTERVALS = [1, 15, 30, 60, 180, 360, 720, 1440] as const;

export const WATER_LEVEL_VARIABLES = [
  "sampleInterval",
  "sampleCount",
  "filteredSampleCount",
  "spikeCount",
  "minimum",
  "maximum",
  "rawMode",
  "calculatedWaterLevel",
  "median",
  "frequentRangeLow",
  "frequentRangeHigh",
  "estimatedMovAvg",
  "distance",
] as const;

export interface WaterLevelHistoryOptions {
  skip?: number;
  take?: number;
  interval?: number;
  startDate?: string;
  endDate?: string;
}

export class WaterLevelService {
  private dashboardCache = new InMemoryCache<WaterLevelPublicDTO[]>(
    CACHE_CONFIG.waterLevel.stationDashboard
  );
  private historyCache = new InMemoryCache<WaterLevelHistoryDTO>(
    CACHE_CONFIG.waterLevel.parameterHistory
  );
  private variableHistoryCache = new InMemoryCache<WaterLevelHistoryMetricDTO>(
    CACHE_CONFIG.waterLevel.parameterHistory
  );

  private ongoingDashboardRequest?: Promise<WaterLevelPublicDTO[]>;
  private ongoingHistoryRequests = new Map<string, Promise<WaterLevelHistoryDTO>>();
  private ongoingVariableHistoryRequests = new Map<
    string,
    Promise<WaterLevelHistoryMetricDTO>
  >();

  async getDashboardStations(): Promise<WaterLevelPublicDTO[]> {
    const cacheKey = "water-level-dashboard";
    const cached = this.dashboardCache.get(cacheKey);
    if (cached) return cached;

    if (this.ongoingDashboardRequest) {
      return this.ongoingDashboardRequest;
    }

    this.ongoingDashboardRequest = (async () => {
      try {
        const rawData = await getWaterLevelDashboardFromKloudtrackApi();
        const dashboardStations = this.transformDashboard(rawData);

        this.dashboardCache.set(
          cacheKey,
          dashboardStations,
          CACHE_CONFIG.waterLevel.stationDashboard
        );

        return dashboardStations;
      } catch (error) {
        console.error("[getWaterLevelDashboardStations] Failed to fetch water-level dashboard:", error);
        throw new AppError("Failed to fetch water-level dashboard", 500);
      } finally {
        this.ongoingDashboardRequest = undefined;
      }
    })();

    return this.ongoingDashboardRequest;
  }

  async getStationDashboardData(stationId: string): Promise<WaterLevelPublicDTO> {
    const stationData = (await this.getDashboardStations()).find(
      (item) => item.station.stationPublicId === stationId
    );

    if (!stationData) {
      throw new AppError(`Station ${stationId} was not found in water-level dashboard`, 404);
    }

    return stationData;
  }

  async getAllStations(): Promise<StationPublicInfo[]> {
    return (await this.getDashboardStations()).map((item) => item.station);
  }

  async getStationParameterHistory(
    stationId: string,
    parameter: string,
    options: WaterLevelHistoryOptions = {}
  ): Promise<WaterLevelHistoryMetricDTO> {
    this.assertValidParameter(parameter);
    this.assertValidInterval(options.interval);

    const upstreamParameter = this.toUpstreamParameter(parameter);
    const queryParams = this.buildQueryParams(options);
    const cacheKey = `water-level-parameter-history-${stationId}-${parameter}-${queryParams.toString()}`;
    const cached = this.variableHistoryCache.get(cacheKey);
    if (cached) return cached;

    const ongoingRequest = this.ongoingVariableHistoryRequests.get(cacheKey);
    if (ongoingRequest) return ongoingRequest;

    const request = (async () => {
      try {
        const rawData = await getWaterLevelMetricHistoryFromKloudtrackApi(
          stationId,
          upstreamParameter,
          Object.fromEntries(queryParams)
        );
        const result = {
          station: this.transformStation(rawData.station),
          waterLevel: this.sortVariableHistory(rawData.waterLevel || []),
        };
        this.variableHistoryCache.set(
          cacheKey,
          result,
          CACHE_CONFIG.waterLevel.parameterHistory
        );
        return result;
      } catch (error) {
        console.error(
          `[getStationWaterLevelParameterHistory] Failed for ${parameter} at station ${stationId}:`,
          error
        );
        throw new AppError(
          `Failed to fetch ${parameter} water-level history for station ${stationId}`,
          500
        );
      } finally {
        this.ongoingVariableHistoryRequests.delete(cacheKey);
      }
    })();

    this.ongoingVariableHistoryRequests.set(cacheKey, request);
    return request;
  }

  private transformDashboard(rawData: WaterLevelDashboardRaw): WaterLevelPublicDTO[] {
    const dashboardStations = rawData
      .filter((item) => Boolean(item?.station))
      .map((item) => ({
        station: this.transformStation(item.station),
        waterLevel: this.transformWaterLevel(item.waterLevel),
      }));

    const configuredStationIds = stationIds.waterLevel.stationIdToFetch.map((item) => item.stationId);
    if (configuredStationIds.length === 0) return dashboardStations;

    const dashboardByStationId = new Map(
      dashboardStations.map((item) => [item.station.stationPublicId, item])
    );

    return configuredStationIds
      .map((stationId) => dashboardByStationId.get(stationId))
      .filter((item): item is WaterLevelPublicDTO => Boolean(item));
  }

  private transformHistory(rawData: {
    station: StationRaw;
    waterLevel: WaterLevelMetricRaw[];
  }): WaterLevelHistoryDTO {
    return {
      station: this.transformStation(rawData.station),
      waterLevel: (rawData.waterLevel || [])
        .map((reading) => this.transformWaterLevel(reading))
        .filter((reading): reading is WaterLevelMetrics => Boolean(reading))
        .sort((a, b) => {
          const aTime = new Date(a.recordedAt).getTime();
          const bTime = new Date(b.recordedAt).getTime();
          return aTime - bTime;
        }),
    };
  }

  private transformStation(station: StationRaw): StationPublicInfo {
    const location = Array.isArray(station.location)
      ? station.location
      : [station.location?.lng ?? 0, station.location?.lat ?? 0];
    const stationConfig = WATER_LEVEL_STATION_CONFIG_BY_ID.get(station.id);

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
      referenceThreshold:
        typeof stationConfig?.referenceThreshold === "number"
          ? stationConfig.referenceThreshold
          : undefined,
    };
  }

  private transformWaterLevel(reading: WaterLevelMetricRaw | null | undefined): WaterLevelMetrics | null {
    if (!reading) return null;

    return {
      waterLevelId: reading.id,
      recordedAt: reading.recordedAt,
      startTimestamp: reading.startTimestamp ?? null,
      endTimestamp: reading.endTimestamp ?? null,
      sampleInterval: toTwoDecimalPlaces(reading.sampleInterval),
      sampleCount: toTwoDecimalPlaces(reading.sampleCount),
      filteredSampleCount: toTwoDecimalPlaces(reading.filteredSampleCount),
      spikeCount: toTwoDecimalPlaces(reading.spikeCount),
      minimum: toTwoDecimalPlaces(reading.minimum),
      maximum: toTwoDecimalPlaces(reading.maximum),
      rawMode: toTwoDecimalPlaces(reading.rawMode),
      calculatedWaterLevel: toTwoDecimalPlaces(reading.calculatedWaterLevel),
      median: toTwoDecimalPlaces(reading.median),
      frequentRangeLow: toTwoDecimalPlaces(reading.frequentRangeLow),
      frequentRangeHigh: toTwoDecimalPlaces(reading.frequentRangeHigh),
      estimatedMovAvg: toTwoDecimalPlaces(reading.estimatedMovAvg),
    };
  }

  private sortVariableHistory(
    readings: WaterLevelHistoryMetricPointRaw[]
  ): WaterLevelHistoryMetricDataPoint[] {
    return readings
      .map((reading) => ({
        id: reading.id,
        recordedAt: reading.recordedAt,
        createdAt: reading.createdAt,
        value: toTwoDecimalPlaces(reading.value),
      }))
      .sort((a, b) => {
        const aTime = new Date(a.recordedAt).getTime();
        const bTime = new Date(b.recordedAt).getTime();
        return aTime - bTime;
      });
  }

  private buildQueryParams(options: WaterLevelHistoryOptions): URLSearchParams {
    const params = new URLSearchParams();

    if (options.skip !== undefined) params.set("skip", options.skip.toString());
    if (options.take !== undefined) params.set("take", options.take.toString());
    if (options.interval !== undefined) params.set("interval", options.interval.toString());
    if (options.startDate) params.set("startDate", options.startDate);
    if (options.endDate) params.set("endDate", options.endDate);

    return params;
  }

  private assertValidInterval(interval?: number): void {
    if (interval === undefined) return;

    if (!WATER_LEVEL_INTERVALS.includes(interval as (typeof WATER_LEVEL_INTERVALS)[number])) {
      throw new AppError(
        `Interval must be one of: ${WATER_LEVEL_INTERVALS.join(", ")}`,
        400
      );
    }
  }

  private assertValidParameter(parameter: string): void {
    if (!WATER_LEVEL_VARIABLES.includes(parameter as (typeof WATER_LEVEL_VARIABLES)[number])) {
      throw new AppError(`Unsupported water-level parameter: ${parameter}`, 400);
    }
  }

  private toUpstreamParameter(parameter: string): string {
    return parameter === "calculatedWaterLevel" ? "distance" : parameter;
  }

  clearAllCaches(): void {
    this.dashboardCache.clear();
    this.historyCache.clear();
    this.variableHistoryCache.clear();
  }
}

export const waterLevelService = new WaterLevelService();
