"use client";

import {
  createContext,
  useContext,
  useMemo,
  type ReactNode,
} from "react";
import { useStationParameterData } from "@/hooks/useStationParameterData";
import type { TelemetryMetricRaw } from "@/types/telemetry-raw";

/**
 * Provides one shared precipitation-history request to the dashboard tree.
 * The provider fetches yesterday and today once, derives today's accumulated
 * precipitation, and lets cards and charts reuse the same request state.
 */

export interface StationPrecipitationHistory {
  data: TelemetryMetricRaw[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  startOfTodayISO: string;
  dayPrecipitation: number;
  enabled: boolean;
}

interface StationPrecipitationHistoryProviderProps {
  stationId: string | null | undefined;
  enabled: boolean;
  children: ReactNode;
}

const StationPrecipitationHistoryContext =
  createContext<StationPrecipitationHistory | null>(null);

/** Fetches and derives the precipitation state stored by the provider. */
const useStationPrecipitationHistoryData = (
  stationId: string | null | undefined,
  enabled: boolean,
): StationPrecipitationHistory => {
  // Freeze the chart window for this mount so rerenders do not restart the request.
  const dateRange = useMemo(() => {
    const now = new Date();
    // Compute midnight today in Philippine Standard Time (UTC+8)
    const phNow = new Date(now.getTime() + 8 * 60 * 60 * 1000);
    const startOfTodayUtc = new Date(
      Date.UTC(phNow.getUTCFullYear(), phNow.getUTCMonth(), phNow.getUTCDate(), 0, 0, 0) - 8 * 60 * 60 * 1000
    );
    const startOfYesterdayUtc = new Date(startOfTodayUtc.getTime() - 24 * 60 * 60 * 1000);
    const endOfTodayUtc = new Date(startOfTodayUtc.getTime() + 24 * 60 * 60 * 1000);

    return {
      startDate: startOfYesterdayUtc.toISOString(),
      endDate: endOfTodayUtc.toISOString(),
      startOfTodayISO: startOfTodayUtc.toISOString(),
    };
  }, []);

  const { data, loading, error, refetch } = useStationParameterData(
    stationId,
    "precipitation",
    {
      enabled,
      startDate: dateRange.startDate,
      endDate: dateRange.endDate,
    },
  );

  const dayPrecipitation = useMemo(() => {
    // The upstream time-series points represent 15-minute sampled rolling hourly rate (mm/h).
    // Numerical integration across 15-minute intervals (dt = 15/60 = 0.25 hours).
    const todayPoints = data.filter((point) => point.recordedAt >= dateRange.startOfTodayISO);
    if (todayPoints.length === 0) return 0;

    const integratedTotal = todayPoints.reduce((sum, pt) => {
      const val = Number.isFinite(pt.value) ? Number(pt.value) : 0;
      return sum + val * 0.25;
    }, 0);

    return Math.round(integratedTotal * 10) / 10;
  }, [data, dateRange.startOfTodayISO]);

  return useMemo(
    () => ({
      data,
      loading,
      error,
      refetch,
      startOfTodayISO: dateRange.startOfTodayISO,
      dayPrecipitation,
      enabled,
    }),
    [
      data,
      dateRange.startOfTodayISO,
      dayPrecipitation,
      enabled,
      error,
      loading,
      refetch,
    ],
  );
};

/** Makes precipitation history available without duplicate descendant requests. */
export const StationPrecipitationHistoryProvider = ({
  stationId,
  enabled,
  children,
}: StationPrecipitationHistoryProviderProps) => {
  const history = useStationPrecipitationHistoryData(stationId, enabled);

  return (
    <StationPrecipitationHistoryContext.Provider value={history}>
      {children}
    </StationPrecipitationHistoryContext.Provider>
  );
};

/** Reads the shared precipitation history and today's accumulated total. */
export const useStationPrecipitationHistory = (): StationPrecipitationHistory => {
  const history = useContext(StationPrecipitationHistoryContext);

  if (!history) {
    throw new Error(
      "useStationPrecipitationHistory must be used inside StationPrecipitationHistoryProvider",
    );
  }

  return history;
};
