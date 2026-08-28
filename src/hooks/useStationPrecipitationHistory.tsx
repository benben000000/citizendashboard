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
    const startOfToday = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate(),
      0,
      0,
      0,
      0,
    );
    const startOfYesterday = new Date(startOfToday.getTime() - 24 * 60 * 60 * 1000);
    const endOfToday = new Date(startOfToday.getTime() + 24 * 60 * 60 * 1000);

    return {
      startDate: startOfYesterday.toISOString(),
      endDate: endOfToday.toISOString(),
      startOfTodayISO: startOfToday.toISOString(),
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

  const dayPrecipitation = useMemo(
    () =>
      data.reduce((total, point) => {
        if (point.recordedAt < dateRange.startOfTodayISO) return total;
        return total + (Number.isFinite(point.value) ? point.value : 0);
      }, 0),
    [data, dateRange.startOfTodayISO],
  );

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
