import { useState, useEffect } from "react";
import type { TelemetryPublicDTO } from "@/types/telemetry";
import { CACHE_CONFIG } from "@/lib/config/cache.config";

/**
 * Owns the shared dashboard telemetry snapshot used by the selected station
 * and station map. It hydrates from server-provided data, refreshes on an
 * interval and window focus, and derives the currently selected station.
 */

const EMPTY_DASHBOARD_STATIONS: TelemetryPublicDTO[] = [];
const DASHBOARD_REFRESH_INTERVAL_MS =
  CACHE_CONFIG.telemetry.stationDashboardClientRefresh * 1000;

function findStationData(
  dashboardStations: TelemetryPublicDTO[],
  stationId: string
): TelemetryPublicDTO | null {
  return (
    dashboardStations.find(
      (item) => item.station.stationPublicId === stationId
    ) ?? null
  );
}

/** Returns dashboard telemetry plus the selected station's current reading. */
export function useStationData(
  stationId: string,
  initialDashboardStations: TelemetryPublicDTO[] = EMPTY_DASHBOARD_STATIONS
) {
  const hasInitialDashboardStations = initialDashboardStations.length > 0;

  // Keep one shared dashboard snapshot for the selected station, map, and refresh loop.
  const [dashboardStations, setDashboardStations] = useState<TelemetryPublicDTO[]>(
    initialDashboardStations
  );
  const [data, setData] = useState<TelemetryPublicDTO | null>(() =>
    stationId ? findStationData(initialDashboardStations, stationId) : null
  );
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!stationId) {
      setData(null);
      return;
    }

    setData(findStationData(dashboardStations, stationId));
  }, [dashboardStations, stationId]);

  useEffect(() => {
    let isMounted = true;
    let isFetching = false;

    const fetchData = async () => {
      // Avoid overlapping dashboard refreshes from interval/focus events.
      if (isFetching) return;
      isFetching = true;
      setIsRefreshing(true);
      setError(null);

      try {
        const response = await fetch("/api/telemetry/dashboard", {
          cache: "default",
        });
        const result = await response.json();

        if (isMounted && result.success && Array.isArray(result.data)) {
          setDashboardStations(result.data as TelemetryPublicDTO[]);
        } else if (isMounted) {
          setError(result.message || "Failed to load station data");
        }
      } catch (error) {
        console.error(`Failed to fetch station data:`, error);
        if (isMounted) {
          setError("Failed to load station data");
        }
      } finally {
        isFetching = false;
        if (isMounted) setIsRefreshing(false);
      }
    };

    if (!hasInitialDashboardStations) {
      void fetchData();
    }

    const interval = setInterval(fetchData, DASHBOARD_REFRESH_INTERVAL_MS);

    // Refresh when the user returns to the tab, while still respecting server cache headers.
    const handleFocus = () => {
      void fetchData();
    };
    window.addEventListener("focus", handleFocus);

    return () => {
      isMounted = false;
      clearInterval(interval);
      window.removeEventListener("focus", handleFocus);
    };
  }, [hasInitialDashboardStations]);

  return { data, dashboardStations, isRefreshing, error };
}
