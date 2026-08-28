import { useState, useEffect, useCallback, useRef } from "react";
import { CACHE_CONFIG } from "@/lib/config/cache.config";
import { PARAMETERS } from "@/lib/constants/parameters";
import type { ParameterType } from "@/types/parameter";
import type { TelemetryMetricRaw } from "@/types/telemetry-raw";

/**
 * Fetches one station parameter's historical series for a requested date
 * window. It handles polling, retries, cancellation, and stale-response
 * protection when users quickly switch stations or metrics.
 */

interface UseStationParameterDataOptions {
  enabled?: boolean;
  pollIntervalMs?: number;
  /**
   * ISO string. When provided, appended as ?startDate=... to the API call.
   * Use this to fetch more than the default window (e.g. 48 hrs for yesterday+today).
   */
  startDate?: string;
  /**
   * ISO string. When provided, appended as ?endDate=... to the API call.
   * Use this to cap the response to a fixed window.
   */
  endDate?: string;
}

interface UseStationParameterDataResult {
  data: TelemetryMetricRaw[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/** Returns a historical metric series and its request state. */
export function useStationParameterData(
  stationId: string | null | undefined,
  parameterKey: ParameterType | null | undefined,
  options: UseStationParameterDataOptions = {}
): UseStationParameterDataResult {
  const {
    enabled = true,
    pollIntervalMs = CACHE_CONFIG.telemetry.parameterHistory * 1000,
    startDate,
    endDate,
  } = options;

  const [data, setData] = useState<TelemetryMetricRaw[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track the active request so quick station/metric changes cannot commit stale data.
  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  // Fetch one parameter history series for the selected station and date window.
  const fetchData = useCallback(
    async (opts?: { silent?: boolean }) => {
      if (!enabled || !stationId || !parameterKey) return;

      // Resolve the public metric key to the API parameter name before building the URL.
      const parameter = PARAMETERS.find((p) => p.key === parameterKey);
      if (!parameter) {
        setError("Unsupported parameter for historical data.");
        return;
      }

      if (!opts?.silent) {
        setLoading(true);
        setError(null);
      }

      // Cancel the previous request and tag this one as the newest request.
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const requestId = ++requestIdRef.current;

      try {
        // Build the API URL with optional server-side history bounds.
        const url = new URL(
          `/api/telemetry/station/${stationId}/parameter/${parameter.apiKey}`,
          window.location.origin
        );
        if (startDate) {
          url.searchParams.set("startDate", startDate);
        }
        if (endDate) {
          url.searchParams.set("endDate", endDate);
        }

        const response = await fetch(url.toString(), {
          cache: "default",
          signal: controller.signal,
        });
        const result = await response.json();

        // Ignore an older response if a newer request started while this was in flight.
        if (requestId !== requestIdRef.current) return;

        if (result.success && result.data) {
          setData(result.data as TelemetryMetricRaw[]);
        } else {
          throw new Error(result.message || "Failed to fetch parameter data");
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        console.error(`Failed to fetch ${parameterKey} data:`, err);
        setError(`Failed to load ${parameter.label} data. Please try again.`);
      } finally {
        // Only clear controller/loading state for the request that is still current.
        if (abortRef.current === controller) {
          abortRef.current = null;
        }

        if (requestId === requestIdRef.current) {
          setLoading(false);
        }
      }
    },
    [enabled, stationId, parameterKey, startDate, endDate]
  );

  // Reset visible data when the station or metric changes, then load fresh data.
  useEffect(() => {
    setData([]);
    setError(null);
    if (!enabled || !stationId || !parameterKey) return;
    fetchData();
  }, [enabled, stationId, parameterKey, fetchData]);

  // Keep the chart refreshed using the same interval as the server cache TTL.
  useEffect(() => {
    if (!enabled || !stationId || !parameterKey || !pollIntervalMs) return;
    const intervalId = setInterval(() => {
      fetchData({ silent: true });
    }, pollIntervalMs);
    return () => clearInterval(intervalId);
  }, [enabled, stationId, parameterKey, pollIntervalMs, fetchData]);

  // Abort any in-flight request when the component using this hook unmounts.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // Expose a manual retry path for error UI.
  const refetch = useCallback(async () => {
    await fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch };
}
