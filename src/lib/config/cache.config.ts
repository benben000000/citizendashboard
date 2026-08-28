// Cache durations in seconds
export const CACHE_CONFIG = {
  // Telemetry data caches
  telemetry: {
    stationDashboard: 300, // 5 minutes - Dashboard telemetry data
    stationDashboardClientRefresh: 60, // 1 minute - Client polling cadence
    parameterHistory: 600, // 10 minutes - Parameter-specific history data
  },

  waterLevel: {
    stationDashboard: 300,
    stationDashboardClientRefresh: 60,
    parameterHistory: 600,
  },

  // Next.js API route revalidation
  apiRoutes: {
    telemetry: 0,
    telemetryParameter: 0,
    waterLevel: 0,
    waterLevelParameter: 0,
    insights: 0,
  },

  // HTTP Cache-Control headers
  http: {
    telemetry: {
      sMaxAge: 60,
      staleWhileRevalidate: 30,
    },
    waterLevel: {
      sMaxAge: 60,
      staleWhileRevalidate: 30,
    },
    insights: {
      sMaxAge: 300,
      staleWhileRevalidate: 60,
    },
  },

  // Next.js fetch cache (for external API calls)
  fetch: {
    kloudtrackApi: 0,
  },
} as const;

/**
 * Helper function to get Cache-Control header value
 */
export function getCacheControlHeader(
  sMaxAge: number,
  staleWhileRevalidate?: number
): string {
  if (staleWhileRevalidate != undefined) {
    return `public, s-maxage=${sMaxAge}, stale-while-revalidate=${staleWhileRevalidate}`;
  }
  return `public, s-maxage=${sMaxAge}`;
}
