/**
 * Server-side client for communicating with Kloudtrack API
 * This runs ONLY on the server and includes the secret API token
 */
import { DashboardRaw, TelemetryHistoryMetricRaw, TelemetryHistoryTakeRaw } from "@/types/telemetry-raw";
import {
  WaterLevelDashboardRaw,
  WaterLevelHistoryRaw,
  WaterLevelHistoryMetricRaw,
} from "@/types/water-level-raw";

const KLOUDTRACK_API_BASE_URL = process.env.KLOUDTRACK_API_BASE_URL || "http://citizen.kloudtechsea.com/api";

const KLOUDTRACK_API_TOKEN = process.env.KLOUDTRACK_API_TOKEN;

interface KloudtrackApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

class KloudtrackApiClient {
  private baseURL: string;
  private apiToken: string | undefined;

  constructor(baseURL: string | undefined, apiToken: string | undefined) {
    this.baseURL = baseURL || "";
    this.apiToken = apiToken;
  }

  /**
   * Make authenticated request to Kloudtrack API
   */
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    if (typeof window !== "undefined") {
      // Prevent fetch in browser / static build
      throw new Error("Kloudtrack API requests must be called from the server");
    }
    const url = `${this.baseURL}${endpoint}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options?.headers as Record<string, string>),
    };

    // Add authorization header if token is available
    if (this.apiToken) {
      headers["x-kloudtrack-key"] = `${this.apiToken}`;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: options?.signal ?? controller.signal,
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Kloudtrack API Error: ${response.status} ${response.statusText}`);
      }

      const apiResponse = (await response.json()) as KloudtrackApiResponse<T>;

      if (!apiResponse.success) {
        throw new Error(apiResponse.message || "Kloudtrack API request failed");
      }
      return apiResponse.data;
    } catch (error) {
      console.error("Kloudtrack API request failed:", error);
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: "GET" });
  }

  async post<T>(endpoint: string, data: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async put<T>(endpoint: string, data: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: "DELETE" });
  }
}

// Export singleton instance
export const kloudtrackApi = new KloudtrackApiClient(KLOUDTRACK_API_BASE_URL, KLOUDTRACK_API_TOKEN);

// Export specific API methods
export async function getDashboardData(): Promise<DashboardRaw> {
  return kloudtrackApi.get<DashboardRaw>(`/telemetry/dashboard`);
}

export async function getLatestTelemetryFromKloudtrackApi(stationId: string): Promise<TelemetryHistoryTakeRaw> {
  return kloudtrackApi.get<TelemetryHistoryTakeRaw>(`/telemetry/station/${stationId}/history?take=1`);
}

export async function getTelemetryMetricHistoryFromKloudtrackApi(stationId: string, parameter: string, params: Record<string, string>): Promise<TelemetryHistoryMetricRaw> {
  const queryString = new URLSearchParams(params).toString();
  return kloudtrackApi.get<TelemetryHistoryMetricRaw>(`/telemetry/station/${stationId}/history/${parameter}?${queryString}`);
}




export async function getWaterLevelDashboardFromKloudtrackApi(): Promise<WaterLevelDashboardRaw> {
  return kloudtrackApi.get<WaterLevelDashboardRaw>("/water-level/dashboard");
}

export async function getWaterLevelMetricHistoryFromKloudtrackApi(
  stationId: string,
  variable: string,
  params: Record<string, string>
): Promise<WaterLevelHistoryMetricRaw> {
  const queryString = new URLSearchParams(params).toString();
  const suffix = queryString ? `?${queryString}` : "";
  return kloudtrackApi.get<WaterLevelHistoryMetricRaw>(`/water-level/station/${stationId}/history/${variable}${suffix}`);
}
