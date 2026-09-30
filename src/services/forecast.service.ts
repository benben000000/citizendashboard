/**
 * Operational Forecast Service — reads the validated Python model's output.
 *
 * AUTHORITATIVE SOURCE
 * --------------------
 * `prediction-model/data/mqtt_live_predictions.json` is written by
 * `prediction-model/src/mqtt_live_ingestor.py`, which runs the production
 * `GarciaWeatherLNN` checkpoint from `prediction-model/data/bundles/h{1,3,6,12,24}/`
 * under the frozen `inference_policy.json`. That is the model whose behaviour is
 * documented in `prediction-model/MODEL_REGISTRY.md`.
 *
 * This service replaces the previous in-TypeScript reimplementation of the LNN
 * (`prediction.service.ts::computeLnnMultiHorizonForecast`), which used a
 * different input dimensionality (4 vs 8), a different hidden size (8 vs 32),
 * hardcoded normalisation constants, and weights that matched no artifact in
 * the repository. No audited scorecard applied to it.
 *
 * DESIGN RULES
 * ------------
 *  1. FAIL CLOSED. If the model output is missing or stale, this returns an
 *     explicit unavailable result. It never synthesises weather.
 *  2. PRESERVE GOVERNANCE. The model's `model_status`, `not_for_life_safety`,
 *     `input_quality_gate`, blocked-target statuses, and provenance hashes are
 *     carried through to the UI. They were previously discarded at the service
 *     boundary, which is how imperative flood copy ended up attached to a model
 *     explicitly marked as not for life safety.
 *  3. NO INVENTED FIELDS. No radar, no Doppler, no convective buoyancy, no
 *     fabricated confidence percentages.
 */

import fs from "fs";
import path from "path";

import { resolveDeviceId } from "@/lib/constants/mqtt-station-map";

/** Default staleness ceiling. The ingestor runs on a short cadence. */
const DEFAULT_MAX_AGE_SECONDS = Number(
  process.env.FORECAST_MAX_AGE_SECONDS ?? process.env.MQTT_STATUS_MAX_AGE_SECONDS ?? 600
);

const CACHE_FILE = path.join(
  process.cwd(),
  "prediction-model",
  "data",
  "mqtt_live_predictions.json"
);

/* -------------------------------------------------------------------------- */
/* Types mirroring prediction-model/src/inference.py output contract          */
/* -------------------------------------------------------------------------- */

export type HeatIndexRiskCategory =
  | "NO_RISK"
  | "CAUTION"
  | "EXTREME CAUTION"
  | "DANGER"
  | "EXTREME DANGER";

export type PressureTendency = "RISING" | "STEADY" | "FALLING";

export type BlockedTargetStatus = "BLOCKED_BY_SENSOR_CALIBRATION" | "SECONDARY_BETA_DAYLIGHT_ONLY";

export interface InputQualityGate {
  learned_output_trusted: boolean;
  sensor_quality_flag: string;
  policy: string;
}

export interface OperationalForecast {
  api_mode: string;
  product_name: string;
  bundle_version: string;
  active_bundle_horizon: string;
  active_bundle_path: string | null;
  model_version: string;
  model_seed: string | number;
  /** e.g. RESEARCH_PROTOTYPE | CANDIDATE_RESEARCH */
  model_status: string;
  not_for_life_safety: boolean;
  forecast_origin_timestamp: string;
  target_timestamp: string | null;
  forecast_horizon: string;

  temperature_c: number;
  relative_humidity_pct: number;
  pressure_hpa: number;
  pressure_tendency: PressureTendency;
  wind_speed_kmh: number;
  wind_direction_deg: number | null;
  wind_calm: boolean;
  heat_index_c: number;
  heat_index_risk_category: HeatIndexRiskCategory;
  chance_of_rain_pct: number;
  expected_precipitation_mm: number;

  uv_index: { status: BlockedTargetStatus; value: null; reason: string };
  light_intensity: { status: BlockedTargetStatus; value: null; reason: string };
  anomaly_status: {
    has_anomaly: boolean;
    sensor_quality_flag: string;
    total_anomaly_count: number;
    summary: string;
    actionable: boolean;
  } | null;
  input_quality_gate?: InputQualityGate;

  selected_source_by_variable: Record<string, string>;
  rain_probability_source: string;
  rain_model_weight: number;
  rain_persistence_weight: number;
  rain_operational_threshold: number;
  rain_operational_alert: boolean;

  policy_version: string;
  policy_code_commit: string;
  model_code_commit: string;
  provenance: {
    implementation_commit?: string;
    artifact_commit?: string;
    model_weights_commit?: string;
    bundle_version?: string;
    checkpoint_sha256?: string;
    policy_sha256?: string;
  };
  weather_uncertainty: { status: string; reason: string };
  diagnostics: {
    raw_learned_predictions: Record<string, number | null>;
    persistence_observations: Record<string, number | null>;
  };
  water_level_beta: {
    predicted_water_level_m: number;
    status: string;
    not_for_life_safety: boolean;
  };
}

/**
 * Per-station entry as written by `mqtt_live_ingestor.py`.
 *
 * The forecast is nested under `operational_prediction.forecast`, not at the top
 * level of the station entry. `prediction_status` distinguishes
 * READY / WAITING_FOR_COMPLETE_SEQUENCE / ERROR so an absent forecast can be
 * reported with its actual cause instead of a generic message.
 */
export interface OperationalForecastStation {
  station_id: string;
  topic?: string;
  timestamp: string;
  qc_status: string;
  raw_telemetry: {
    temperature_c: number | null;
    humidity_pct: number | null;
    pressure_hpa: number | null;
    wind_speed_kmh: number | null;
    wind_direction_deg: number | null;
    water_level_m: number | null;
    rain_mm: number | null;
  };
  operational_prediction?: {
    generated_at: string;
    source: string;
    sequence_length: number;
    max_sequence_gap_seconds?: number;
    forecast: OperationalForecast;
  };
  /**
   * One entry per published horizon ("1h", "3h", "6h", "12h", "24h"). Each
   * horizon has its own trained bundle, so a horizon is only ever served from
   * the model that was actually trained for it.
   */
  operational_predictions?: Record<
    string,
    {
      generated_at: string;
      source: string;
      horizon?: string;
      sequence_length: number;
      max_sequence_gap_seconds?: number;
      forecast: OperationalForecast;
    }
  >;
  published_horizons?: string[];
  prediction_errors_by_horizon?: Record<string, string>;
  prediction_status?: string;
  prediction_error?: string;
  /**
   * Flattened view of `operational_prediction.forecast`, populated by
   * `getOperationalForecast` on the success path only. Null/undefined means no
   * validated forecast exists — never treat that as a benign forecast.
   */
  forecast?: OperationalForecast;
}

export type ForecastUnavailableReason =
  | "CACHE_FILE_MISSING"
  | "CACHE_UNREADABLE"
  | "STATION_NOT_RESOLVED"
  | "NO_FORECAST_FOR_STATION"
  | "STALE"
  | "HORIZON_MISMATCH";

export interface ForecastAvailable {
  available: true;
  station: OperationalForecastStation;
  ageSeconds: number;
  lastUpdated: string;
}

export interface ForecastUnavailable {
  available: false;
  reason: ForecastUnavailableReason;
  message: string;
  ageSeconds: number | null;
  lastUpdated: string | null;
}

export type ForecastResult = ForecastAvailable | ForecastUnavailable;

/* -------------------------------------------------------------------------- */
/* Cache access                                                               */
/* -------------------------------------------------------------------------- */

interface MqttCacheShape {
  last_updated?: string;
  total_active_stations?: number;
  stations?: Record<string, OperationalForecastStation>;
}

let readCache: (() => MqttCacheShape | null) | null = null;
let cachedCache: MqttCacheShape | null = null;
let cachedCacheMtimeMs = 0;

function loadCache(): MqttCacheShape | null {
  try {
    const stat = fs.statSync(CACHE_FILE);
    if (cachedCache && stat.mtimeMs === cachedCacheMtimeMs) {
      return cachedCache;
    }
    const parsed = JSON.parse(fs.readFileSync(CACHE_FILE, "utf-8")) as MqttCacheShape;
    cachedCache = parsed;
    cachedCacheMtimeMs = stat.mtimeMs;
    return parsed;
  } catch {
    return null;
  }
}

function computeAgeSeconds(isoTimestamp: string | undefined | null): number | null {
  if (!isoTimestamp) return null;
  const ms = Date.parse(isoTimestamp);
  if (!Number.isFinite(ms)) return null;
  return Math.max(0, Math.floor((Date.now() - ms) / 1000));
}

/**
 * Fetch the validated operational forecast for a station.
 *
 * @param stationPublicId Dashboard station id (Kloudtrack public id or KT-* device id)
 * @param stationName     Dashboard display name, used to resolve the device id
 * @param horizon         Requested horizon, e.g. "1h". The live ingestor currently
 *                        publishes the 1h bundle only; other horizons return
 *                        HORIZON_MISMATCH rather than a substituted value.
 */
export function getOperationalForecast(
  stationPublicId: string | null | undefined,
  stationName: string | null | undefined,
  horizon: string = "1h"
): ForecastResult {
  const cache = loadCache();

  if (!cache) {
    const exists = fs.existsSync(CACHE_FILE);
    return {
      available: false,
      reason: exists ? "CACHE_UNREADABLE" : "CACHE_FILE_MISSING",
      message: exists
        ? "The live model cache exists but could not be parsed. No forecast is available."
        : "The live model cache is not present. The forecast ingestor is not running.",
      ageSeconds: null,
      lastUpdated: null,
    };
  }

  const lastUpdated = cache.last_updated ?? null;
  const stations = cache.stations ?? {};

  const deviceId = resolveDeviceId(stationPublicId, stationName);
  if (!deviceId) {
    return {
      available: false,
      reason: "STATION_NOT_RESOLVED",
      message:
        "This station is not present in the live MQTT telemetry fleet, so no model " +
        "forecast exists for it. Showing current observations only.",
      ageSeconds: computeAgeSeconds(lastUpdated),
      lastUpdated,
    };
  }

  const entry = stations[deviceId];
  if (!entry) {
    return {
      available: false,
      reason: "NO_FORECAST_FOR_STATION",
      message:
        "The station is in the telemetry fleet but has not yet produced a model " +
        "forecast. Showing current observations only.",
      ageSeconds: computeAgeSeconds(lastUpdated),
      lastUpdated,
    };
  }

  // Staleness gate. A stale forecast must not be presented as current.
  //
  // Freshness is judged on THIS STATION's own observation time, NOT on the cache
  // file's `last_updated`. `last_updated` moves every time ANY station reports,
  // so a station that went offline hours ago keeps passing a file-level check
  // while its peers are healthy. Measured directly: 7 of 14 cached stations
  // still carried forecasts from a superseded policy while others were current.
  // Resolve the requested horizon BEFORE the staleness gate. Freshness must be
  // judged on the entry actually being served: a station whose 24h bundle is
  // current while its 1h is not would otherwise have both rejected.
  const byHorizon = entry.operational_predictions ?? undefined;
  const key = horizon ?? "1h";
  const selected =
    (byHorizon ? byHorizon[key] : undefined) ??
    // Backward compatibility with a cache written before multi-horizon: the
    // legacy field IS the 1h entry, so only fall back to it for the 1h request.
    (key === "1h" ? entry.operational_prediction : undefined);

  const forecastTs = selected?.generated_at ?? entry.timestamp;
  const ageSeconds = computeAgeSeconds(forecastTs);
  if (ageSeconds === null || ageSeconds > DEFAULT_MAX_AGE_SECONDS) {
    return {
      available: false,
      reason: "STALE",
      message:
        `This station's last model forecast is ${ageSeconds ?? "unknown"} seconds old, ` +
        `beyond the ${DEFAULT_MAX_AGE_SECONDS}s freshness limit. Showing current ` +
        `observations only.`,
      ageSeconds,
      lastUpdated: forecastTs,
    };
  }

  const forecast = selected?.forecast ?? null;

  if (!forecast) {
    // A horizon that exists but was not produced is a different situation from
    // the station having no forecast at all. Name what IS available, so the
    // reader is not sent to current observations for a request that simply was
    // not served.
    const published = entry.published_horizons ?? Object.keys(entry.operational_predictions ?? {});
    const perHorizonError = entry.prediction_errors_by_horizon?.[key];
    if (published.length > 0 || perHorizonError) {
      return {
        available: false,
        reason: perHorizonError ? "NO_FORECAST_FOR_STATION" : "HORIZON_MISMATCH",
        message: perHorizonError
          ? `The ${key} forecast could not be produced: ${perHorizonError}. ` +
            `Other horizons (${published.join(", ") || "none"}) are unaffected.`
          : `This station publishes ${published.join(", ")}; a ${key} forecast is not ` +
            `available. Choose one of the published horizons.`,
        ageSeconds,
        lastUpdated: forecastTs,
      };
    }
    // Report the ingestor's actual reason rather than a generic message.
    const status = entry.prediction_status ?? "UNKNOWN";
    const detail =
      status === "WAITING_FOR_COMPLETE_SEQUENCE"
        ? "Telemetry is arriving but the station has not yet produced a full " +
          `${selected?.sequence_length ?? 24}-sample sequence.`
        : status === "ERROR"
          ? `The ingestor reported an error: ${
              entry.prediction_errors_by_horizon?.[key] ??
              entry.prediction_error ??
              "no detail provided"
            }`
          : "Telemetry is being received but no forecast has been generated yet.";
    return {
      available: false,
      reason: "NO_FORECAST_FOR_STATION",
      message: `${detail} Showing current observations only.`,
      ageSeconds,
      lastUpdated,
    };
  }

  // Defence in depth. The horizon was already resolved above, so reaching here
  // with a mismatch means an entry is mislabelled or the cache is inconsistent.
  // Never serve one horizon's numbers under another horizon's label.
  if (horizon && forecast.forecast_horizon !== horizon) {
    return {
      available: false,
      reason: "HORIZON_MISMATCH",
      message:
        `The entry for ${horizon} is labelled ${forecast.forecast_horizon}. ` +
        `Refusing to serve one horizon's numbers under another's label.`,
      ageSeconds,
      lastUpdated,
    };
  }

  return {
    available: true,
    station: { ...entry, forecast },
    ageSeconds,
    lastUpdated: forecastTs,
  };
}

/** Freshness budget currently enforced, in seconds. */
export const FORECAST_FRESHNESS_BUDGET_SECONDS = DEFAULT_MAX_AGE_SECONDS;

/** Test seam: forces a cache re-read. */
export function __resetForecastCache(): void {
  cachedCache = null;
  cachedCacheMtimeMs = 0;
  readCache = null;
}
