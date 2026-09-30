/**
 * expected-accuracy.reference.ts
 *
 * The "expected accuracy" half of the provenance panel: a static, offline
 * benchmark of the served model, per variable, per horizon.
 *
 * ── WHY THIS IS A SEPARATE, LABELLED THING ─────────────────────────────────
 * It is NOT measured error. Nothing here has been checked against a live
 * observation. It is the offline holdout score of the model bundle, quoted so an
 * operator can judge whether a number looks plausible -- and the panel keeps it
 * visually and structurally apart from verified accuracy, which is scored
 * against what actually happened.
 *
 * The single most important property of this file is what it does NOT contain:
 *
 *   NO POOLED NUMBER. There is no overall MAE here, and one must never be added.
 *   An average over degC, %RH, hPa and m/s is not a quantity: the scales do not
 *   share a unit, so an hPa error numerically dominates a degC error while
 *   meaning no more. This mirrors the deliberate absence of a pooled metric in
 *   `prediction-model/src/verify_predictions.py`.
 *
 * ── HONESTY RULES FOR THIS FILE ─────────────────────────────────────────────
 *  - `value: null` + `status: "NOT_MEASURED"` means nobody has a number for that
 *    cell. Do not interpolate between horizons to fill one in.
 *  - `status: "PERSISTENCE_EQUIVALENT"` is a measurement, not a missing value:
 *    the model was evaluated and matched the persistence baseline. It is
 *    reported with `beatsPersistencePct: null` rather than 0, because "no skill
 *    demonstrated" and "not yet looked at" are different findings.
 *  - Only horizons with a released figure appear. The live policy also serves
 *    3h and 12h rain; those cells are absent rather than guessed.
 *
 * ── PROVENANCE OF THESE FIGURES ─────────────────────────────────────────────
 * Measured on the held-out split of the committed telemetry corpus
 * (`weather_telemetry.csv`, sha256 86ce9064…bea4a) against the
 * `GarciaWeatherLNN` bundles under `prediction-model/data/bundles/`, versus a
 * persistence baseline evaluated on the same split and same origins.
 * Re-derive with the training/reporting commands in
 * `prediction-model/docs/` before treating these as current; they are pinned
 * here so the panel does not silently drift away from the model in production.
 */

import type {
  ExpectedAccuracyCell,
  ExpectedAccuracySection,
  VariableName,
} from "./prediction-provenance-core";

export const EXPECTED_ACCURACY_SOURCE = "offline holdout split, GarciaWeatherLNN bundles vs persistence";
export const EXPECTED_ACCURACY_AS_OF = "2026-09-23";
export const EXPECTED_ACCURACY_CORPUS_SHA256_PREFIX = "86ce906453aa";

/**
 * Rain occurrence carries extra baselines the other variables do not. Brier is
 * unitless, so it is the one metric here that is legitimately comparable to
 * another occurrence-based score -- but it is still never pooled with an MAE.
 */
const RAIN_BEATS = ["persistence", "climatology", "all 7 NWP models"] as const;

function measured(
  horizonHours: number,
  value: number,
  beatsPersistencePct: number | null,
  beatsBaselines: readonly string[] = ["persistence"]
): ExpectedAccuracyCell {
  return {
    horizonHours,
    metric: "mae",
    value,
    beatsPersistencePct,
    status: "MEASURED",
    beatsBaselines,
  };
}

function equivalent(horizonHours: number): ExpectedAccuracyCell {
  return {
    horizonHours,
    metric: "mae",
    value: null,
    beatsPersistencePct: null,
    status: "PERSISTENCE_EQUIVALENT",
    beatsBaselines: [],
  };
}

function notMeasured(horizonHours: number): ExpectedAccuracyCell {
  return {
    horizonHours,
    metric: "mae",
    value: null,
    beatsPersistencePct: null,
    status: "NOT_MEASURED",
    beatsBaselines: [],
  };
}

function brierCell(
  horizonHours: number,
  value: number,
  beatsBaselines: readonly string[]
): ExpectedAccuracyCell {
  return {
    horizonHours,
    metric: "brier",
    value,
    beatsPersistencePct: null,
    status: "MEASURED",
    beatsBaselines,
  };
}

/**
 * Temperature, degC.
 * +6h 1.477 (+13.0% vs persistence), +12h 1.647 (+23.8% vs persistence).
 * 1h / 3h / 24h are persistence-equivalent.
 */
const TEMPERATURE_CELLS: ExpectedAccuracyCell[] = [
  equivalent(1),
  equivalent(3),
  measured(6, 1.477, 13.0),
  measured(12, 1.647, 23.8),
  equivalent(24),
];

/**
 * Wind speed, m/s (converted from the km/h the audit trail stores; see
 * VARIABLE_UNITS.wind_speed).
 * +6h 1.217 (+7.8%), +12h 1.243 (+17.8%).
 */
const WIND_SPEED_CELLS: ExpectedAccuracyCell[] = [
  notMeasured(1),
  notMeasured(3),
  measured(6, 1.217, 7.8),
  measured(12, 1.243, 17.8),
  notMeasured(24),
];

/** Humidity and pressure: persistence-equivalent at every served horizon. */
const HUMIDITY_CELLS: ExpectedAccuracyCell[] = [1, 3, 6, 12, 24].map(equivalent);
const PRESSURE_CELLS: ExpectedAccuracyCell[] = [1, 3, 6, 12, 24].map(equivalent);

/**
 * Rain occurrence, Brier score (unitless, lower is better).
 * 0.137 at +1h, 0.251 at +24h, beating persistence, climatology and all seven
 * NWP models. Intermediate horizons were not released and are absent rather
 * than interpolated.
 */
const RAIN_CELLS: ExpectedAccuracyCell[] = [
  brierCell(1, 0.137, RAIN_BEATS),
  brierCell(24, 0.251, RAIN_BEATS),
];

const VARIABLE_CELLS: Record<string, ExpectedAccuracyCell[]> = {
  temperature: TEMPERATURE_CELLS,
  wind_speed: WIND_SPEED_CELLS,
  humidity: HUMIDITY_CELLS,
  pressure: PRESSURE_CELLS,
  rain_occurrence: RAIN_CELLS,
};

/**
 * Assemble the static section, optionally overlaid with values served by the
 * endpoint. Anything the endpoint supplies for a variable replaces the pinned
 * reference wholesale, so the two can never disagree cell by cell.
 */
export function buildExpectedAccuracySection(
  override?: ExpectedAccuracySection | null
): ExpectedAccuracySection {
  if (override && override.variables && Object.keys(override.variables).length > 0) {
    return {
      source: override.source || EXPECTED_ACCURACY_SOURCE,
      asOf: override.asOf ?? EXPECTED_ACCURACY_AS_OF,
      coverageNote: override.coverageNote ?? null,
      variables: override.variables,
    };
  }

  return {
    source: EXPECTED_ACCURACY_SOURCE,
    asOf: EXPECTED_ACCURACY_AS_OF,
    coverageNote:
      "Horizons with no released holdout figure are listed as NOT_MEASURED rather than interpolated. " +
      "PERSISTENCE_EQUIVALENT means the model was evaluated and matched the persistence baseline.",
    variables: VARIABLE_CELLS,
  };
}

/**
 * Which audited variables the reference actually speaks to, so the panel can
 * distinguish "no reference exists for this variable" from "the reference has no
 * value for this horizon".
 */
export const EXPECTED_ACCURACY_VARIABLES: readonly VariableName[] = [
  "temperature",
  "humidity",
  "pressure",
  "wind_speed",
];
