/**
 * Forecast provenance — the honest answer to "where did this number come from?"
 *
 * THE PROBLEM THIS FILE EXISTS TO SOLVE
 * -------------------------------------
 * The served inference policy (`prediction-model/data/inference_policy.json`,
 * policy_version 2.0.0) picks a SOURCE for every (horizon, variable) cell. As
 * shipped:
 *
 *     1h   temp/hum/pres/wind/winddir = persistence_fallback
 *     3h   temp/hum/pres/wind/winddir = persistence_fallback
 *     6h   temp=learned_model  wind=learned_model  rest persistence
 *     12h  temp=learned_model  wind=learned_model  rest persistence
 *     24h  wind=learned_model                        rest persistence
 *     all  heat_index = derived_from_selected_temp_and_humidity
 *
 * That is 20 of the 25 cells the policy chooses between the network and the
 * last observation served by PERSISTENCE. A dashboard that renders those
 * numbers next to the three that are genuinely modelled, with no distinction,
 * tells the reader that a 24h humidity figure is a forecast. It is not. It is
 * the last observation, carried forward, and it carries no information about
 * the next 24 hours at all.
 *
 * WHY THIS IS A RUNTIME MODULE IN `src/types`
 * -------------------------------------------
 * Two very different consumers must agree on the classification exactly:
 *   - `src/services/forecast.service.ts` (Node, reads the cache with `fs`), and
 *   - the prediction components (bundled for the browser, must not pull in `fs`).
 * A shared, dependency-free module is the only way to have ONE implementation.
 * The alternative — a token table in the service and a near-copy in a
 * component — is exactly the kind of duplication that lets a UI drift from the
 * engine. So the types and their only dependency-free derivations live here.
 * No I/O, no imports, no globals: safe to import from either side.
 *
 * DESIGN RULES
 * ------------
 *  1. FAIL CLOSED. An unrecognised policy token classifies as `"unknown"`, never
 *     as a model output. A future token must not be silently read as skill.
 *  2. NO POOLING. Mean absolute error is never averaged across variables: an
 *     hPa error dominates a degC error numerically while meaning no more.
 *     `verify_predictions.py` refuses to pool for the same reason.
 *  3. NO INVENTED DEFAULTS. A missing verification artifact yields an explicit
 *     unavailable status, never a zero and never a remembered number.
 */

/* -------------------------------------------------------------------------- */
/* Per-variable source selection                                              */
/* -------------------------------------------------------------------------- */

/**
 * The six variables the engine routes through the per-variable source policy.
 *
 * `heat_index` is included because it is a policy decision too — it is always
 * derived from whichever temperature and humidity were selected, so it is
 * labelled `derived` rather than being silently presented as modelled.
 */
export type ForecastVariable =
  | "temperature"
  | "humidity"
  | "pressure"
  | "windSpeed"
  | "windDirection"
  | "heatIndex";

/**
 * How a single served value was produced.
 *
 * - `learned_model`       the neural network produced it.
 * - `persistence_fallback` the last observation, carried forward unchanged.
 * - `derived`             computed from two other served values (heat index).
 * - `suppressed`          a quality gate forced a fallback; the number is the
 *                         last observation and the model output was withheld.
 * - `unknown`             the engine emitted a token this build does not
 *                         recognise. Treated as "not model output" everywhere.
 */
export type ForecastSourceId =
  | "learned_model"
  | "persistence_fallback"
  | "derived"
  | "suppressed"
  | "unknown";

/** Raw policy token -> classification. See `inference.py` for the emitters. */
const SOURCE_TOKEN_KINDS: Readonly<Record<string, ForecastSourceId>> = Object.freeze({
  learned_model: "learned_model",
  persistence_fallback: "persistence_fallback",
  // Emitted by inference.py when the input-sequence anomaly gate quarantines
  // the sequence. The value is still the origin observation.
  persistence_fallback_input_quarantined: "suppressed",
  derived_from_selected_temp_and_humidity: "derived",
});

/** Engine key (`selected_source_by_variable`) -> DTO variable name. */
const RAW_KEY_BY_VARIABLE: Readonly<Record<ForecastVariable, string>> = Object.freeze({
  temperature: "temperature",
  humidity: "humidity",
  pressure: "pressure",
  windSpeed: "wind_speed",
  windDirection: "wind_direction",
  heatIndex: "heat_index",
});

/** Stable render order. Also the denominator for the "x of y" counts. */
export const FORECAST_VARIABLES: readonly ForecastVariable[] = Object.freeze([
  "temperature",
  "humidity",
  "pressure",
  "windSpeed",
  "windDirection",
  "heatIndex",
] as const);

/** The raw policy token for each variable, as the engine spells it. */
export const RAW_POLICY_KEYS: readonly string[] = FORECAST_VARIABLES.map(
  (v) => RAW_KEY_BY_VARIABLE[v]
);

/**
 * Classify a raw policy token. Unrecognised input is `"unknown"`, never
 * `"learned_model"` — a token this build has never seen cannot be assumed to
 * carry skill.
 */
export function forecastSourceIdFromToken(token: unknown): ForecastSourceId {
  if (typeof token !== "string") return "unknown";
  return SOURCE_TOKEN_KINDS[token] ?? "unknown";
}

/** One resolved variable: its source, and what that means for trust. */
export interface ForecastVariableSource {
  variable: ForecastVariable;
  /** The literal token the engine emitted, preserved for audit. */
  rawToken: string | null;
  sourceId: ForecastSourceId;
  /** True only when the value came out of the neural network. */
  isModelOutput: boolean;
  /** True when the value is the last observation carried forward. */
  isCarriedForward: boolean;
}

/** The rain channel is a weighted blend, not one of the six policy variables. */
export interface RainBlendProvenance {
  source: string | null;
  modelWeight: number | null;
  persistenceWeight: number | null;
  /** True when any of the model's weight survives into the published number. */
  includesModelOutput: boolean;
}

export interface ForecastSourceProvenance {
  /** The horizon these sources apply to, e.g. "6h". */
  horizon: string;
  policyVersion: string | null;
  variables: ForecastVariableSource[];
  modelOutputCount: number;
  carriedForwardCount: number;
  derivedCount: number;
  suppressedCount: number;
  unknownCount: number;
  totalCount: number;
  /** Convenience: "2 of 6" style numerator, for the operator summary line. */
  modelBackedLabel: string;
  rain: RainBlendProvenance;
}

export interface BuildProvenanceInput {
  horizon?: string | null;
  policyVersion?: string | null;
  /** `selected_source_by_variable` exactly as the engine emitted it. */
  selectedSourceByVariable?: Record<string, unknown> | null;
  /** `input_quality_gate.learned_output_trusted`, when present. */
  learnedOutputTrusted?: boolean | null;
  rainProbabilitySource?: string | null;
  rainModelWeight?: number | null;
  rainPersistenceWeight?: number | null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Build the per-variable provenance block for one forecast.
 *
 * Defence in depth: when the engine's own quality gate reports that learned
 * outputs were NOT trusted, every `learned_model` token is downgraded to
 * `suppressed`. inference.py already rewrites the token in that case
 * (`persistence_fallback_input_quarantined`); if it ever failed to, the served
 * value would still be the origin observation and must not read as modelled.
 */
export function buildSourceProvenance(input: BuildProvenanceInput): ForecastSourceProvenance {
  const selected = input.selectedSourceByVariable ?? {};
  const trusted = input.learnedOutputTrusted !== false;

  const variables: ForecastVariableSource[] = FORECAST_VARIABLES.map((variable) => {
    const rawKey = RAW_KEY_BY_VARIABLE[variable];
    const rawValue = selected[rawKey];
    const rawToken = typeof rawValue === "string" ? rawValue : null;
    let sourceId = forecastSourceIdFromToken(rawToken);

    if (sourceId === "learned_model" && !trusted) {
      sourceId = "suppressed";
    }

    return {
      variable,
      rawToken,
      sourceId,
      isModelOutput: sourceId === "learned_model",
      isCarriedForward: sourceId === "persistence_fallback" || sourceId === "suppressed",
    };
  });

  const count = (id: ForecastSourceId): number =>
    variables.filter((v) => v.sourceId === id).length;

  const modelOutputCount = count("learned_model");

  const modelWeight = input.rainModelWeight;
  const persistenceWeight = input.rainPersistenceWeight;

  return {
    horizon: input.horizon ?? "unknown",
    policyVersion: input.policyVersion ?? null,
    variables,
    modelOutputCount,
    carriedForwardCount: count("persistence_fallback"),
    derivedCount: count("derived"),
    suppressedCount: count("suppressed"),
    unknownCount: count("unknown"),
    totalCount: variables.length,
    modelBackedLabel: `${modelOutputCount} of ${variables.length}`,
    rain: {
      source: input.rainProbabilitySource ?? null,
      modelWeight: isFiniteNumber(modelWeight) ? modelWeight : null,
      persistenceWeight: isFiniteNumber(persistenceWeight) ? persistenceWeight : null,
      includesModelOutput: isFiniteNumber(modelWeight) ? modelWeight > 0 : false,
    },
  };
}

/**
 * Build a provenance block for the case where NO model forecast is attached.
 *
 * Nothing is model output. The operator still needs to be told that, rather
 * than shown an empty strip that reads as "no information".
 */
export function buildUnavailableProvenance(reason: string): ForecastSourceProvenance {
  return {
    horizon: "unknown",
    policyVersion: null,
    variables: FORECAST_VARIABLES.map((variable) => ({
      variable,
      rawToken: null,
      sourceId: "unknown" as ForecastSourceId,
      isModelOutput: false,
      isCarriedForward: false,
    })),
    modelOutputCount: 0,
    carriedForwardCount: 0,
    derivedCount: 0,
    suppressedCount: 0,
    unknownCount: FORECAST_VARIABLES.length,
    totalCount: FORECAST_VARIABLES.length,
    modelBackedLabel: `0 of ${FORECAST_VARIABLES.length}`,
    rain: {
      source: null,
      modelWeight: null,
      persistenceWeight: null,
      includesModelOutput: false,
    },
  };
}

/**
 * Downgrade one channel from `learned_model` to `suppressed`.
 *
 * This exists for a specific, verified gap in the engine. `inference.py` has a
 * live anemometer health check: when it reports a dead or absent sensor it
 * replaces the served wind speed with the last observation — but it does NOT
 * rewrite `selected_source_by_variable.wind_speed`, so the policy token still
 * reads `learned_model`. Without this correction the dashboard would label a
 * carried-forward wind reading as model output on exactly the stations whose
 * wind sensor is broken.
 *
 * A no-op when the channel is not `learned_model`, or when not quarantined.
 */
export function applyChannelQuarantine(
  provenance: ForecastSourceProvenance,
  channel: ForecastVariable,
  quarantined: boolean
): ForecastSourceProvenance {
  if (!quarantined) return provenance;

  const target = provenance.variables.find((v) => v.variable === channel);
  if (!target || target.sourceId !== "learned_model") return provenance;

  return {
    ...provenance,
    variables: provenance.variables.map((v) =>
      v.variable === channel
        ? { ...v, sourceId: "suppressed" as ForecastSourceId, isModelOutput: false, isCarriedForward: true }
        : v
    ),
    modelOutputCount: Math.max(0, provenance.modelOutputCount - 1),
    suppressedCount: provenance.suppressedCount + 1,
    modelBackedLabel: `${Math.max(0, provenance.modelOutputCount - 1)} of ${provenance.totalCount}`,
  };
}

/* -------------------------------------------------------------------------- */
/* Verified accuracy                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Variables the verification layer can score, with the units it compares in.
 *
 * Units are taken from `verify_predictions.py::VAR_TO_TELEMETRY`: both sides
 * are degC, %RH, hPa and m/s, which is the only reason a straight comparison
 * is valid. An unlisted variable scores `unit: null` and the UI must not
 * invent one.
 */
const VERIFICATION_UNITS: Readonly<Record<string, string>> = Object.freeze({
  temperature: "°C",
  humidity: "%",
  pressure: "hPa",
  wind_speed: "m/s",
});

/**
 * Why verified accuracy is or is not shown.
 *
 * - `available`                 scored records with a finite MAE were found.
 * - `no_scored_verifications`   the summary exists but nothing has matured
 *                               and been scored yet. The current live state.
 * - `stale`                     the summary predates the freshness budget.
 * - `unavailable`               the artifact is missing or unparseable.
 *
 * There is deliberately no `empty` / `default` case: a status that means
 * "we have nothing, so assume zero error" is the exact failure this project
 * has already been bitten by.
 */
export type VerifiedAccuracyStatus =
  | "available"
  | "no_scored_verifications"
  | "stale"
  | "unavailable";

export interface VerifiedAccuracyEntry {
  /** Engine telemetry key, e.g. "temperature". */
  variable: string;
  /** Which producer was scored, e.g. "lln" | "nwp" | "blend" | "persistence". */
  producer: string;
  /** Mean absolute error, in `unit`. Never pooled with another variable. */
  mae: number;
  /** Sample count behind `mae`. Always shown alongside it. */
  sampleCount: number;
  unit: string | null;
}

export interface VerifiedAccuracy {
  status: VerifiedAccuracyStatus;
  /** Operator-facing explanation of the status. Null when available. */
  reason: string | null;
  /** Records scored by the verification pass. */
  scoredCount: number;
  /** Forecasts whose target time has not arrived yet. */
  pendingCount: number;
  /** Predictions with no observation inside the tolerance. */
  unmatchedCount: number;
  /** How far from the target an observation may be and still count. */
  toleranceMinutes: number | null;
  /** Non-empty only when `status === "available"`. */
  entries: VerifiedAccuracyEntry[];
  /**
   * mtime of the summary artifact, as an ISO string. This file has no
   * `generated_at` field, so file mtime is the only honest statement of when
   * the numbers were computed.
   */
  observedAt: string | null;
  ageSeconds: number | null;
  /**
   * Always `null`. Mean absolute error is not averaged across degC, %RH, hPa
   * and m/s; the value is present in the shape only so a consumer cannot
   * reintroduce the mistake by reaching for it.
   */
  pooledMae: null;
  note: string;
}

export interface SummarizeVerificationOptions {
  /** ISO mtime of the summary artifact, or null if it could not be stat'ed. */
  observedAt: string | null;
  /** `Date.now()` at read time, so the function is deterministic under test. */
  now: number;
  maxAgeSeconds: number;
}

const VERIFICATION_NOTE =
  "Scored against what actually happened, per variable and per producer. " +
  "Errors are never averaged across variables — the units do not share a scale.";

function unavailable(
  status: Exclude<VerifiedAccuracyStatus, "available">,
  reason: string
): VerifiedAccuracy {
  return {
    status,
    reason,
    scoredCount: 0,
    pendingCount: 0,
    unmatchedCount: 0,
    toleranceMinutes: null,
    entries: [],
    observedAt: null,
    ageSeconds: null,
    pooledMae: null,
    note: VERIFICATION_NOTE,
  };
}

/** An explicit "not available" result. Use this; do not fabricate a number. */
export function verifiedAccuracyUnavailable(
  reason: string,
  status: "unavailable" | "stale" = "unavailable"
): VerifiedAccuracy {
  return unavailable(status, reason);
}

/**
 * Normalise `prediction-model/data/prediction_verification_summary.json` —
 * the artifact written by `prediction-model/src/verify_predictions.py` — into a
 * shape the UI can render without having to invent anything.
 *
 * Rules, in order of precedence:
 *  1. Unreadable / non-object input            -> `unavailable`.
 *  2. Artifact older than the freshness budget  -> `stale`, entries withheld.
 *  3. No finite MAE with a non-zero sample size -> `no_scored_verifications`.
 *  4. Otherwise                                -> `available`.
 *
 * A `null` MAE is how `verify_predictions.py` spells "the list was empty"
 * (`mean()` returns `None` for `[]`). It is dropped, never rendered as 0.
 */
export function summarizeVerification(
  raw: unknown,
  options: SummarizeVerificationOptions
): VerifiedAccuracy {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    return unavailable("unavailable", "Verification summary missing or unreadable.");
  }

  const summary = raw as Record<string, unknown>;
  const observedAt = options.observedAt;
  let ageSeconds: number | null = null;

  if (observedAt) {
    const ms = Date.parse(observedAt);
    if (Number.isFinite(ms)) {
      ageSeconds = Math.max(0, Math.floor((options.now - ms) / 1000));
    }
  }

  if (ageSeconds !== null && options.maxAgeSeconds > 0 && ageSeconds > options.maxAgeSeconds) {
    const stale = unavailable(
      "stale",
      `Last verification pass is ${ageSeconds}s old, beyond the ` +
        `${options.maxAgeSeconds}s freshness budget. Run verify_predictions.py to refresh.`
    );
    return { ...stale, observedAt, ageSeconds };
  }

  const scoredCount = isFiniteNumber(summary.scored) ? summary.scored : 0;
  const pendingCount = isFiniteNumber(summary.pending) ? summary.pending : 0;
  const unmatchedCount = isFiniteNumber(summary.unmatched) ? summary.unmatched : 0;
  const toleranceMinutes = isFiniteNumber(summary.tolerance_minutes)
    ? summary.tolerance_minutes
    : null;

  const maeByProducer =
    summary.mae_by_variable_and_producer && typeof summary.mae_by_variable_and_producer === "object"
      ? (summary.mae_by_variable_and_producer as Record<string, unknown>)
      : {};
  const nByProducer =
    summary.n_by_variable_and_producer && typeof summary.n_by_variable_and_producer === "object"
      ? (summary.n_by_variable_and_producer as Record<string, unknown>)
      : {};

  const entries: VerifiedAccuracyEntry[] = [];

  for (const variable of Object.keys(maeByProducer).sort()) {
    const perProducer = maeByProducer[variable];
    if (!perProducer || typeof perProducer !== "object" || Array.isArray(perProducer)) continue;

    const nForVariable =
      nByProducer[variable] && typeof nByProducer[variable] === "object"
        ? (nByProducer[variable] as Record<string, unknown>)
        : {};

    for (const producer of Object.keys(perProducer as Record<string, unknown>).sort()) {
      const mae = (perProducer as Record<string, unknown>)[producer];
      const n = nForVariable[producer];
      if (!isFiniteNumber(mae)) continue;
      // A mean over an empty sample is not a mean. verify_predictions.py
      // guards this too, but the guard belongs here as well as there.
      if (!isFiniteNumber(n) || n < 1) continue;

      entries.push({
        variable,
        producer,
        mae,
        sampleCount: n,
        unit: VERIFICATION_UNITS[variable] ?? null,
      });
    }
  }

  if (entries.length === 0) {
    const reason =
      pendingCount > 0
        ? `No matured prediction has been scored yet. ${pendingCount} forecast(s) are ` +
          `still waiting for their target time to pass.`
        : "No matured prediction has been scored yet.";
    return {
      status: "no_scored_verifications",
      reason,
      scoredCount,
      pendingCount,
      unmatchedCount,
      toleranceMinutes,
      entries: [],
      observedAt,
      ageSeconds,
      pooledMae: null,
      note: VERIFICATION_NOTE,
    };
  }

  return {
    status: "available",
    reason: null,
    scoredCount,
    pendingCount,
    unmatchedCount,
    toleranceMinutes,
    entries,
    observedAt,
    ageSeconds,
    pooledMae: null,
    note: VERIFICATION_NOTE,
  };
}
