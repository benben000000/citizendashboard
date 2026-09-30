/**
 * prediction-provenance-core.ts
 *
 * Pure, framework-free logic behind the prediction provenance / audit panel.
 *
 * WHY THIS IS SEPARATE FROM THE COMPONENT
 * ---------------------------------------
 * Two reasons, both about honesty rather than layout.
 *
 * 1. Every claim the panel makes about a forecast is a *derivation* over the
 *    audit trail -- is this value carried forward from an observation, does the
 *    recorded producer label agree with the policy, is there any verified error
 *    for this variable. Derivations are where a UI quietly starts lying (a
 *    missing MAE rendered as 0, an unknown source rendered as "model"). Keeping
 *    them here, with no React and no i18n, means they can be tested against
 *    fixtures instead of being eyeballed.
 *
 * 2. It runs in the test runner. See `__tests__/` next to this file.
 *
 * GROUND RULES ENFORCED HERE (do not relax them at the call site)
 * --------------------------------------------------------------
 *  - MAE is NEVER averaged across variables. degC, %RH, hPa and m/s do not
 *    share a scale; an hPa error dwarfs a degC error numerically while meaning
 *    no more. `buildAccuracyTable` keys strictly by variable, and each row
 *    carries its own unit. This mirrors the deliberate absence of a pooled MAE
 *    in `prediction-model/src/verify_predictions.py`.
 *  - "Verified" (scored against observations that have since arrived) and
 *    "expected" (a static offline benchmark) are different claims and are never
 *    merged. A missing verified score is `null`, never 0.
 *  - A value that was merely carried forward from an observation is not a
 *    prediction. `classifyServedBy` and `isCarriedForward` exist so the panel
 *    can tell the difference at a glance.
 *
 * Wire formats mirror `prediction-model/src/prediction_audit.py` and
 * `verify_predictions.py` field-for-field. See README-endpoint.md.
 */

/* ────────────────────────────────────────────────────────────────────────────
 * Variables and units
 *
 * UNIT TRAP, READ THIS
 * --------------------
 * `AuditRecord.variables.wind_speed.value` is written from the model output
 * key `wind_speed_kmh`, and `verify_predictions.py` scores it against the
 * telemetry key `wind_speed_kmh`. It is therefore KILOMETRES PER HOUR, despite
 * the field being called `wind_speed`. This was confirmed empirically: all
 * 6,785 live records carry `variables.wind_speed.value ===
 * forecast_raw.wind_speed_kmh` exactly.
 *
 * Offline accuracy figures for wind are conventionally quoted in m/s
 * (1.217 m/s at +6h). Both are therefore accepted, both are converted
 * explicitly, and the raw value is always kept alongside the canonical one so
 * nothing is hidden by a conversion factor.
 * ──────────────────────────────────────────────────────────────────────────── */

export type VariableName = "temperature" | "humidity" | "pressure" | "wind_speed";

/** The four variables the ingestor writes into `AuditRecord.variables`. */
export const AUDITED_VARIABLES: readonly VariableName[] = [
  "temperature",
  "humidity",
  "pressure",
  "wind_speed",
] as const;

export type MetricKind = "mae" | "brier";

export interface VariableUnitSpec {
  variable: VariableName;
  label: string;
  /** Unit of the number as it appears in `AuditRecord.variables[name].value`. */
  auditUnit: string;
  /** Unit every accuracy figure is displayed in. */
  canonicalUnit: string;
  /** Key in `ObservationAudit.telemetry` that carries this variable's truth. */
  telemetryKey: string;
  /** Metric measured for this variable. Brier is unitless (rain occurrence). */
  metric: MetricKind;
}

export const VARIABLE_UNITS: Readonly<Record<VariableName, VariableUnitSpec>> = {
  temperature: {
    variable: "temperature",
    label: "Temperature",
    auditUnit: "degC",
    canonicalUnit: "degC",
    telemetryKey: "temperature_c",
    metric: "mae",
  },
  humidity: {
    variable: "humidity",
    label: "Humidity",
    auditUnit: "%RH",
    canonicalUnit: "%RH",
    telemetryKey: "humidity_pct",
    metric: "mae",
  },
  pressure: {
    variable: "pressure",
    label: "Pressure",
    auditUnit: "hPa",
    canonicalUnit: "hPa",
    telemetryKey: "pressure_hpa",
    metric: "mae",
  },
  wind_speed: {
    variable: "wind_speed",
    label: "Wind speed",
    auditUnit: "km/h",
    canonicalUnit: "m/s",
    telemetryKey: "wind_speed_kmh",
    metric: "mae",
  },
};

/** Rain occurrence is scored with Brier, not MAE, and is not an audited cell. */
export const RAIN_VARIABLE = {
  variable: "rain_occurrence",
  label: "Rain occurrence",
  auditUnit: "probability",
  canonicalUnit: "Brier (unitless)",
  metric: "brier" as MetricKind,
};

/** Exact conversions from a unit to the canonical unit for that variable. */
const TO_CANONICAL: Partial<Record<VariableName, (value: number) => number>> = {
  wind_speed: (value) => value / 3.6,
};

/**
 * Convert an error magnitude into the variable's canonical unit.
 *
 * Returns `null` for a null input: a missing measurement converts to nothing,
 * never to zero.
 */
export function toCanonicalUnit(variable: VariableName, value: number | null | undefined): number | null {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  const convert = TO_CANONICAL[variable];
  const converted = convert ? convert(value) : value;
  return Number.isFinite(converted) ? converted : null;
}

/** Unit of `unit` expressed in the canonical unit for `variable`. */
export function conversionNote(
  variable: VariableName,
  fromUnit: string
): string | null {
  const spec = VARIABLE_UNITS[variable];
  if (!spec || fromUnit === spec.canonicalUnit) return null;
  return `converted from ${fromUnit}`;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Wire formats
 *
 * Structural subsets of the Python records. Field names match the JSONL
 * exactly; anything the trail may omit is optional here, because
 * `prediction_audit.write()` explicitly degrades a record to `null` fields
 * rather than dropping the line.
 * ──────────────────────────────────────────────────────────────────────────── */

export interface AuditVariableEntry {
  value: number | null;
  producer: string | null;
  nwp_raw?: number | null;
  nwp_corrected?: number | null;
}

export interface AuditRecord {
  schema_version?: number;
  record_id: string;
  recorded_at_utc: string | null;
  origin_timestamp_utc: string | null;
  station_id: string | null;
  device_id: string | null;
  horizon_hours: number | null;
  provenance?: {
    model?: {
      bundle?: string | null;
      checkpoint_trained_at?: string | null;
      checkpoint_sha256?: string | null;
      seed?: number | null;
    } | null;
    policy?: {
      policy_version?: string | null;
      policy_code_commit?: string | null;
      policy_regenerated_by?: string | null;
    } | null;
    nwp?: {
      applied?: boolean;
      reason?: string | null;
      source?: string | null;
      cycle?: string | null;
      issued_at?: string | null;
    } | null;
  } | null;
  variables: Record<string, AuditVariableEntry | null | undefined>;
  sensor_health?: { verdict?: string | null; reason?: string | null; n?: number | null } | null;
  extra?: Record<string, unknown> | null;
  forecast_raw?: Record<string, unknown> | null;
}

export interface ObservationRecord {
  record_id: string;
  recorded_at_utc: string | null;
  observed_at_utc: string | null;
  station_id: string | null;
  telemetry: Record<string, number | null | undefined>;
}

export interface VerificationVariable {
  predicted: number | null;
  actual: number | null;
  error: number | null;
  abs_error: number | null;
  producer: string | null;
}

export interface VerificationRecord {
  record_id: string;
  prediction_record_id: string | null;
  verified_at_utc: string | null;
  station_id: string | null;
  horizon_hours: number | null;
  origin_timestamp_utc: string | null;
  target_timestamp_utc: string | null;
  observation_record_id: string | null;
  observation_offset_minutes: number | null;
  policy_version: string | null;
  model_bundle: string | null;
  variables: Record<string, VerificationVariable | null | undefined>;
}

export interface PolicyHorizonConfig {
  selectedSources: Record<string, string | null | undefined>;
  rainModelWeight?: number | null;
  rainPersistenceWeight?: number | null;
  operationalRainThreshold?: number | null;
  calibrationCodeCommit?: string | null;
}

export interface PolicyDocument {
  policyVersion: string | null;
  policyCodeCommit: string | null;
  policyRegeneratedBy: string | null;
  /** Keyed by horizon hours as a string, matching `inference_policy.json`. */
  horizons: Record<string, PolicyHorizonConfig>;
}

export interface BundleManifest {
  bundle: string;
  checkpointSha256: string | null;
  trainingTimestamp: string | null;
  implementationCommit: string | null;
  modelFamily: string | null;
  modelStatus: string | null;
  uncertaintyStatus: string | null;
  randomSeed: number | null;
}

/**
 * How much of the tail the server actually read.
 *
 * The trail is multi-megabyte and is being appended to by a running ingestor,
 * so a read is always bounded and the panel must be able to say so. A
 * `mode` other than `"tail"` means the server had a reason to read further and
 * should say what it was.
 */
export interface ReadCoverage {
  mode: "tail" | "full" | "unavailable";
  /** Bytes of the file the server sought into and read. */
  bytesScanned: number;
  /** Total size of the file at read time, when known. */
  fileBytes: number | null;
  linesScanned: number;
  recordsReturned: number;
  /** First line was cut in half by the seek and had to be discarded. */
  discardedPartialHead: boolean;
  /** Last line was cut in half by a write that was still in flight. */
  truncatedTailLine: boolean;
  skippedMalformedLines: number;
  earliestIncludedUtc: string | null;
  latestIncludedUtc: string | null;
  /** True when records exist before `earliestIncludedUtc` that we did not see. */
  coverageIncomplete: boolean;
}

export interface VerificationSection {
  /** Path of `prediction_verification.jsonl`, for the operator to find. */
  sourcePath: string | null;
  exists: boolean;
  /** Verbatim counters from `verify()`. */
  scored: number;
  pending: number;
  unmatched: number;
  alreadyVerified: number;
  toleranceMinutes: number | null;
  coverage: ReadCoverage | null;
  /** Verbatim per-variable accumulators from `verify()`. Never merged. */
  maeByVariable: Record<string, number | null>;
  maeByVariableAndProducer: Record<string, Record<string, number | null>>;
  nByVariableAndProducer: Record<string, Record<string, number | null>>;
  /** Bounded sample of verification records, newest last. */
  records: VerificationRecord[];
}

export interface OriginObservation {
  observedAtUtc: string | null;
  recordId: string | null;
  /** Minutes between the observation and the forecast's origin timestamp. */
  ageMinutes: number | null;
  telemetry: Record<string, number | null | undefined>;
  coverage: ReadCoverage | null;
}

export interface ExpectedAccuracyCell {
  horizonHours: number;
  metric: MetricKind;
  /** Error magnitude in the variable's canonical unit, or null if not measured. */
  value: number | null;
  /**
   * Positive means the model beat the persistence baseline by this percentage.
   * `null` means "no skill was measured", which is different from 0% -- a cell
   * can be persistence-equivalent and still be an honest measurement, and the
   * panel renders the two differently.
   */
  beatsPersistencePct: number | null;
  status: "MEASURED" | "PERSISTENCE_EQUIVALENT" | "NOT_MEASURED";
  /** Baselines the offline benchmark reported the model as beating. */
  beatsBaselines: readonly string[];
}

export interface ExpectedAccuracySection {
  /** Where these numbers come from. Always shown; they are not live data. */
  source: string;
  asOf: string | null;
  /** Set when the reference is known to be incomplete. */
  coverageNote: string | null;
  /** Keyed by variable name; `rain_occurrence` is not an audited cell. */
  variables: Record<string, ExpectedAccuracyCell[]>;
}

export interface ProvenancePayload {
  schemaVersion: number;
  generatedAtUtc: string | null;
  stationId: string;
  /** The single record the panel is describing, or null when none matched. */
  forecast: AuditRecord | null;
  /** Bounded, oldest first. */
  forecastHistory: AuditRecord[];
  forecastCoverage: ReadCoverage | null;
  verification: VerificationSection | null;
  policy: PolicyDocument | null;
  bundles: Record<string, BundleManifest | null | undefined>;
  expectedAccuracy: ExpectedAccuracySection | null;
  originObservation: OriginObservation | null;
  /** Non-fatal problems the server hit, e.g. a locked or rotated file. */
  warnings: string[];
}

/* ────────────────────────────────────────────────────────────────────────────
 * Bounded JSONL parsing
 * ──────────────────────────────────────────────────────────────────────────── */

export interface ParsedLines {
  records: Array<Record<string, unknown>>;
  malformed: number;
  /** First line was a fragment produced by seeking into the middle of a file. */
  partialHead: boolean;
  /** Last line was a fragment produced by a write that had not completed. */
  partialTail: boolean;
  completeLines: number;
}

/**
 * Parse a bounded chunk of JSONL, tolerating damage at both ends.
 *
 * Mirrors `prediction_audit.read_records()`, which skips unparseable lines
 * rather than failing. The two extra tolerances are specific to tail reading:
 *
 *  - `partialHead` is expected and benign. A bounded read seeks to
 *    `size - N`, which normally lands inside a line; that fragment is dropped.
 *  - `partialTail` is expected while an ingestor is actively appending. A line
 *    is only considered complete if it ends with a newline, so a half-written
 *    record is never surfaced as if it were whole.
 *
 * A truncated final line is deliberately NOT dropped (unlike the Python
 * reader) -- it is reported, so the panel can say the trail is being written to
 * right now instead of silently hiding the newest record.
 */
export function parseTailLines(chunk: string): ParsedLines {
  const records: Array<Record<string, unknown>> = [];
  let malformed = 0;
  let partialHead = false;
  let partialTail = false;

  if (chunk.length === 0) {
    return { records, malformed, partialHead, partialTail, completeLines: 0 };
  }

  const endsWithNewline = chunk.endsWith("\n");
  const lines = chunk.split("\n");
  const start = 0;
  let end = lines.length;

  if (endsWithNewline) {
    // The split leaves a trailing "" that is not a record.
    end = lines.length - 1;
  } else {
    // No trailing newline: the final line is a fragment of an in-flight write.
    partialTail = true;
  }

  // The caller passes a tail read starting at an arbitrary byte offset. Only
  // the first line can be a fragment, and only when the chunk begins mid-line.
  if (start === 0 && end > 0) {
    partialHead = true;
  }

  for (let i = start; i < end; i += 1) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      malformed += 1;
      continue;
    }
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      malformed += 1;
      continue;
    }
    records.push(parsed as Record<string, unknown>);
  }

  return { records, malformed, partialHead, partialTail, completeLines: records.length };
}

/* ────────────────────────────────────────────────────────────────────────────
 * Record selection
 * ──────────────────────────────────────────────────────────────────────────── */

function toEpoch(value: string | null | undefined): number {
  if (!value) return Number.NEGATIVE_INFINITY;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : Number.NEGATIVE_INFINITY;
}

/** `record_id` is a sha256 prefix, so it is a safe deterministic tie-break. */
function byNewestThenId(a: AuditRecord, b: AuditRecord): number {
  const delta = toEpoch(b.recorded_at_utc) - toEpoch(a.recorded_at_utc);
  if (delta !== 0) return delta;
  const originDelta = toEpoch(b.origin_timestamp_utc) - toEpoch(a.origin_timestamp_utc);
  if (originDelta !== 0) return originDelta;
  return String(b.record_id).localeCompare(String(a.record_id));
}

export interface ForecastSelection {
  stationId?: string | null;
  horizonHours?: number | null;
  recordId?: string | null;
}

/**
 * Pick the one record the panel should describe.
 *
 * `recordId` wins outright when supplied -- that is the "show me this exact
 * published number" path. Otherwise the newest record for the station at the
 * requested horizon wins.
 *
 * The trail contains predictions that were computed but never published
 * (`extra.prediction_status` may be `"ERROR"`). Those are kept -- an audit tool
 * that hides failures is worse than no audit tool -- but `recordId` is the only
 * way to reach one deliberately.
 */
export function selectForecastRecord(
  records: readonly AuditRecord[],
  selection: ForecastSelection = {}
): AuditRecord | null {
  if (records.length === 0) return null;

  if (selection.recordId) {
    const exact = records.find((r) => r.record_id === selection.recordId);
    if (exact) return exact;
  }

  const stationId = selection.stationId ?? undefined;
  const horizonHours =
    selection.horizonHours === null || selection.horizonHours === undefined
      ? undefined
      : Number(selection.horizonHours);

  const candidates = records.filter((record) => {
    if (stationId !== undefined && record.station_id !== stationId) return false;
    if (horizonHours !== undefined) {
      const h = record.horizon_hours;
      if (h === null || h === undefined || Number(h) !== horizonHours) return false;
    }
    return true;
  });

  if (candidates.length === 0) return null;
  return [...candidates].sort(byNewestThenId)[0];
}

/* ────────────────────────────────────────────────────────────────────────────
 * Producer / source classification
 * ──────────────────────────────────────────────────────────────────────────── */

export type SourceKind = "LEARNED_MODEL" | "PERSISTENCE" | "NWP" | "NWP_BLEND" | "DERIVED" | "UNKNOWN";

export type ServedBy = "MODEL" | "PERSISTENCE" | "DERIVED" | "UNKNOWN";

/** Producer strings `prediction_audit` documents: lln | nwp | blend | persistence. */
export function normaliseProducer(raw: string | null | undefined): SourceKind {
  const value = (raw ?? "").trim().toLowerCase();
  switch (value) {
    case "lln":
      return "LEARNED_MODEL";
    case "nwp":
      return "NWP";
    case "blend":
      return "NWP_BLEND";
    case "persistence":
    case "persistence_fallback":
      return "PERSISTENCE";
    case "":
      return "UNKNOWN";
    default:
      return value.includes("persist") ? "PERSISTENCE" : "UNKNOWN";
  }
}

/** Source strings `inference_policy.json` documents. */
export function normalisePolicySource(raw: string | null | undefined): SourceKind {
  const value = (raw ?? "").trim().toLowerCase();
  if (value.length === 0) return "UNKNOWN";
  if (value.startsWith("persistence")) return "PERSISTENCE";
  if (value.startsWith("learned")) return "LEARNED_MODEL";
  if (value.startsWith("derived")) return "DERIVED";
  if (value === "nwp") return "NWP";
  if (value === "blend") return "NWP_BLEND";
  return "UNKNOWN";
}

export function sourceKindLabel(kind: SourceKind): string {
  switch (kind) {
    case "LEARNED_MODEL":
      return "learned model";
    case "PERSISTENCE":
      return "persistence";
    case "NWP":
      return "NWP";
    case "NWP_BLEND":
      return "NWP blend";
    case "DERIVED":
      return "derived";
    default:
      return "unknown";
  }
}

export interface CellClassification {
  horizonHours: number;
  variable: VariableName;
  policySource: string | null;
  policyKind: SourceKind;
  recordedProducer: string | null;
  recordedKind: SourceKind;
  /**
   * How the number was actually produced.
   *
   * Derived from the POLICY, not from `variables[v].producer`. See
   * `producerLabelTrustworthy` for why.
   */
  servedBy: ServedBy;
  /** The two sources disagree. Shown, never silently resolved. */
  labelConflict: boolean;
  conflictReason: string | null;
}

function servedByKind(kind: SourceKind): ServedBy {
  switch (kind) {
    case "LEARNED_MODEL":
    case "NWP":
    case "NWP_BLEND":
      return "MODEL";
    case "PERSISTENCE":
      return "PERSISTENCE";
    case "DERIVED":
      return "DERIVED";
    default:
      return "UNKNOWN";
  }
}

function conflictReason(policyKind: SourceKind, recordedKind: SourceKind): string | null {
  const policySays = servedByKind(policyKind);
  const recordSays = servedByKind(recordedKind);

  if (policyKind === "UNKNOWN" || recordedKind === "UNKNOWN") {
    return "one side is unknown, so the two cannot be reconciled";
  }
  if (policySays === recordSays) return null;

  if (policyKind === "PERSISTENCE") {
    return "policy serves this cell from the origin observation, but the audit record labels it a model prediction";
  }
  if (policyKind === "DERIVED") {
    return "policy derives this cell from other variables, but the audit record names a direct producer";
  }
  return "policy selects a model source, but the audit record does not name a model producer";
}

/**
 * Classify one horizon x variable cell.
 *
 * `policySources` comes from `inference_policy.json` for that horizon. When the
 * policy has no entry for the horizon, the cell is `UNKNOWN` and the recorded
 * producer is used -- with an explicit conflict reason, because "we do not know"
 * is not the same as "it was a model".
 */
export function classifyCell(
  horizonHours: number,
  variable: VariableName,
  policySources: Record<string, string | null | undefined> | null | undefined,
  recordedProducer: string | null | undefined
): CellClassification {
  const policySource =
    policySources && typeof policySources === "object"
      ? (policySources[variable] ?? null)
      : null;
  const policyKind = normalisePolicySource(policySource);
  const recordedKind = normaliseProducer(recordedProducer);
  const effectivePolicyKind = policyKind === "UNKNOWN" ? recordedKind : policyKind;

  return {
    horizonHours,
    variable,
    policySource,
    policyKind,
    recordedProducer: recordedProducer ?? null,
    recordedKind,
    servedBy: servedByKind(effectivePolicyKind),
    labelConflict: conflictReason(policyKind, recordedKind) !== null,
    conflictReason: conflictReason(policyKind, recordedKind),
  };
}

/* ────────────────────────────────────────────────────────────────────────────
 * The source matrix: every horizon x variable cell in one pass
 * ──────────────────────────────────────────────────────────────────────────── */

export interface SourceMatrix {
  horizons: number[];
  cells: CellClassification[];
  totalCells: number;
  persistenceCells: number;
  modelCells: number;
  derivedCells: number;
  unknownCells: number;
  conflictCells: number;
  /**
   * False when every cell carries the same producer string while the policy
   * varies across cells.
   *
   * This is a real condition today: `mqtt_live_ingestor.py` hardcodes
   * `producer="lln"` for all four audited variables, so `variables[v].producer`
   * cannot distinguish persistence from model and must not be trusted on its
   * own. The panel surfaces this instead of rendering "lln" everywhere.
   */
  producerLabelTrustworthy: boolean;
  producerLabelWarning: string | null;
}

function parseHorizonKey(key: string): number | null {
  const n = Number(key);
  return Number.isFinite(n) && n > 0 ? n : null;
}

export interface BuildSourceMatrixInput {
  policy: PolicyDocument | null | undefined;
  /** Producer strings observed in the trail, per horizon and variable. */
  observedProducers?: Record<string, Record<string, string | null | undefined>>;
}

/**
 * Build the full grid, not just the displayed horizon.
 *
 * The point of the panel is that an operator can see, in one dense strip, how
 * much of what we publish is actually modelled. Truncating to the selected
 * horizon would hide exactly the fact worth seeing: the majority of cells are
 * persistence.
 */
export function buildSourceMatrix(input: BuildSourceMatrixInput): SourceMatrix {
  const policy = input.policy ?? null;
  const horizonKeys = Object.keys(policy?.horizons ?? {});
  const horizons = horizonKeys
    .map(parseHorizonKey)
    .filter((n): n is number => n !== null)
    .sort((a, b) => a - b);

  const observed = input.observedProducers ?? {};

  const cells: CellClassification[] = [];
  for (const horizon of horizons) {
    const sources = policy?.horizons[String(horizon)]?.selectedSources ?? null;
    for (const variable of AUDITED_VARIABLES) {
      cells.push(
        classifyCell(horizon, variable, sources, observed[String(horizon)]?.[variable] ?? null)
      );
    }
  }

  const observedProducerValues = new Set<string>();
  for (const byVariable of Object.values(observed)) {
    for (const value of Object.values(byVariable ?? {})) {
      if (typeof value === "string" && value.trim().length > 0) observedProducerValues.add(value.trim());
    }
  }

  const policyVaries = new Set(
    cells.filter((c) => c.policyKind !== "UNKNOWN").map((c) => c.policyKind)
  );
  const constantLabel =
    observedProducerValues.size === 1 && policyVaries.size > 1;

  let producerLabelWarning: string | null = null;
  if (constantLabel) {
    const only = [...observedProducerValues][0];
    producerLabelWarning =
      `Audit records label every cell producer "${only}", but the inference policy selects ` +
      `${policyVaries.size > 2 ? "several different sources" : "more than one source"} across these cells. ` +
      `variables[].producer is not currently able to distinguish persistence from a model prediction; ` +
      `the source column below is taken from the policy instead.`;
  }

  const count = (kind: ServedBy) => cells.filter((c) => c.servedBy === kind).length;

  return {
    horizons,
    cells,
    totalCells: cells.length,
    persistenceCells: count("PERSISTENCE"),
    modelCells: count("MODEL"),
    derivedCells: count("DERIVED"),
    unknownCells: count("UNKNOWN"),
    conflictCells: cells.filter((c) => c.labelConflict).length,
    producerLabelTrustworthy: !constantLabel,
    producerLabelWarning,
  };
}

/* ────────────────────────────────────────────────────────────────────────────
 * Carried-forward detection
 * ──────────────────────────────────────────────────────────────────────────── */

/**
 * Tolerance for "this number is the observation, moved forward".
 *
 * Deliberately tight. Both sides are IEEE doubles that came out of the same
 * JSON encoder, so a true carry is bit-identical; 1e-9 only absorbs the
 * rounding a server-side JSON round-trip can introduce. It is far too tight to
 * call two genuinely distinct predictions "the same".
 */
export const CARRY_FORWARD_EPSILON = 1e-9;

export interface CarryForwardVerdict {
  variable: VariableName;
  publishedValue: number | null;
  observationValue: number | null;
  /** True only when both numbers exist and agree to within the epsilon. */
  carriedForward: boolean;
  /** True when the published value differs from the origin observation. */
  diverged: boolean;
  /** Signed published minus observed, in the audit unit. */
  delta: number | null;
  auditUnit: string;
}

/**
 * Decide, per variable, whether a published number is simply the origin
 * observation carried forward.
 *
 * This is what makes the persistence distinction visible on the number itself
 * and not just in a source column: a cell the policy serves from persistence
 * whose value nevertheless differs from the observation is a finding, and is
 * reported as `diverged` rather than quietly assumed to be a carry.
 *
 * Returns one verdict per audited variable, always, so a missing observation is
 * an explicit "cannot tell" rather than a missing row.
 */
export function assessCarryForward(
  record: AuditRecord | null | undefined,
  originObservation: OriginObservation | null | undefined
): CarryForwardVerdict[] {
  const variables = record?.variables ?? {};
  const telemetry = originObservation?.telemetry ?? {};

  return AUDITED_VARIABLES.map((variable) => {
    const spec = VARIABLE_UNITS[variable];
    const entry = variables[variable] ?? null;
    const published = finiteOrNull(entry?.value);
    const observed = finiteOrNull(telemetry[spec.telemetryKey]);

    if (published === null || observed === null) {
      return {
        variable,
        publishedValue: published,
        observationValue: observed,
        carriedForward: false,
        diverged: false,
        delta: null,
        auditUnit: spec.auditUnit,
      };
    }

    const delta = published - observed;
    const carriedForward = Math.abs(delta) <= CARRY_FORWARD_EPSILON;

    return {
      variable,
      publishedValue: published,
      observationValue: observed,
      carriedForward,
      diverged: !carriedForward,
      delta,
      auditUnit: spec.auditUnit,
    };
  });
}

function finiteOrNull(value: unknown): number | null {
  if (value === null || value === undefined || typeof value === "boolean") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Verified accuracy, per variable. Never pooled.
 * ──────────────────────────────────────────────────────────────────────────── */

export interface ProducerAccuracy {
  producer: string;
  kind: SourceKind;
  n: number;
  /** Mean absolute error in the variable's audit unit, or null when n is 0. */
  meanAbsError: number | null;
  meanSignedError: number | null;
}

export interface VerifiedVariableAccuracy {
  variable: VariableName;
  label: string;
  metric: MetricKind;
  /** Unit the mean absolute error below is expressed in. */
  auditUnit: string;
  canonicalUnit: string;
  /** Verbatim accumulator from `verify()`; null when nothing was scored. */
  reportedMae: number | null;
  /** Recomputed from the bounded sample; may be null if the sample is empty. */
  sampledMae: number | null;
  sampledMeanSignedError: number | null;
  sampledN: number;
  sampledNByProducer: Record<string, number>;
  /** Mean absolute error converted to the canonical unit, or null. */
  canonicalMae: number | null;
  byProducer: ProducerAccuracy[];
  /** The panel shows this instead of a number when nothing has been scored. */
  emptyReason: string | null;
}

export interface VerifiedAccuracyTable {
  variables: VerifiedVariableAccuracy[];
  scored: number;
  pending: number;
  unmatched: number;
  alreadyVerified: number;
  toleranceMinutes: number | null;
  exists: boolean;
  sourcePath: string | null;
  coverage: ReadCoverage | null;
  /** True when nothing has been scored yet. Drives the explicit empty state. */
  hasNoVerifiedHistory: boolean;
  summary: string;
}

function mean(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const total = values.reduce((acc, v) => acc + v, 0);
  const m = total / values.length;
  return Number.isFinite(m) ? m : null;
}

export interface BuildVerifiedTableInput {
  verification: VerificationSection | null | undefined;
  /** Bounded verification records, used to recompute per-variable statistics. */
  records?: readonly VerificationRecord[];
}

/**
 * Build the per-variable verified accuracy table.
 *
 * STRUCTURAL GUARANTEE: the result is a flat list of rows, one per variable,
 * each carrying its own `auditUnit` and `canonicalUnit`. There is no aggregate
 * field, and no code path here combines values from two variables. This is the
 * UI-side counterpart to the absent `mae_overall` in `verify_predictions.py`.
 *
 * A variable with no scored samples gets `reportedMae: null` and a non-null
 * `emptyReason`. It never gets 0.
 */
export function buildVerifiedAccuracyTable(
  input: BuildVerifiedTableInput
): VerifiedAccuracyTable {
  const section = input.verification ?? null;
  const records = input.records ?? section?.records ?? [];

  const scored = finiteOrNull(section?.scored) ?? 0;
  const pending = finiteOrNull(section?.pending) ?? 0;
  const unmatched = finiteOrNull(section?.unmatched) ?? 0;
  const alreadyVerified = finiteOrNull(section?.alreadyVerified) ?? 0;

  const variables: VerifiedVariableAccuracy[] = AUDITED_VARIABLES.map((variable) => {
    const spec = VARIABLE_UNITS[variable];

    const absByProducer = new Map<string, number[]>();
    const signedByProducer = new Map<string, number[]>();
    const absAll: number[] = [];
    const signedAll: number[] = [];

    for (const record of records) {
      const scoredVar = record?.variables?.[variable] ?? null;
      const abs = finiteOrNull(scoredVar?.abs_error);
      if (abs === null) continue;
      const producer = (scoredVar?.producer ?? "unknown").trim() || "unknown";
      if (!absByProducer.has(producer)) absByProducer.set(producer, []);
      if (!signedByProducer.has(producer)) signedByProducer.set(producer, []);
      absByProducer.get(producer)!.push(abs);
      absAll.push(abs);
      const signed = finiteOrNull(scoredVar?.error);
      if (signed !== null) {
        signedAll.push(signed);
        signedByProducer.get(producer)!.push(signed);
      }
    }

    const reportedMae = finiteOrNull(section?.maeByVariable?.[variable]);
    const reportedByProducer = section?.maeByVariableAndProducer?.[variable] ?? undefined;
    const reportedNByProducer = section?.nByVariableAndProducer?.[variable] ?? undefined;

    const byProducer: ProducerAccuracy[] = [...absByProducer.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([producer, absList]) => ({
        producer,
        kind: normaliseProducer(producer),
        // Prefer the full-trail accumulator; fall back to the bounded sample.
        n: finiteOrNull(reportedNByProducer?.[producer]) ?? absList.length,
        meanAbsError: finiteOrNull(reportedByProducer?.[producer]) ?? mean(absList),
        meanSignedError: mean(signedByProducer.get(producer) ?? []),
      }));

    const sampledMae = mean(absAll);

    let emptyReason: string | null = null;
    if (records.length === 0) {
      emptyReason = !section?.exists
        ? "prediction_verification.jsonl does not exist yet — verify_predictions.py has not scored any forecast."
        : "no verification records were returned in the bounded tail read.";
    } else if (absAll.length === 0) {
      emptyReason =
        "no matured forecast in the read window carries a score for this variable.";
    }

    return {
      variable,
      label: spec.label,
      metric: spec.metric,
      auditUnit: spec.auditUnit,
      canonicalUnit: spec.canonicalUnit,
      reportedMae,
      sampledMae,
      sampledMeanSignedError: mean(signedAll),
      sampledN: absAll.length,
      sampledNByProducer: Object.fromEntries(
        [...absByProducer.entries()].map(([producer, list]) => [producer, list.length])
      ),
      canonicalMae: toCanonicalUnit(variable, reportedMae ?? sampledMae),
      byProducer,
      emptyReason,
    };
  });

  const hasNoVerifiedHistory = scored === 0 || variables.every((v) => v.sampledN === 0 && v.reportedMae === null);

  const summary = !section?.exists
    ? "No verification trail. Nothing has been scored against observations yet."
    : hasNoVerifiedHistory
      ? `0 forecasts scored, ${pending} matured-awaiting-score, ${unmatched} unmatched. No measured error exists for any variable.`
      : `${scored} forecasts scored against observations; per-variable mean absolute error below.`;

  return {
    variables,
    scored,
    pending,
    unmatched,
    alreadyVerified,
    toleranceMinutes: finiteOrNull(section?.toleranceMinutes),
    exists: Boolean(section?.exists),
    sourcePath: section?.sourcePath ?? null,
    coverage: section?.coverage ?? null,
    hasNoVerifiedHistory,
    summary,
  };
}

/* ────────────────────────────────────────────────────────────────────────────
 * Expected (offline benchmark) accuracy, per variable
 * ──────────────────────────────────────────────────────────────────────────── */

export type ExpectedCellStatus = ExpectedAccuracyCell["status"];

export interface ExpectedVariableAccuracy {
  variable: string;
  label: string;
  metric: MetricKind;
  canonicalUnit: string;
  auditUnit: string | null;
  /** True when this variable is not an audited `variables[]` cell. */
  outsideAuditedCells: boolean;
  cells: ExpectedAccuracyCell[];
  measuredCells: number;
  /** Cells where the benchmark found no skill over persistence. */
  persistenceEquivalentCells: number;
  unmeasuredCells: number;
}

/**
 * Shape the static reference into rows. Per variable only, same rule as the
 * verified table: nothing is ever combined.
 */
export function buildExpectedAccuracyTable(
  section: ExpectedAccuracySection | null | undefined
): ExpectedVariableAccuracy[] {
  const source = section?.variables ?? {};
  const names = [...new Set([...AUDITED_VARIABLES, RAIN_VARIABLE.variable])].filter((name) =>
    Object.prototype.hasOwnProperty.call(source, name)
  );

  return names.map((name) => {
    const isAudited = (AUDITED_VARIABLES as readonly string[]).includes(name);
    const spec = isAudited ? VARIABLE_UNITS[name as VariableName] : null;
    const cells = [...(source[name] ?? [])].sort((a, b) => a.horizonHours - b.horizonHours);

    return {
      variable: name,
      label: spec?.label ?? RAIN_VARIABLE.label,
      metric: spec?.metric ?? RAIN_VARIABLE.metric,
      canonicalUnit: spec?.canonicalUnit ?? RAIN_VARIABLE.canonicalUnit,
      auditUnit: spec?.auditUnit ?? null,
      outsideAuditedCells: !isAudited,
      cells,
      measuredCells: cells.filter((c) => c.status === "MEASURED").length,
      persistenceEquivalentCells: cells.filter((c) => c.status === "PERSISTENCE_EQUIVALENT")
        .length,
      unmeasuredCells: cells.filter((c) => c.status === "NOT_MEASURED").length,
    };
  });
}

/**
 * The measured percentage by which the model beat persistence, if the benchmark
 * recorded one.
 *
 * Returns `null` for `NOT_MEASURED` and for `PERSISTENCE_EQUIVALENT`; the panel
 * renders those as different words rather than as 0%.
 */
export function skillOverPersistence(
  section: ExpectedAccuracySection | null | undefined,
  variable: string,
  horizonHours: number
): number | null {
  const cells = section?.variables?.[variable] ?? [];
  const cell = cells.find((c) => c.horizonHours === horizonHours);
  if (!cell || cell.status !== "MEASURED") return null;
  return finiteOrNull(cell.beatsPersistencePct);
}

/* ────────────────────────────────────────────────────────────────────────────
 * Small display helpers. Kept here so they are testable and so the panel has no
 * ad-hoc formatting of its own.
 * ──────────────────────────────────────────────────────────────────────────── */

/** `null` becomes an em dash, never `0`. */
export function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function shortSha(value: string | null | undefined, length = 12): string {
  if (!value) return "—";
  return value.length <= length ? value : value.slice(0, length);
}

/** Human byte size for the read-coverage line. */
export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB"];
  let scaled = value / 1024;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toFixed(scaled < 10 ? 2 : 1)} ${units[unit]}`;
}

/**
 * A compact UTC stamp. Operators reading an audit trail want one timezone and
 * one format; local time in a browser is the wrong answer to that question.
 */
export function formatUtc(value: string | null | undefined): string {
  if (!value) return "—";
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return "—";
  return `${new Date(ms).toISOString().replace("T", " ").replace(/\.\d+Z$/, "Z")}`;
}

export function formatMinutes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (Math.abs(value) < 1) return `${(value * 60).toFixed(0)}s`;
  if (Math.abs(value) < 90) return `${value.toFixed(1)}m`;
  return `${(value / 60).toFixed(1)}h`;
}

/** Difference between two timestamps, in minutes. Null when either is unusable. */
export function minutesBetween(
  later: string | null | undefined,
  earlier: string | null | undefined
): number | null {
  const a = later ? Date.parse(later) : Number.NaN;
  const b = earlier ? Date.parse(earlier) : Number.NaN;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return (a - b) / 60000;
}

/**
 * Describe why an accuracy cell is blank, in words an operator can act on.
 *
 * The four cases are deliberately distinct. "Not measured" and "measured, no
 * skill" are different findings and must not collapse into a blank cell.
 */
export function describeCellGap(
  status: ExpectedCellStatus,
  context: { verifiedExists: boolean; pending: number }
): string {
  switch (status) {
    case "MEASURED":
      return "";
    case "PERSISTENCE_EQUIVALENT":
      return "no skill over persistence — measured, and indistinguishable from carrying the origin value forward";
    case "NOT_MEASURED":
    default:
      return context.verifiedExists
        ? "not measured on the held-out split; no offline figure released"
        : `not measured; ${context.pending} matured forecast(s) still awaiting score`;
  }
}

/** Coverage sentence for the read footer. Never claims completeness it lacks. */
export function describeCoverage(coverage: ReadCoverage | null | undefined): string {
  if (!coverage || coverage.mode === "unavailable") {
    return "Audit trail not readable from this process.";
  }
  if (coverage.mode === "tail" && coverage.coverageIncomplete) {
    return (
      `Tail read ${formatBytes(coverage.bytesScanned)} of ${formatBytes(coverage.fileBytes)} ` +
      `(${coverage.linesScanned} lines, ${coverage.recordsReturned} records returned). ` +
      `Earlier records exist and were not read.`
    );
  }
  if (coverage.mode === "tail") {
    return (
      `Tail read ${formatBytes(coverage.bytesScanned)} (${coverage.linesScanned} lines, ` +
      `${coverage.recordsReturned} records returned).`
    );
  }
  return (
    `Full read: ${coverage.recordsReturned} records from ${formatBytes(coverage.fileBytes)}, ` +
    `covering ${formatUtc(coverage.earliestIncludedUtc)} to ${formatUtc(coverage.latestIncludedUtc)}.`
  );
}
