/**
 * Tests for the pure derivations behind the provenance panel.
 *
 * The panel's credibility rests entirely on the things it refuses to do:
 * pool MAE across units, render a missing score as zero, or claim a value was
 * predicted when it was carried forward. Each of those has a test here.
 *
 * Run with `node --test` after compiling; see README-endpoint.md for the
 * exact command, since this repo has no test framework installed.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  AUDITED_VARIABLES,
  CARRY_FORWARD_EPSILON,
  VARIABLE_UNITS,
  assessCarryForward,
  buildExpectedAccuracyTable,
  buildSourceMatrix,
  buildVerifiedAccuracyTable,
  classifyCell,
  conversionNote,
  describeCellGap,
  describeCoverage,
  formatBytes,
  formatMetric,
  formatMinutes,
  formatNumber,
  formatUtc,
  minutesBetween,
  normalisePolicySource,
  normaliseProducer,
  parseTailLines,
  selectForecastRecord,
  shortSha,
  skillOverPersistence,
  toCanonicalUnit,
  type AuditRecord,
  type ExpectedAccuracySection,
  type OriginObservation,
  type PolicyDocument,
  type ProvenancePayload,
  type ReadCoverage,
  type VerificationRecord,
  type VerificationSection,
} from "../prediction-provenance-core";

import {
  EXPECTED_ACCURACY_AS_OF,
  buildExpectedAccuracySection,
} from "../expected-accuracy.reference";

/* ── fixtures ──────────────────────────────────────────────────────────────── */

const H6_SOURCES: Record<string, string> = {
  temperature: "learned_model",
  humidity: "persistence_fallback",
  pressure: "persistence_fallback",
  wind_speed: "learned_model",
  wind_direction: "persistence_fallback",
};

const POLICY: PolicyDocument = {
  policyVersion: "2.0.0",
  policyCodeCommit: "cf0a37e239fd6cc5a3a43affb6fe69148ebba7bf",
  policyRegeneratedBy: "prediction-model/src/refit_policy.py",
  horizons: {
    "1": { selectedSources: { temperature: "persistence_fallback", humidity: "persistence_fallback", pressure: "persistence_fallback", wind_speed: "persistence_fallback" } },
    "3": { selectedSources: { temperature: "persistence_fallback", humidity: "persistence_fallback", pressure: "persistence_fallback", wind_speed: "persistence_fallback" } },
    "6": { selectedSources: H6_SOURCES },
    "12": { selectedSources: H6_SOURCES },
    "24": { selectedSources: { temperature: "persistence_fallback", humidity: "persistence_fallback", pressure: "persistence_fallback", wind_speed: "learned_model" } },
  },
};

function makeRecord(overrides: Partial<AuditRecord> = {}): AuditRecord {
  return {
    record_id: "rec-1",
    recorded_at_utc: "2026-09-30T08:36:12.179267Z",
    origin_timestamp_utc: "2026-09-30T07:48:11.817678Z",
    station_id: "KT-8CEE47DC5194",
    device_id: "KT-8CEE47DC5194",
    horizon_hours: 12,
    provenance: {
      model: { bundle: "h12", checkpoint_trained_at: null, seed: null },
      policy: { policy_version: "2.0.0", policy_code_commit: "cf0a37e2" },
      nwp: { applied: false, reason: "router not wired into the live ingestor yet" },
    },
    variables: {
      temperature: { value: 27.19, producer: "lln" },
      humidity: { value: 59.0, producer: "lln" },
      pressure: { value: 1007.0, producer: "lln" },
      wind_speed: { value: 0.4, producer: "lln" },
    },
    sensor_health: { verdict: "ok", reason: "channel behaves like weather", n: 348 },
    extra: { prediction_status: "READY" },
    ...overrides,
  };
}

const ORIGIN_OBS: OriginObservation = {
  observedAtUtc: "2026-09-30T07:48:11.817678Z",
  recordId: "obs-1",
  ageMinutes: 0.42,
  telemetry: { temperature_c: 27.19, humidity_pct: 61.0, pressure_hpa: 1007.0, wind_speed_kmh: 0.4 },
  coverage: null,
};

function makeVerificationRecord(
  station: string,
  variable: "temperature" | "humidity" | "pressure" | "wind_speed",
  predicted: number,
  actual: number,
  producer = "lln"
): VerificationRecord {
  return {
    record_id: `ver-${variable}-${predicted}`,
    prediction_record_id: `p-${variable}-${predicted}`,
    verified_at_utc: "2026-09-30T12:00:00Z",
    station_id: station,
    horizon_hours: 12,
    origin_timestamp_utc: "2026-09-30T00:00:00Z",
    target_timestamp_utc: "2026-09-30T12:00:00Z",
    observation_record_id: "o-1",
    observation_offset_minutes: 3.2,
    policy_version: "2.0.0",
    model_bundle: "h12",
    variables: {
      [variable]: { predicted, actual, error: predicted - actual, abs_error: Math.abs(predicted - actual), producer },
    },
  };
}

function makeCoverage(overrides: Partial<ReadCoverage> = {}): ReadCoverage {
  return {
    mode: "tail",
    bytesScanned: 1048576,
    fileBytes: 16458441,
    linesScanned: 432,
    recordsReturned: 200,
    discardedPartialHead: true,
    truncatedTailLine: false,
    skippedMalformedLines: 0,
    earliestIncludedUtc: "2026-09-30T09:02:11.000000Z",
    latestIncludedUtc: "2026-09-30T08:36:12.179267Z",
    coverageIncomplete: true,
    ...overrides,
  };
}

/* ── bounded tail parsing ──────────────────────────────────────────────────── */

describe("parseTailLines", () => {
  it("reports the partial head a mid-file seek produces and parses the rest", () => {
    // A real tail read starts partway through a record, so the first line is a
    // fragment that cannot parse. It is reported, and the rest still reads.
    const chunk = '"station_id":"KT-8CEE"\n{"c":3}\n{"d":4}\n';
    const result = parseTailLines(chunk);
    assert.equal(result.records.length, 2);
    assert.equal(result.partialHead, true, "a mid-file seek always fragments the first line");
    assert.equal(result.malformed, 1, "the head fragment is counted as malformed, not silently kept");
    assert.equal(result.partialTail, false);
  });

  it("skips malformed lines instead of failing the read", () => {
    const result = parseTailLines('{"a":1}\nnot json\n[1,2,3]\n{"b":2}\n');
    assert.deepEqual(result.records.map((r) => (r as { a?: number }).a ?? (r as { b: number }).b), [1, 2]);
    assert.equal(result.malformed, 2);
  });

  it("treats a final line with no newline as an in-flight write, not a record", () => {
    const result = parseTailLines('{"a":1}\n{"b":2');
    assert.equal(result.records.length, 1, "the half-written record must not be surfaced as whole");
    assert.equal(result.partialTail, true);
  });

  it("handles an empty and whitespace-only chunk", () => {
    assert.equal(parseTailLines("").records.length, 0);
    assert.equal(parseTailLines("\n\n\n").records.length, 0);
  });

  it("skips blank interior lines without inflating the record count", () => {
    const result = parseTailLines('{"a":1}\n\n\n{"b":2}\n');
    assert.equal(result.completeLines, 2);
    assert.equal(result.malformed, 0);
  });
});

/* ── record selection ──────────────────────────────────────────────────────── */

describe("selectForecastRecord", () => {
  const older = makeRecord({ record_id: "a", recorded_at_utc: "2026-09-30T08:00:00Z" });
  const newer = makeRecord({ record_id: "b", recorded_at_utc: "2026-09-30T09:00:00Z" });
  const other = makeRecord({ record_id: "c", station_id: "KT-OTHER", horizon_hours: 6 });

  it("returns null for an empty trail rather than throwing", () => {
    assert.equal(selectForecastRecord([], { stationId: "KT-1" }), null);
  });

  it("picks the newest record for the station and horizon", () => {
    const picked = selectForecastRecord([older, newer, other], {
      stationId: "KT-8CEE47DC5194",
      horizonHours: 12,
    });
    assert.equal(picked?.record_id, "b");
  });

  it("never crosses stations", () => {
    const picked = selectForecastRecord([other], { stationId: "KT-8CEE47DC5194" });
    assert.equal(picked, null);
  });

  it("honours an explicit recordId over recency", () => {
    const picked = selectForecastRecord([older, newer], {
      stationId: "KT-8CEE47DC5194",
      recordId: "a",
    });
    assert.equal(picked?.record_id, "a");
  });

  it("keeps ERROR records reachable; an audit surface must not hide failures", () => {
    const failed = makeRecord({ record_id: "err", extra: { prediction_status: "ERROR" } });
    const picked = selectForecastRecord([failed], { recordId: "err" });
    assert.equal(picked?.record_id, "err");
    assert.equal(picked?.extra?.prediction_status, "ERROR");
  });

  it("breaks a recorded_at tie deterministically on origin then id", () => {
    const x = makeRecord({ record_id: "x", origin_timestamp_utc: "2026-09-30T05:00:00Z" });
    const y = makeRecord({ record_id: "y", origin_timestamp_utc: "2026-09-30T06:00:00Z" });
    assert.equal(selectForecastRecord([x, y], {})?.record_id, "y");
    assert.equal(selectForecastRecord([y, x], {})?.record_id, "y");
  });

  it("treats a float horizon equal to its integer form as a match", () => {
    const rec = makeRecord({ horizon_hours: 12.0 });
    assert.equal(selectForecastRecord([rec], { horizonHours: 12 })?.record_id, "rec-1");
  });
});

/* ── producer and source classification ────────────────────────────────────── */

describe("normaliseProducer / normalisePolicySource", () => {
  it("maps the four documented producer strings", () => {
    assert.equal(normaliseProducer("lln"), "LEARNED_MODEL");
    assert.equal(normaliseProducer("nwp"), "NWP");
    assert.equal(normaliseProducer("blend"), "NWP_BLEND");
    assert.equal(normaliseProducer("persistence"), "PERSISTENCE");
    assert.equal(normaliseProducer("PERSISTENCE_FALLBACK"), "PERSISTENCE");
  });

  it("does not invent a kind for a missing producer", () => {
    assert.equal(normaliseProducer(null), "UNKNOWN");
    assert.equal(normaliseProducer(""), "UNKNOWN");
    assert.equal(normaliseProducer("mystery-9000"), "UNKNOWN");
  });

  it("maps the policy source strings", () => {
    assert.equal(normalisePolicySource("persistence_fallback"), "PERSISTENCE");
    assert.equal(normalisePolicySource("learned_model"), "LEARNED_MODEL");
    assert.equal(normalisePolicySource("derived_from_selected_temp_and_humidity"), "DERIVED");
    assert.equal(normalisePolicySource(undefined), "UNKNOWN");
  });
});

describe("classifyCell", () => {
  it("serves a persistence-selected cell as persistence even when labelled lln", () => {
    const cell = classifyCell(6, "humidity", H6_SOURCES, "lln");
    assert.equal(cell.servedBy, "PERSISTENCE");
    assert.equal(cell.policyKind, "PERSISTENCE");
    assert.equal(cell.recordedKind, "LEARNED_MODEL");
    assert.equal(cell.labelConflict, true);
    assert.match(cell.conflictReason ?? "", /carries it from the origin observation|origin observation/);
  });

  it("serves a learned cell as a model", () => {
    const cell = classifyCell(6, "temperature", H6_SOURCES, "lln");
    assert.equal(cell.servedBy, "MODEL");
    assert.equal(cell.labelConflict, false);
    assert.equal(cell.conflictReason, null);
  });

  it("agrees when the record names persistence and the policy does too", () => {
    const cell = classifyCell(24, "pressure", POLICY.horizons["24"].selectedSources, "persistence");
    assert.equal(cell.servedBy, "PERSISTENCE");
    assert.equal(cell.labelConflict, false);
  });

  it("flags a conflict in the other direction: record says persistence, policy says model", () => {
    const cell = classifyCell(6, "temperature", H6_SOURCES, "persistence");
    assert.equal(cell.servedBy, "MODEL");
    assert.equal(cell.labelConflict, true);
  });

  it("falls back to the recorded producer when the policy has no entry", () => {
    const cell = classifyCell(99, "temperature", null, "lln");
    assert.equal(cell.servedBy, "MODEL");
    assert.equal(cell.policyKind, "UNKNOWN");
    assert.equal(cell.labelConflict, true, "an unknown side is a conflict, not agreement");
  });

  it("is UNKNOWN when neither side knows", () => {
    const cell = classifyCell(6, "temperature", {}, null);
    assert.equal(cell.servedBy, "UNKNOWN");
    assert.equal(cell.labelConflict, true);
  });
});

describe("buildSourceMatrix", () => {
  it("covers every audited variable at every policy horizon", () => {
    const matrix = buildSourceMatrix({ policy: POLICY });
    assert.equal(matrix.horizons.length, 5);
    assert.equal(matrix.totalCells, 5 * AUDITED_VARIABLES.length);
    assert.equal(matrix.cells.length, matrix.totalCells);
  });

  it("counts the persistence cells in the fixture policy", () => {
    const matrix = buildSourceMatrix({ policy: POLICY });
    // h1:4 h3:4 h6:2 h12:2 h24:3
    assert.equal(matrix.persistenceCells, 15);
    assert.equal(matrix.modelCells, 5);
    assert.equal(matrix.persistenceCells + matrix.modelCells, matrix.totalCells);
  });

  it("warns that a constant producer label cannot reflect a varying policy", () => {
    // This is the live condition: mqtt_live_ingestor.py hardcodes producer="lln"
    // for all four variables while the policy varies across cells.
    const observed: Record<string, Record<string, string | null>> = {};
    for (const h of [1, 3, 6, 12, 24]) {
      observed[String(h)] = {
        temperature: "lln",
        humidity: "lln",
        pressure: "lln",
        wind_speed: "lln",
      };
    }
    const matrix = buildSourceMatrix({ policy: POLICY, observedProducers: observed });
    assert.equal(matrix.producerLabelTrustworthy, false);
    assert.match(matrix.producerLabelWarning ?? "", /not currently able to distinguish persistence/);
    assert.equal(matrix.conflictCells, 15, "the 15 persistence cells are all mislabelled lln");
  });

  it("stays quiet when the recorded producer actually tracks the policy", () => {
    const observed: Record<string, Record<string, string | null>> = {};
    for (const [h, cfg] of Object.entries(POLICY.horizons)) {
      observed[h] = {};
      for (const v of AUDITED_VARIABLES) {
        observed[h][v] =
          cfg.selectedSources[v] === "learned_model" ? "lln" : "persistence";
      }
    }
    const matrix = buildSourceMatrix({ policy: POLICY, observedProducers: observed });
    assert.equal(matrix.producerLabelTrustworthy, true);
    assert.equal(matrix.producerLabelWarning, null);
    assert.equal(matrix.conflictCells, 0);
  });

  it("returns an empty matrix rather than throwing when there is no policy", () => {
    const matrix = buildSourceMatrix({ policy: null });
    assert.equal(matrix.totalCells, 0);
    assert.equal(matrix.persistenceCells, 0);
    assert.equal(matrix.unknownCells, 0);
  });
});

/* ── carried-forward detection ─────────────────────────────────────────────── */

describe("assessCarryForward", () => {
  it("flags a value identical to the origin observation as carried forward", () => {
    const verdicts = assessCarryForward(makeRecord(), ORIGIN_OBS);
    const temp = verdicts.find((v) => v.variable === "temperature")!;
    assert.equal(temp.carriedForward, true);
    assert.equal(temp.diverged, false);
    assert.equal(temp.delta, 0);
  });

  it("does not flag a value that differs from the origin observation", () => {
    const verdicts = assessCarryForward(makeRecord(), ORIGIN_OBS);
    const humidity = verdicts.find((v) => v.variable === "humidity")!;
    assert.equal(humidity.carriedForward, false);
    assert.equal(humidity.diverged, true);
    assert.equal(humidity.delta, -2);
  });

  it("returns one verdict per audited variable even with no observation", () => {
    const verdicts = assessCarryForward(makeRecord(), null);
    assert.equal(verdicts.length, AUDITED_VARIABLES.length);
    assert.ok(verdicts.every((v) => v.carriedForward === false && v.diverged === false));
    assert.ok(verdicts.every((v) => v.observationValue === null));
  });

  it("cannot call a value carried forward when the observation is missing", () => {
    const verdicts = assessCarryForward(makeRecord(), { ...ORIGIN_OBS, telemetry: {} });
    const temp = verdicts.find((v) => v.variable === "temperature")!;
    assert.equal(temp.publishedValue, 27.19);
    assert.equal(temp.observationValue, null);
    assert.equal(temp.carriedForward, false);
  });

  it("does not let rounding differences create a false carry", () => {
    const eps = CARRY_FORWARD_EPSILON;
    const record = makeRecord();
    record.variables.temperature = { value: 27.19 + eps * 2, producer: "lln" };
    const temp = assessCarryForward(record, ORIGIN_OBS).find((v) => v.variable === "temperature")!;
    assert.equal(temp.carriedForward, false, "a genuine model output must not read as a carry");
  });

  it("reports the delta in the audit unit, not the canonical unit", () => {
    const record = makeRecord();
    record.variables.wind_speed = { value: 10.0, producer: "lln" };
    const obs = { ...ORIGIN_OBS, telemetry: { ...ORIGIN_OBS.telemetry, wind_speed_kmh: 4.0 } };
    const wind = assessCarryForward(record, obs).find((v) => v.variable === "wind_speed")!;
    assert.equal(wind.auditUnit, "km/h");
    assert.equal(wind.delta, 6);
  });
});

/* ── verified accuracy ─────────────────────────────────────────────────────── */

describe("buildVerifiedAccuracyTable", () => {
  const emptySection: VerificationSection = {
    sourcePath: "prediction-model/data/prediction_verification.jsonl",
    exists: false,
    scored: 0,
    pending: 81,
    unmatched: 0,
    alreadyVerified: 0,
    toleranceMinutes: 30,
    coverage: null,
    maeByVariable: {},
    maeByVariableAndProducer: {},
    nByVariableAndProducer: {},
    records: [],
  };

  it("reports an explicit empty state instead of a zero error", () => {
    const table = buildVerifiedAccuracyTable({ verification: emptySection });
    assert.equal(table.hasNoVerifiedHistory, true);
    assert.equal(table.exists, false);
    for (const row of table.variables) {
      assert.equal(row.reportedMae, null, `${row.variable} must not show 0 for an unmeasured MAE`);
      assert.equal(row.canonicalMae, null);
      assert.ok(row.emptyReason, `${row.variable} must explain its gap`);
    }
    assert.match(table.summary, /Nothing has been scored/);
  });

  it("names the pending count when the trail exists but nothing has matured", () => {
    const table = buildVerifiedAccuracyTable({
      verification: { ...emptySection, exists: true },
    });
    assert.equal(table.hasNoVerifiedHistory, true);
    assert.match(table.summary, /81 matured-awaiting-score/);
    assert.equal(table.pending, 81);
  });

  it("scores per variable independently", () => {
    const records: VerificationRecord[] = [
      makeVerificationRecord("KT-1", "temperature", 28.0, 26.0), // |e| = 2.0
      makeVerificationRecord("KT-1", "temperature", 28.0, 27.0), // |e| = 1.0
      makeVerificationRecord("KT-1", "pressure", 1008.0, 1000.0), // |e| = 8.0
    ];
    const table = buildVerifiedAccuracyTable({
      verification: { ...emptySection, exists: true, scored: 3, records },
      records,
    });

    const temp = table.variables.find((v) => v.variable === "temperature")!;
    const pressure = table.variables.find((v) => v.variable === "pressure")!;
    assert.equal(temp.sampledMae, 1.5);
    assert.equal(pressure.sampledMae, 8.0);
    assert.equal(table.hasNoVerifiedHistory, false);
  });

  /**
   * The load-bearing test. MAE across degC, %RH, hPa and m/s is not a quantity:
   * the scales do not share a unit, so an hPa error numerically dominates a degC
   * error while meaning no more. `verify_predictions.py` deliberately reports
   * no pooled figure, and the UI must not invent one.
   */
  it("never produces a pooled cross-unit aggregate", () => {
    const records: VerificationRecord[] = [
      makeVerificationRecord("KT-1", "temperature", 28.0, 26.0), // 2.0 degC
      makeVerificationRecord("KT-1", "humidity", 90.0, 60.0), //    30 %RH
      makeVerificationRecord("KT-1", "pressure", 1015.0, 1000.0), // 15 hPa
      makeVerificationRecord("KT-1", "wind_speed", 10.0, 5.0), //    5 km/h
    ];
    const table = buildVerifiedAccuracyTable({
      verification: { ...emptySection, exists: true, scored: 4, records },
      records,
    });

    const absErrors = [2, 30, 15, 5];
    const pooled = absErrors.reduce((a, b) => a + b, 0) / absErrors.length;
    const everyNumber = table.variables.flatMap((row) => [
      row.reportedMae,
      row.sampledMae,
      row.canonicalMae,
      ...row.byProducer.map((p) => p.meanAbsError),
    ]);
    assert.ok(
      everyNumber.every((n) => n === null || Math.abs(n - pooled) > 1e-9),
      "no returned figure may equal the pooled mean across units"
    );

    // One row per variable, each carrying only its own unit.
    assert.equal(table.variables.length, AUDITED_VARIABLES.length);
    const unitByVariable = new Map(table.variables.map((r) => [r.variable, r.auditUnit]));
    assert.equal(unitByVariable.get("temperature"), "degC");
    assert.equal(unitByVariable.get("humidity"), "%RH");
    assert.equal(unitByVariable.get("pressure"), "hPa");
    assert.equal(unitByVariable.get("wind_speed"), "km/h");

    // Three distinct unit-bearing rows means nothing collapsed into one bucket.
    assert.equal(new Set([...unitByVariable.values()]).size, 4);
  });

  it("keeps the per-producer breakdown separate from the per-variable figure", () => {
    // lln: 28 vs 26 -> +2.  nwp: 28 vs 24 -> +4, 20 vs 24 -> -4.
    // Pooling the producers into one MAE would hide that nwp is twice as wrong
    // as lln here, which is the whole reason the producer split exists.
    const records: VerificationRecord[] = [
      makeVerificationRecord("KT-1", "temperature", 28.0, 26.0, "lln"),
      makeVerificationRecord("KT-1", "temperature", 28.0, 24.0, "nwp"),
      makeVerificationRecord("KT-1", "temperature", 20.0, 24.0, "nwp"),
    ];
    const table = buildVerifiedAccuracyTable({
      verification: { ...emptySection, exists: true, scored: 3, records },
      records,
    });
    const temp = table.variables.find((v) => v.variable === "temperature")!;
    assert.equal(temp.byProducer.length, 2);
    const nwp = temp.byProducer.find((p) => p.producer === "nwp")!;
    const lln = temp.byProducer.find((p) => p.producer === "lln")!;
    assert.equal(lln.meanAbsError, 2);
    assert.equal(lln.n, 1);
    assert.equal(nwp.meanAbsError, 4, "|4| and |-4| both count toward MAE");
    assert.equal(nwp.n, 2);
    assert.equal(nwp.meanSignedError, 0, "the nwp bias cancels even though its MAE is high");
    assert.equal(temp.sampledMae, (2 + 4 + 4) / 3);
    assert.equal(temp.sampledN, 3);
  });

  it("prefers the full-trail accumulator over the bounded sample when both exist", () => {
    const records = [makeVerificationRecord("KT-1", "temperature", 28.0, 26.0)];
    const table = buildVerifiedAccuracyTable({
      verification: {
        ...emptySection,
        exists: true,
        scored: 5000,
        records,
        maeByVariable: { temperature: 1.477 },
        maeByVariableAndProducer: { temperature: { lln: 1.477 } },
        nByVariableAndProducer: { temperature: { lln: 5000 } },
      },
      records,
    });
    const temp = table.variables.find((v) => v.variable === "temperature")!;
    assert.equal(temp.reportedMae, 1.477);
    assert.equal(temp.byProducer[0].n, 5000);
    assert.equal(temp.byProducer[0].meanAbsError, 1.477);
  });

  it("survives a completely absent verification section", () => {
    const table = buildVerifiedAccuracyTable({ verification: null });
    assert.equal(table.hasNoVerifiedHistory, true);
    assert.equal(table.exists, false);
    assert.equal(table.variables.length, AUDITED_VARIABLES.length);
    assert.ok(table.variables.every((v) => v.reportedMae === null));
  });

  it("ignores a verification record whose score was never filled in", () => {
    const bad = makeVerificationRecord("KT-1", "humidity", 90, 60);
    bad.variables.humidity = { predicted: 90, actual: null, error: null, abs_error: null, producer: "lln" };
    const table = buildVerifiedAccuracyTable({
      verification: { ...emptySection, exists: true, scored: 1, records: [bad] },
      records: [bad],
    });
    const humidity = table.variables.find((v) => v.variable === "humidity")!;
    assert.equal(humidity.sampledN, 0);
    assert.equal(humidity.reportedMae, null);
    assert.match(humidity.emptyReason ?? "", /no matured forecast/);
  });
});

/* ── units ─────────────────────────────────────────────────────────────────── */

describe("unit handling", () => {
  it("documents that the audit's wind_speed field is km/h, not m/s", () => {
    // Confirmed empirically against all 6,785 live records:
    // variables.wind_speed.value === forecast_raw.wind_speed_kmh exactly.
    assert.equal(VARIABLE_UNITS.wind_speed.auditUnit, "km/h");
    assert.equal(VARIABLE_UNITS.wind_speed.canonicalUnit, "m/s");
    assert.equal(VARIABLE_UNITS.wind_speed.telemetryKey, "wind_speed_kmh");
  });

  it("converts wind km/h to m/s and leaves the other variables untouched", () => {
    assert.equal(toCanonicalUnit("wind_speed", 36), 10);
    assert.equal(toCanonicalUnit("temperature", 1.477), 1.477);
    assert.equal(toCanonicalUnit("pressure", 2), 2);
    assert.equal(toCanonicalUnit("humidity", 3), 3);
  });

  it("converts a missing value to null, never to zero", () => {
    assert.equal(toCanonicalUnit("wind_speed", null), null);
    assert.equal(toCanonicalUnit("wind_speed", undefined), null);
    assert.equal(toCanonicalUnit("wind_speed", Number.NaN), null);
    assert.equal(toCanonicalUnit("wind_speed", Number.POSITIVE_INFINITY), null);
  });

  it("only reports a conversion note when a conversion actually happened", () => {
    assert.equal(conversionNote("wind_speed", "km/h"), "converted from km/h");
    assert.equal(conversionNote("wind_speed", "m/s"), null);
    assert.equal(conversionNote("temperature", "degC"), null);
  });
});

/* ── expected accuracy ─────────────────────────────────────────────────────── */

describe("expected accuracy reference", () => {
  const section = buildExpectedAccuracySection(null);

  it("carries the released offline figures", () => {
    assert.equal(skillOverPersistence(section, "temperature", 6), 13.0);
    assert.equal(skillOverPersistence(section, "temperature", 12), 23.8);
    assert.equal(skillOverPersistence(section, "wind_speed", 6), 7.8);
    assert.equal(skillOverPersistence(section, "wind_speed", 12), 17.8);
    assert.equal(section.asOf, EXPECTED_ACCURACY_AS_OF);
  });

  it("returns null skill for a persistence-equivalent cell, not 0%", () => {
    // The difference matters: 0% would read as "measured, no skill", when in
    // fact these cells carry no measured MAE at all.
    assert.equal(skillOverPersistence(section, "temperature", 1), null);
    assert.equal(skillOverPersistence(section, "temperature", 24), null);
    assert.equal(skillOverPersistence(section, "humidity", 6), null);
    assert.equal(skillOverPersistence(section, "pressure", 6), null);
  });

  it("returns null for a horizon with no released figure", () => {
    assert.equal(skillOverPersistence(section, "wind_speed", 1), null);
    assert.equal(skillOverPersistence(section, "wind_speed", 24), null);
    assert.equal(skillOverPersistence(section, "temperature", 99), null);
  });

  it("marks unmeasured cells distinctly from persistence-equivalent ones", () => {
    const table = buildExpectedAccuracyTable(section);
    const wind = table.find((r) => r.variable === "wind_speed")!;
    assert.equal(wind.unmeasuredCells, 3);
    assert.equal(wind.persistenceEquivalentCells, 0);

    const humidity = table.find((r) => r.variable === "humidity")!;
    assert.equal(humidity.persistenceEquivalentCells, 5);
    assert.equal(humidity.unmeasuredCells, 0);
  });

  it("reports rain occurrence with Brier and flags it as outside the audited cells", () => {
    const table = buildExpectedAccuracyTable(section);
    const rain = table.find((r) => r.variable === "rain_occurrence");
    assert.ok(rain);
    assert.equal(rain.metric, "brier");
    assert.equal(rain.outsideAuditedCells, true);
    assert.equal(rain.measuredCells, 2);
    const values = rain.cells.filter((c) => c.status === "MEASURED").map((c) => c.value);
    assert.deepEqual(values, [0.137, 0.251]);
  });

  it("never interpolates a missing rain horizon", () => {
    const rain = section.variables.rain_occurrence;
    assert.deepEqual(rain.map((c) => c.horizonHours), [1, 24]);
    assert.ok(!rain.some((c) => c.horizonHours === 6));
  });

  it("lets a server-supplied section replace the pinned one wholesale", () => {
    const override: ExpectedAccuracySection = {
      source: "server",
      asOf: "2026-10-01",
      coverageNote: null,
      variables: {
        temperature: [
          { horizonHours: 6, metric: "mae", value: 1.4, beatsPersistencePct: 15, status: "MEASURED", beatsBaselines: ["persistence"] },
        ],
      },
    };
    const merged = buildExpectedAccuracySection(override);
    assert.equal(merged.source, "server");
    assert.deepEqual(Object.keys(merged.variables), ["temperature"]);
    assert.equal(skillOverPersistence(merged, "temperature", 6), 15);
    assert.equal(skillOverPersistence(merged, "humidity", 6), null);
  });

  it("treats an empty server override as absent, keeping the pinned reference", () => {
    const merged = buildExpectedAccuracySection({ source: "", asOf: null, coverageNote: null, variables: {} });
    assert.equal(merged.source.length > 0, true);
    assert.equal(skillOverPersistence(merged, "temperature", 6), 13.0);
  });
});

describe("describeCellGap", () => {
  it("words a measured cell as having no gap", () => {
    assert.equal(describeCellGap("MEASURED", { verifiedExists: false, pending: 0 }), "");
  });

  it("distinguishes 'no skill' from 'not measured'", () => {
    const equivalent = describeCellGap("PERSISTENCE_EQUIVALENT", { verifiedExists: true, pending: 0 });
    const unmeasured = describeCellGap("NOT_MEASURED", { verifiedExists: true, pending: 0 });
    assert.notEqual(equivalent, unmeasured);
    assert.match(equivalent, /no skill over persistence/);
    assert.match(unmeasured, /not measured on the held-out split/);
  });

  it("points at the pending count when nothing has been verified at all", () => {
    assert.match(describeCellGap("NOT_MEASURED", { verifiedExists: false, pending: 81 }), /81 matured/);
  });
});

/* ── coverage and formatting ───────────────────────────────────────────────── */

describe("read coverage honesty", () => {
  it("says so when earlier records were not read", () => {
    const text = describeCoverage(makeCoverage());
    assert.match(text, /Earlier records exist and were not read/);
    // 16,458,441 bytes is the live prediction trail size; ~15.7 MiB.
    assert.match(text, /Tail read 1\.00 MiB of 15\.7 MiB/);
  });

  it("drops the incompleteness clause when the whole file was read", () => {
    const text = describeCoverage(makeCoverage({ mode: "full", coverageIncomplete: false, fileBytes: 2048 }));
    assert.match(text, /Full read/);
    assert.doesNotMatch(text, /were not read/);
  });

  it("does not claim a read that did not happen", () => {
    assert.match(describeCoverage(null), /not readable/);
    assert.match(describeCoverage(makeCoverage({ mode: "unavailable" })), /not readable/);
  });
});

describe("formatting never fabricates a number", () => {
  it("renders a missing metric as an em dash, not as zero", () => {
    assert.equal(formatMetric(null), "—");
    assert.equal(formatMetric(undefined), "—");
    assert.equal(formatMetric(Number.NaN), "—");
    assert.equal(formatMetric(0), "0.000");
    assert.notEqual(formatMetric(null), formatMetric(0), "absent and zero must be distinguishable");
  });

  it("formats numbers, minutes and shas without throwing", () => {
    assert.equal(formatNumber(null), "—");
    assert.equal(formatNumber(1007), "1007.00");
    assert.equal(formatMinutes(null), "—");
    assert.equal(formatMinutes(0.42), "25s");
    assert.equal(formatMinutes(45), "45.0m");
    assert.equal(shortSha(null), "—");
    assert.equal(shortSha("abcdefghijklmnop"), "abcdefghijkl");
    assert.equal(formatUtc(null), "—");
    assert.equal(formatUtc("not-a-date"), "—");
    assert.equal(formatUtc("2026-09-30T08:36:12.179267Z"), "2026-09-30 08:36:12Z");
    assert.equal(minutesBetween(null, "2026-09-30T08:00:00Z"), null);
    assert.equal(minutesBetween("2026-09-30T08:30:00Z", "2026-09-30T08:00:00Z"), 30);
  });

  it("formats byte sizes across magnitudes", () => {
    assert.equal(formatBytes(0), "0 B");
    assert.equal(formatBytes(512), "512 B");
    assert.equal(formatBytes(1024), "1.00 KiB");
    assert.equal(formatBytes(16458441), "15.7 MiB");
    assert.equal(formatBytes(null), "—");
  });
});

/* ── end-to-end on a realistic payload ─────────────────────────────────────── */

describe("full payload derivation", () => {
  const payload: ProvenancePayload = {
    schemaVersion: 1,
    generatedAtUtc: "2026-09-30T10:00:00Z",
    stationId: "KT-8CEE47DC5194",
    forecast: makeRecord(),
    forecastHistory: [makeRecord()],
    forecastCoverage: makeCoverage(),
    verification: {
      sourcePath: "prediction-model/data/prediction_verification.jsonl",
      exists: false,
      scored: 0,
      pending: 81,
      unmatched: 0,
      alreadyVerified: 0,
      toleranceMinutes: 30,
      coverage: null,
      maeByVariable: {},
      maeByVariableAndProducer: {},
      nByVariableAndProducer: {},
      records: [],
    },
    policy: POLICY,
    bundles: {
      h12: {
        bundle: "h12",
        checkpointSha256: "73697f9d10427e7b64a567a6ffb243dcf969e03a5f9eb5538eb5ee01d80fb4e2",
        trainingTimestamp: "2026-09-23T06:49:26.290505+00:00",
        implementationCommit: "b72b16ae",
        modelFamily: "GarciaWeatherLNN",
        modelStatus: "ACTIVE_PRODUCTION",
        uncertaintyStatus: "UNAVAILABLE",
        randomSeed: 42,
      },
    },
    expectedAccuracy: null,
    originObservation: ORIGIN_OBS,
    warnings: ["observation_audit.jsonl was rotated mid-read"],
  };

  it("derives every panel section from one realistic payload", () => {
    const matrix = buildSourceMatrix({ policy: payload.policy, observedProducers: {} });
    assert.equal(matrix.persistenceCells, 15);
    assert.equal(matrix.totalCells, 20);

    const carry = assessCarryForward(payload.forecast, payload.originObservation);
    assert.equal(carry.find((v) => v.variable === "temperature")!.carriedForward, true);

    const verified = buildVerifiedAccuracyTable({ verification: payload.verification });
    assert.equal(verified.hasNoVerifiedHistory, true);
    assert.ok(verified.variables.every((v) => v.reportedMae === null));

    const expected = buildExpectedAccuracyTable(buildExpectedAccuracySection(payload.expectedAccuracy));
    assert.equal(expected.length, 5, "four audited variables plus rain occurrence");
    assert.match(describeCoverage(payload.forecastCoverage), /were not read/);
    assert.equal(payload.warnings.length, 1);
  });
});
