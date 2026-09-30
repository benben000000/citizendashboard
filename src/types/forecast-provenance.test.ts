/**
 * Tests for the forecast provenance / verified-accuracy contract.
 *
 * RUN
 * ---
 * The repo has no test runner configured (`package.json` -> `test` is an echo,
 * vitest is not installed). These run on the Node built-in runner, which strips
 * TypeScript types natively:
 *
 *     node --test src/types/forecast-provenance.test.ts
 *
 * They import only dependency-free modules, so no path-alias loader is needed.
 *
 * WHAT IS BEING ASSERTED
 * -----------------------
 * The failure this guards against is a dashboard that renders a number without
 * saying where it came from, and a second one that renders a measured-error
 * number that was never measured. Both are quiet failures: the page looks fine.
 * So the assertions are about classification and about honest absence, not about
 * layout.
 */

import { test, describe } from "node:test";
import assert from "node:assert/strict";

import type {
  ForecastSourceId,
  ForecastVariable,
  applyChannelQuarantine as ApplyChannelQuarantine,
  buildSourceProvenance as BuildSourceProvenance,
  buildUnavailableProvenance as BuildUnavailableProvenance,
  forecastSourceIdFromToken as ForecastSourceIdFromToken,
  summarizeVerification as SummarizeVerification,
  verifiedAccuracyUnavailable as VerifiedAccuracyUnavailable,
} from "./forecast-provenance";

// The runtime import goes through a `.ts` file URL: the project's
// `moduleResolution: "bundler"` permits extensionless specifiers, but native
// Node ESM does not. The companion `import type` above keeps full type
// checking on everything destructured here.
const {
  FORECAST_VARIABLES,
  applyChannelQuarantine,
  buildSourceProvenance,
  buildUnavailableProvenance,
  forecastSourceIdFromToken,
  summarizeVerification,
  verifiedAccuracyUnavailable,
}: {
  FORECAST_VARIABLES: readonly ForecastVariable[];
  applyChannelQuarantine: typeof ApplyChannelQuarantine;
  buildSourceProvenance: typeof BuildSourceProvenance;
  buildUnavailableProvenance: typeof BuildUnavailableProvenance;
  forecastSourceIdFromToken: typeof ForecastSourceIdFromToken;
  summarizeVerification: typeof SummarizeVerification;
  verifiedAccuracyUnavailable: typeof VerifiedAccuracyUnavailable;
} = await import(new URL("./forecast-provenance.ts", import.meta.url).href);

/**
 * The served policy, transcribed from
 * `prediction-model/data/inference_policy.json` (policy_version 2.0.0).
 * If this drifts from the file, this test is the thing that notices.
 */
const SERVED_POLICY: Record<string, Record<string, string>> = {
  "1h": {
    temperature: "persistence_fallback",
    humidity: "persistence_fallback",
    pressure: "persistence_fallback",
    wind_speed: "persistence_fallback",
    wind_direction: "persistence_fallback",
    heat_index: "derived_from_selected_temp_and_humidity",
  },
  "3h": {
    temperature: "persistence_fallback",
    humidity: "persistence_fallback",
    pressure: "persistence_fallback",
    wind_speed: "persistence_fallback",
    wind_direction: "persistence_fallback",
    heat_index: "derived_from_selected_temp_and_humidity",
  },
  "6h": {
    temperature: "learned_model",
    humidity: "persistence_fallback",
    pressure: "persistence_fallback",
    wind_speed: "learned_model",
    wind_direction: "persistence_fallback",
    heat_index: "derived_from_selected_temp_and_humidity",
  },
  "12h": {
    temperature: "learned_model",
    humidity: "persistence_fallback",
    pressure: "persistence_fallback",
    wind_speed: "learned_model",
    wind_direction: "persistence_fallback",
    heat_index: "derived_from_selected_temp_and_humidity",
  },
  "24h": {
    temperature: "persistence_fallback",
    humidity: "persistence_fallback",
    pressure: "persistence_fallback",
    wind_speed: "learned_model",
    wind_direction: "persistence_fallback",
    heat_index: "derived_from_selected_temp_and_humidity",
  },
};

/** Convenience: classify one variable at one horizon. */
function sourceAt(horizon: string, variable: string): ForecastSourceId {
  const provenance = buildSourceProvenance({
    horizon,
    selectedSourceByVariable: SERVED_POLICY[horizon],
  });
  const found = provenance.variables.find((v) => v.variable === variable);
  assert.ok(found, `${variable} missing from provenance`);
  return found.sourceId;
}

/* -------------------------------------------------------------------------- */
/* Token classification — fail closed                                          */
/* -------------------------------------------------------------------------- */

describe("forecastSourceIdFromToken", () => {
  test("classifies every token the engine emits", () => {
    assert.equal(forecastSourceIdFromToken("learned_model"), "learned_model");
    assert.equal(forecastSourceIdFromToken("persistence_fallback"), "persistence_fallback");
    assert.equal(
      forecastSourceIdFromToken("derived_from_selected_temp_and_humidity"),
      "derived"
    );
    assert.equal(
      forecastSourceIdFromToken("persistence_fallback_input_quarantined"),
      "suppressed"
    );
  });

  test("an unrecognised token is unknown, never model output", () => {
    // A future policy token must not be silently read as skill.
    assert.equal(forecastSourceIdFromToken("nwp_ecmwf_ifs025"), "unknown");
    assert.equal(forecastSourceIdFromToken(""), "unknown");
    assert.equal(forecastSourceIdFromToken("LEARNED_MODEL"), "unknown");
    assert.equal(forecastSourceIdFromToken(undefined), "unknown");
    assert.equal(forecastSourceIdFromToken(null), "unknown");
    assert.equal(forecastSourceIdFromToken(42), "unknown");
    assert.equal(forecastSourceIdFromToken({}), "unknown");
  });
});

/* -------------------------------------------------------------------------- */
/* Per-variable provenance                                                     */
/* -------------------------------------------------------------------------- */

describe("buildSourceProvenance", () => {
  test("reports all six policy variables, in stable order", () => {
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
    });
    assert.deepEqual(
      provenance.variables.map((v) => v.variable),
      [...FORECAST_VARIABLES]
    );
    assert.equal(provenance.totalCount, 6);
    assert.equal(provenance.horizon, "6h");
  });

  test("1h: every policy variable is carried forward, none is model output", () => {
    const provenance = buildSourceProvenance({
      horizon: "1h",
      selectedSourceByVariable: SERVED_POLICY["1h"],
    });
    assert.equal(provenance.modelOutputCount, 0);
    assert.equal(provenance.carriedForwardCount, 5);
    assert.equal(provenance.derivedCount, 1);
    assert.equal(provenance.modelBackedLabel, "0 of 6");
    // The headline failure mode: a 1h forecast rendered as if it were modelled.
    assert.ok(
      provenance.variables.every((v) => !v.isModelOutput),
      "no 1h variable may be presented as model output"
    );
  });

  test("6h and 12h: exactly temperature and wind speed are model output", () => {
    for (const horizon of ["6h", "12h"]) {
      const provenance = buildSourceProvenance({
        horizon,
        selectedSourceByVariable: SERVED_POLICY[horizon],
      });
      assert.equal(provenance.modelOutputCount, 2, horizon);
      assert.equal(provenance.carriedForwardCount, 3, horizon);
      assert.equal(provenance.derivedCount, 1, horizon);
      assert.equal(sourceAt(horizon, "temperature"), "learned_model", horizon);
      assert.equal(sourceAt(horizon, "windSpeed"), "learned_model", horizon);
      assert.equal(sourceAt(horizon, "humidity"), "persistence_fallback", horizon);
      assert.equal(sourceAt(horizon, "pressure"), "persistence_fallback", horizon);
      assert.equal(sourceAt(horizon, "windDirection"), "persistence_fallback", horizon);
    }
  });

  test("24h: only wind speed is model output; temperature falls back", () => {
    const provenance = buildSourceProvenance({
      horizon: "24h",
      selectedSourceByVariable: SERVED_POLICY["24h"],
    });
    assert.equal(provenance.modelOutputCount, 1);
    assert.equal(provenance.carriedForwardCount, 4);
    assert.equal(sourceAt("24h", "windSpeed"), "learned_model");
    assert.equal(sourceAt("24h", "temperature"), "persistence_fallback");
  });

  test("the served policy serves 20 of 25 model-or-persistence cells by persistence", () => {
    // The claim this change exists to make honest. heat_index is always derived,
    // so it is excluded from the denominator: 5 horizons x 5 policy variables.
    let persistence = 0;
    let learned = 0;
    for (const sources of Object.values(SERVED_POLICY)) {
      for (const [variable, token] of Object.entries(sources)) {
        if (variable === "heat_index") continue;
        const id = forecastSourceIdFromToken(token);
        if (id === "persistence_fallback") persistence += 1;
        if (id === "learned_model") learned += 1;
      }
    }
    assert.equal(persistence + learned, 25);
    assert.equal(learned, 5);
    assert.equal(persistence, 20);
  });

  test("heat index is never counted as model output", () => {
    for (const horizon of Object.keys(SERVED_POLICY)) {
      assert.equal(sourceAt(horizon, "heatIndex"), "derived", horizon);
    }
  });

  test("the raw policy token is preserved verbatim for audit", () => {
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
    });
    const heatIndex = provenance.variables.find((v) => v.variable === "heatIndex");
    assert.equal(heatIndex?.rawToken, "derived_from_selected_temp_and_humidity");
    assert.equal(heatIndex?.sourceId, "derived");
  });

  test("a missing token reads as unknown, not as persistence or as model", () => {
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: { temperature: "learned_model" },
    });
    const missing = provenance.variables.filter((v) => v.sourceId === "unknown");
    assert.equal(missing.length, 5);
    assert.equal(provenance.modelOutputCount, 1);
    assert.equal(provenance.carriedForwardCount, 0);
  });

  test("a wholly absent selection block is all-unknown, not all-persistence", () => {
    const provenance = buildSourceProvenance({ horizon: "1h" });
    assert.equal(provenance.modelOutputCount, 0);
    assert.equal(provenance.carriedForwardCount, 0);
    assert.equal(provenance.unknownCount, 6);
    assert.ok(provenance.variables.every((v) => v.rawToken === null));
  });

  test("learned outputs are downgraded when the engine's quality gate failed", () => {
    // Defence in depth: inference.py rewrites the token when it quarantines the
    // input, but if it ever failed to, the served value is still the origin
    // observation and must not read as modelled.
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
      learnedOutputTrusted: false,
    });
    assert.equal(provenance.modelOutputCount, 0);
    assert.equal(provenance.suppressedCount, 2);
    assert.equal(sourceAt("6h", "temperature"), "learned_model"); // control

    const gated = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
      learnedOutputTrusted: false,
    });
    assert.equal(
      gated.variables.find((v) => v.variable === "temperature")?.sourceId,
      "suppressed"
    );
  });

  test("isCarriedForward covers both plain persistence and a suppressed gate", () => {
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: {
        ...SERVED_POLICY["6h"],
        temperature: "persistence_fallback_input_quarantined",
      },
    });
    const temperature = provenance.variables.find((v) => v.variable === "temperature");
    assert.equal(temperature?.sourceId, "suppressed");
    assert.equal(temperature?.isCarriedForward, true);
    assert.equal(temperature?.isModelOutput, false);
    // Suppressed is neither counted as plain persistence nor as model output.
    assert.equal(provenance.carriedForwardCount, 3);
    assert.equal(provenance.suppressedCount, 1);
  });

  test("the rain blend reports whether any model weight survives", () => {
    const blended = buildSourceProvenance({
      horizon: "1h",
      selectedSourceByVariable: SERVED_POLICY["1h"],
      rainProbabilitySource: "hybrid_blend",
      rainModelWeight: 0.8,
      rainPersistenceWeight: 0.2,
    });
    assert.equal(blended.rain.includesModelOutput, true);
    assert.equal(blended.rain.modelWeight, 0.8);

    const pure = buildSourceProvenance({
      horizon: "1h",
      selectedSourceByVariable: SERVED_POLICY["1h"],
      rainProbabilitySource: "persistence",
      rainModelWeight: 0,
      rainPersistenceWeight: 1,
    });
    assert.equal(pure.rain.includesModelOutput, false);
  });

  test("a non-numeric rain weight does not become a fabricated zero", () => {
    const provenance = buildSourceProvenance({
      horizon: "1h",
      selectedSourceByVariable: SERVED_POLICY["1h"],
      rainModelWeight: Number.NaN,
      rainPersistenceWeight: undefined,
    });
    assert.equal(provenance.rain.modelWeight, null);
    assert.equal(provenance.rain.persistenceWeight, null);
    assert.equal(provenance.rain.includesModelOutput, false);
  });
});

describe("applyChannelQuarantine", () => {
  test("a quarantined sensor downgrades a learned_model label to suppressed", () => {
    // inference.py serves the last observation on a dead anemometer but leaves
    // the policy token at `learned_model`. The label must follow the number.
    const before = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
    });
    assert.equal(before.modelOutputCount, 2);

    const after = applyChannelQuarantine(before, "windSpeed", true);
    const wind = after.variables.find((v) => v.variable === "windSpeed");
    assert.equal(wind?.sourceId, "suppressed");
    assert.equal(wind?.isModelOutput, false);
    assert.equal(wind?.isCarriedForward, true);
    assert.equal(after.modelOutputCount, 1);
    assert.equal(after.modelBackedLabel, "1 of 6");
    // Temperature is untouched.
    assert.equal(
      after.variables.find((v) => v.variable === "temperature")?.sourceId,
      "learned_model"
    );
  });

  test("is a no-op when the sensor is healthy", () => {
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
    });
    assert.equal(applyChannelQuarantine(provenance, "windSpeed", false), provenance);
  });

  test("is a no-op when the channel was not model output to begin with", () => {
    // 1h wind is persistence already; a quarantine warning changes nothing.
    const provenance = buildSourceProvenance({
      horizon: "1h",
      selectedSourceByVariable: SERVED_POLICY["1h"],
    });
    assert.equal(applyChannelQuarantine(provenance, "windSpeed", true), provenance);
    assert.equal(provenance.modelOutputCount, 0);
  });

  test("never drives the model count negative on a fully-persistence horizon", () => {
    const provenance = buildSourceProvenance({
      horizon: "1h",
      selectedSourceByVariable: SERVED_POLICY["1h"],
    });
    const after = applyChannelQuarantine(provenance, "windSpeed", true);
    assert.ok(after.modelOutputCount >= 0);
    assert.equal(after.modelOutputCount, 0);
  });

  test("an unknown channel name leaves the block untouched", () => {
    const provenance = buildSourceProvenance({
      horizon: "6h",
      selectedSourceByVariable: SERVED_POLICY["6h"],
    });
    const after = applyChannelQuarantine(
      provenance,
      "solar_radiation" as ForecastVariable,
      true
    );
    assert.equal(after, provenance);
  });
});

describe("buildUnavailableProvenance", () => {
  test("claims nothing is model output when no forecast is attached", () => {
    const provenance = buildUnavailableProvenance("CACHE_FILE_MISSING");
    assert.equal(provenance.modelOutputCount, 0);
    assert.equal(provenance.totalCount, 6);
    assert.equal(provenance.unknownCount, 6);
    assert.ok(provenance.variables.every((v) => !v.isModelOutput && !v.isCarriedForward));
  });
});

/* -------------------------------------------------------------------------- */
/* Verified accuracy — honest absence                                         */
/* -------------------------------------------------------------------------- */

const NOW = Date.parse("2026-09-30T12:00:00.000Z");
const OPTIONS = { observedAt: "2026-09-30T11:30:00.000Z", now: NOW, maxAgeSeconds: 86400 };

describe("summarizeVerification", () => {
  test("reports available MAE per variable AND per producer, with n", () => {
    const result = summarizeVerification(
      {
        scored: 40,
        pending: 12,
        unmatched: 1,
        tolerance_minutes: 30,
        mae_by_variable: { temperature: 1.477 },
        n_by_variable_and_producer: {
          temperature: { lln: 30, persistence: 10 },
          wind_speed: { lln: 10 },
        },
        mae_by_variable_and_producer: {
          temperature: { lln: 1.477, persistence: 1.698 },
          wind_speed: { lln: 1.217 },
        },
      },
      OPTIONS
    );

    assert.equal(result.status, "available");
    assert.equal(result.reason, null);
    assert.equal(result.scoredCount, 40);
    assert.equal(result.pendingCount, 12);
    assert.equal(result.unmatchedCount, 1);
    assert.equal(result.toleranceMinutes, 30);
    assert.equal(result.entries.length, 3);
    // Sorted by variable, then producer — a stable table is a diffable table.
    assert.deepEqual(
      result.entries.map((e) => `${e.variable}/${e.producer}`),
      ["temperature/lln", "temperature/persistence", "wind_speed/lln"]
    );
    assert.equal(result.entries[0].mae, 1.477);
    assert.equal(result.entries[0].sampleCount, 30);
    assert.equal(result.entries[0].unit, "°C");
    assert.equal(result.entries[2].unit, "m/s");
    assert.equal(result.observedAt, OPTIONS.observedAt);
    assert.equal(result.ageSeconds, 1800);
  });

  test("an unlisted variable gets no invented unit", () => {
    const result = summarizeVerification(
      {
        scored: 1,
        n_by_variable_and_producer: { solar_radiation: { lln: 1 } },
        mae_by_variable_and_producer: { solar_radiation: { lln: 12.5 } },
      },
      OPTIONS
    );
    assert.equal(result.status, "available");
    assert.equal(result.entries[0].unit, null);
  });

  test("MAE is never pooled across variables", () => {
    const result = summarizeVerification(
      {
        scored: 2,
        n_by_variable_and_producer: { pressure: { lln: 2 } },
        mae_by_variable_and_producer: { pressure: { lln: 4.0 } },
      },
      OPTIONS
    );
    assert.equal(result.pooledMae, null);
  });

  test("the live no-score state reports pending, not zero error", () => {
    // This is the artifact's actual shape today: predictions are being made,
    // none have matured past their target time yet.
    const result = summarizeVerification(
      {
        predictions_seen: 81,
        observations_seen: 237,
        scored: 0,
        pending: 81,
        unmatched: 0,
        already_verified: 0,
        tolerance_minutes: 30,
        mae_by_variable: {},
        n_by_variable_and_producer: {},
        mae_by_variable_and_producer: {},
      },
      OPTIONS
    );

    assert.equal(result.status, "no_scored_verifications");
    assert.equal(result.entries.length, 0);
    assert.equal(result.scoredCount, 0);
    assert.equal(result.pendingCount, 81);
    assert.ok(result.reason);
    assert.match(result.reason ?? "", /81/);
    // The critical property: no entries, so nothing can be rendered as a number.
    assert.equal(result.entries.filter((e) => e.mae === 0).length, 0);
  });

  test("a null MAE is dropped, never rendered as 0", () => {
    // verify_predictions.py spells "mean of an empty list" as None.
    const result = summarizeVerification(
      {
        scored: 3,
        n_by_variable_and_producer: { temperature: { lln: 0, persistence: 3 } },
        mae_by_variable_and_producer: { temperature: { lln: null, persistence: 1.2 } },
      },
      OPTIONS
    );
    assert.equal(result.status, "available");
    assert.equal(result.entries.length, 1);
    assert.equal(result.entries[0].producer, "persistence");
    assert.equal(result.entries[0].mae, 1.2);
  });

  test("an MAE with a zero or missing sample count is dropped", () => {
    const result = summarizeVerification(
      {
        scored: 3,
        n_by_variable_and_producer: {
          temperature: { lln: 0 },
          humidity: { lln: 5 },
        },
        mae_by_variable_and_producer: {
          temperature: { lln: 0.0 },
          humidity: { lln: 2.0 },
        },
      },
      OPTIONS
    );
    assert.equal(result.entries.length, 1);
    assert.equal(result.entries[0].variable, "humidity");
  });

  test("a stale summary withholds its numbers rather than showing old ones", () => {
    const result = summarizeVerification(
      {
        scored: 40,
        n_by_variable_and_producer: { temperature: { lln: 30 } },
        mae_by_variable_and_producer: { temperature: { lln: 1.477 } },
      },
      { observedAt: "2026-09-01T00:00:00.000Z", now: NOW, maxAgeSeconds: 86400 }
    );
    assert.equal(result.status, "stale");
    assert.equal(result.entries.length, 0);
    assert.ok(result.reason && result.reason.length > 0);
  });

  test("an unparseable artifact is unavailable, not empty-success", () => {
    for (const bad of [null, undefined, 42, "summary", [], true]) {
      const result = summarizeVerification(bad, OPTIONS);
      assert.equal(result.status, "unavailable", `input: ${JSON.stringify(bad)}`);
      assert.equal(result.entries.length, 0);
      assert.ok(result.reason);
    }
  });

  test("a malformed per-producer block does not throw or invent an entry", () => {
    const result = summarizeVerification(
      {
        scored: 1,
        mae_by_variable_and_producer: {
          temperature: "not-an-object",
          humidity: [1, 2, 3],
          pressure: { lln: { nested: true } },
          wind_speed: { lln: 1.217 },
        },
        n_by_variable_and_producer: { wind_speed: { lln: 8 } },
      },
      OPTIONS
    );
    assert.equal(result.status, "available");
    assert.equal(result.entries.length, 1);
    assert.equal(result.entries[0].variable, "wind_speed");
  });

  test("a summary with no timestamp is not declared stale", () => {
    // Only the file mtime is available. Absent mtime means "cannot judge",
    // which must not be reported as "out of date".
    const result = summarizeVerification(
      {
        scored: 1,
        n_by_variable_and_producer: { temperature: { lln: 1 } },
        mae_by_variable_and_producer: { temperature: { lln: 1.4 } },
      },
      { observedAt: null, now: NOW, maxAgeSeconds: 60 }
    );
    assert.equal(result.status, "available");
    assert.equal(result.ageSeconds, null);
  });

  test("a zero freshness budget disables the staleness gate", () => {
    const result = summarizeVerification(
      {
        scored: 1,
        n_by_variable_and_producer: { temperature: { lln: 1 } },
        mae_by_variable_and_producer: { temperature: { lln: 1.4 } },
      },
      { observedAt: "2020-01-01T00:00:00.000Z", now: NOW, maxAgeSeconds: 0 }
    );
    assert.equal(result.status, "available");
  });

  test("the note states the no-pooling rule on every result", () => {
    const available = summarizeVerification(
      {
        scored: 1,
        n_by_variable_and_producer: { temperature: { lln: 1 } },
        mae_by_variable_and_producer: { temperature: { lln: 1.4 } },
      },
      OPTIONS
    );
    const missing = summarizeVerification(null, OPTIONS);
    assert.match(available.note, /never averaged across variables/);
    assert.match(missing.note, /never averaged across variables/);
  });
});

describe("verifiedAccuracyUnavailable", () => {
  test("defaults to unavailable and can be escalated to stale", () => {
    const plain = verifiedAccuracyUnavailable("because");
    assert.equal(plain.status, "unavailable");
    assert.equal(plain.entries.length, 0);
    assert.equal(plain.pooledMae, null);

    const stale = verifiedAccuracyUnavailable("too old", "stale");
    assert.equal(stale.status, "stale");
    assert.equal(stale.entries.length, 0);
  });
});
