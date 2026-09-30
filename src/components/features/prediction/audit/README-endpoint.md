# Prediction provenance endpoint — contract

`prediction-provenance-panel.tsx` reads exactly one endpoint. This file specifies it.

**No route file is included here on purpose.** `src/app/api/**` and the services
layer are owned by another agent working in parallel, and a new route in a
shared namespace is a collision waiting to happen. The panel is written against
the contract below and will work unchanged once the route exists.

Expected path: **`GET /api/prediction/provenance`**
(the component takes an `endpoint` prop; any same-origin path works).

---

## Query parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `stationId` | string, **required** | — | Station public id, e.g. `KT-8CEE47DC5194` |
| `horizonHours` | number | — | Select the newest record at this horizon |
| `recordId` | string | — | Select this exact audit record, overriding `horizonHours` |
| `maxRecords` | int | `200` | Cap on `forecastHistory` entries |
| `maxVerificationRecords` | int | `500` | Cap on `verification.records` entries |
| `tailBytes` | int | `1048576` (1 MiB) | How far back into the trail to seek |

Unknown parameters are ignored. `recordId` wins over `horizonHours`; the client
sends both and the server resolves the precedence.

---

## Why the server must tail-read

| file | size at time of writing | growth |
|---|---|---|
| `prediction-model/data/prediction_audit.jsonl` | 16.5 MB / 6,785 records | appended continuously by `mqtt_live_ingestor.py` |
| `prediction-model/data/observation_audit.jsonl` | 3.4 MB | same |
| `prediction-model/data/prediction_verification.jsonl` | **does not exist** | written by `verify_predictions.py` |

`prediction_audit.py` rotates at 32 MB × 5 generations, so the trail is bounded
but large and it is gitignored live state. A browser must never fetch it.

Required read algorithm:

1. `fs.stat` the file for the total size.
2. If `size > tailBytes`, `fs.open` and **seek to `size - tailBytes`**, then read
   that window. Otherwise read from 0.
3. Parse only newline-terminated lines.
4. Discard the **first** line if the seek did not land on a line boundary
   (expected in the common case — set `discardedPartialHead: true`).
5. Report the **last** line as `truncatedTailLine: true` when it has no
   terminating newline (an append is in flight).
6. Never hold more than the requested window in memory at once; stream line by
   line if the window is large.

A naive `readFileSync` + `JSON.parse` per line works and is what
`read_records()` does, but it does not scale to 16 MB and growing — hence the
seek.

Rotation hazard: the file may be replaced by `os.replace()` between the `stat`
and the `read`. Retry once on `ENOENT`/`EBADF` and report
`coverage.mode: "unavailable"` if it still fails.

---

## Response

Envelope follows the existing convention (`src/app/api/prediction/mqtt/live/route.ts`):

```jsonc
{
  "success": true,
  "message": "Bounded tail read of the prediction audit trail.",
  "data": { /* ProvenancePayload */ }
}
```

The client validates `success === true` and that `data.stationId` is a non-empty
string; anything else surfaces as an explicit error rather than an empty table.

Field names below are the client's `ProvenancePayload`. The `forecast` sub-object
is the audit record **verbatim** — do not reshape it, the panel renders unknown
fields as evidence that the schema moved.

### `data`

```jsonc
{
  "schemaVersion": 1,
  "generatedAtUtc": "2026-09-30T10:00:00.000Z",
  "stationId": "KT-8CEE47DC5194",

  // The one record being described. verbatim AuditRecord from prediction_audit.jsonl.
  "forecast": { /* see below */ },

  // Newest-last, bounded by maxRecords, already filtered to stationId.
  "forecastHistory": [ /* AuditRecord[] */ ],

  // How the prediction trail was read. Always present, even on failure.
  "forecastCoverage": {
    "mode": "tail",                  // "tail" | "full" | "unavailable"
    "bytesScanned": 1048576,
    "fileBytes": 16458441,           // null when stat failed
    "linesScanned": 432,
    "recordsReturned": 200,
    "discardedPartialHead": true,
    "truncatedTailLine": false,
    "skippedMalformedLines": 0,
    "earliestIncludedUtc": "2026-09-30T09:02:11.000000Z",
    "latestIncludedUtc": "2026-09-30T08:36:12.179267Z",
    "coverageIncomplete": true       // records exist before earliestIncludedUtc
  },

  // Verbatim counters + per-variable accumulators from verify_predictions.py.
  // null when the file does not exist. NEVER a pooled cross-variable figure.
  "verification": {
    "sourcePath": "prediction-model/data/prediction_verification.jsonl",
    "exists": false,
    "scored": 0,
    "pending": 81,
    "unmatched": 0,
    "alreadyVerified": 0,
    "toleranceMinutes": 30,
    "coverage": { /* ReadCoverage | null */ },
    "maeByVariable": {},                       // {"temperature": 1.477, ...}
    "maeByVariableAndProducer": {},            // {"temperature": {"lln": 1.477}}
    "nByVariableAndProducer": {},              // {"temperature": {"lln": 128}}
    "records": [ /* VerificationRecord[] */ ]
  },

  // prediction-model/data/inference_policy.json, reshaped only where noted.
  "policy": {
    "policyVersion": "2.0.0",
    "policyCodeCommit": "cf0a37e2…",
    "policyRegeneratedBy": "prediction-model/src/refit_policy.py",
    // `selected_sources` becomes `selectedSources`. Keys stay horizon-hour strings.
    "horizons": {
      "6": {
        "selectedSources": {
          "temperature": "learned_model",
          "humidity": "persistence_fallback",
          "pressure": "persistence_fallback",
          "wind_speed": "learned_model"
        },
        "rainModelWeight": 0.45,
        "rainPersistenceWeight": 0.55,
        "operationalRainThreshold": 0.1,
        "calibrationCodeCommit": "cf0a37e2…"
      }
    }
  },

  // prediction-model/data/bundles/h{1,3,6,12,24}/bundle_manifest.json
  "bundles": {
    "h12": {
      "bundle": "h12",
      "checkpointSha256": "73697f9d10427e7b64a567a6ffb243dcf969e03a5f9eb5538eb5ee01d80fb4e2",
      "trainingTimestamp": "2026-09-23T06:49:26.290505+00:00",
      "implementationCommit": "b72b16ae960be6627b92dd2445dac10d84b99bef",
      "modelFamily": "GarciaWeatherLNN",
      "modelStatus": "ACTIVE_PRODUCTION",
      "uncertaintyStatus": "UNAVAILABLE",
      "randomSeed": 42
    }
  },

  // The observation the origin observation timestamp refers to, found by a
  // bounded tail read of observation_audit.jsonl. This is what backs the
  // "carried forward from <station> at <time>" claim in the panel.
  "originObservation": {
    "observedAtUtc": "2026-09-30T07:48:11.817678Z",
    "recordId": "…",
    "ageMinutes": 0.42,
    "telemetry": {
      "temperature_c": 27.19,
      "humidity_pct": 59.0,
      "pressure_hpa": 1007.0,
      "wind_speed_kmh": 0.4
    },
    "coverage": { /* ReadCoverage | null */ }
  },

  // Optional. Anything here replaces the pinned reference in
  // expected-accuracy.reference.ts wholesale, per variable.
  "expectedAccuracy": null,

  "warnings": ["observation_audit.jsonl was rotated mid-read; origin observation may be stale"]
}
```

### `data.forecast` (verbatim `AuditRecord`)

Copied straight from `prediction_audit.jsonl`, **including unknown fields**:

```jsonc
{
  "schema_version": 1,
  "record_id": "c212a4a48ec724e3c561",
  "recorded_at_utc": "2026-09-30T08:36:12.179267Z",
  "origin_timestamp_utc": "2026-09-30T07:48:11.817678Z",
  "station_id": "KT-8CEE47DC5194",
  "device_id": "KT-8CEE47DC5194",
  "horizon_hours": 12.0,
  "provenance": {
    "model":  { "bundle": "h12", "checkpoint_trained_at": null, "seed": null },
    "policy": { "policy_version": "2.0.0", "policy_code_commit": "cf0a37e2…",
                "policy_regenerated_by": "prediction-model/src/refit_policy.py" },
    "nwp":    { "applied": false, "reason": "router not wired into the live ingestor yet" }
  },
  "variables": {
    "temperature": { "value": 27.19, "producer": "lln", "nwp_raw": null, "nwp_corrected": null },
    "humidity":    { "value": 59.0,  "producer": "lln", "nwp_raw": null, "nwp_corrected": null },
    "pressure":    { "value": 1007.0,"producer": "lln", "nwp_raw": null, "nwp_corrected": null },
    "wind_speed":  { "value": 0.4,   "producer": "lln", "nwp_raw": null, "nwp_corrected": null }
  },
  "sensor_health": { "verdict": "ok", "reason": "channel behaves like weather", "n": 348 },
  "extra": { "prediction_status": "READY", "horizon_label": "12h", "chance_of_rain_pct": 7.0 },
  "forecast_raw": { /* ... */ }
}
```

Every field that `prediction_audit.write()` can null out is nullable here. The
writer deliberately degrades a field to `null` rather than dropping the record,
so the client types all of them as optional.

---

## Rules the server must not break

1. **No pooled accuracy.** `maeByVariable` and friends stay per-variable, as
   `verify_predictions.py` emits them. Do not add an `maeOverall`. degC, %RH, hPa
   and m/s do not share a scale; a mean across them is not a quantity.
2. **No verification file is not an error.** Return `exists: false` with the
   real `pending` count. Do not synthesise zeros — the panel distinguishes
   "not scored yet" from "scored, error was zero", and that distinction is the
   point of the section.
3. **Report the tail honestly.** `coverageIncomplete: true` whenever records
   exist earlier in the file than the window reached. The panel renders
   "Earlier records exist and were not read" rather than implying completeness.
4. **Do not filter out failures.** Records with `extra.prediction_status: "ERROR"`
   belong in `forecastHistory`; an audit surface that hides them is worse than
   none.
5. **Never return an absolute machine path.** `verify_predictions.py` already
   rewrites `writer.path` to a repo-relative path for exactly this reason. Use
   `prediction-model/data/prediction_verification.jsonl`, not `C:\...`.

---

## Mounting the panel

The panel is self-contained and fetches on its own, so mounting it is one line
in an existing client component — e.g. in
`src/components/features/prediction/prediction-dashboard.tsx`:

```tsx
import PredictionProvenancePanel from "./audit/prediction-provenance-panel";

<PredictionProvenancePanel
  stationId={selectedStation.stationPublicId}
  stationName={selectedStation.stationName}
  horizonHours={Number(selectedHorizon.replace("h", ""))}
  onSelectHorizon={(h) => setSelectedHorizon(`${h}h` as PredictionHorizon)}
/>
```

Or, to avoid a client round-trip entirely, have a server component fetch the
payload and pass `initialPayload`, in which case the panel issues no request.

---

## Tests

`__tests__/prediction-provenance-core.test.ts` and
`__tests__/provenance-api.test.ts` cover the derivation logic and the bounded
read path. They use `node:test`; there is no test framework in this repo, so
they are compiled and run directly:

```powershell
npx tsc src/components/features/prediction/audit/__tests__/*.ts `
  src/components/features/prediction/audit/prediction-provenance-core.ts `
  src/components/features/prediction/audit/provenance-api.ts `
  --outDir <tmp> --module commonjs --target es2022 --moduleResolution node `
  --skipLibCheck --esModuleInterop --strict
node --test <tmp>
```
