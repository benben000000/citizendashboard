## Pipeline status -- 2026-10-01T02:58:04.946622Z

**OK**

- scored: **2591**   pending: 0   duplicate publishes excluded: 8341
- stations verified: **15**   predictions published: 8441

| horizon | scored forecasts | stations | variable entries by producer |
|---|---|---|---|
| +1h | 1140 | 15 | lln 2940, persistence 840, unknown 780 |
| +3h | 906 | 15 | lln 2824, unknown 800 |
| +6h | 545 | 15 | lln 2180 |
| +12h | 0 | 0 | no matured records yet |
| +24h | 0 | 0 | no matured records yet |

Counts are forecasts (one per station and horizon); the producer split is counted over variable entries, so it is roughly four times larger and is not comparable to the scored column.

### Recent pipeline events (last 15 min)
- `watchdog_check` at 2026-10-01T02:43:30.507867Z
- `watchdog_check` at 2026-10-01T02:48:30.520671Z
- `sequence_reset` KT-8CEE47DC5194 at 2026-10-01T02:52:18.602845Z
- `watchdog_check` at 2026-10-01T02:53:30.753037Z
- `watchdog_check` at 2026-10-01T02:57:52.232741Z

### File freshness (minutes since last write)
- `mqtt_live_predictions.json`: 0.0 min
- `observation_audit.jsonl`: 0.0 min
- `prediction_audit.jsonl`: 0.0 min
- `prediction_verification.jsonl`: 30.1 min
