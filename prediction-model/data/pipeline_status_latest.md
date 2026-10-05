## Pipeline status -- 2026-10-05T05:58:05.655527Z

**OK**

- scored: **244**   pending: 4626   duplicate publishes excluded: 966
- stations verified: **17**   predictions published: 13345

| horizon | scored forecasts | stations | variable entries by producer |
|---|---|---|---|
| +1h | 11078 | 17 | persistence 40302, lln 2940, unknown 1053 |
| +3h | 8399 | 17 | persistence 29682, lln 2824, unknown 1073 |
| +6h | 5024 | 15 | lln 11054, persistence 8874, unknown 167 |
| +12h | 1559 | 15 | persistence 3116, lln 3062, unknown 4 |
| +24h | 312 | 14 | persistence 936, lln 312 |

Counts are forecasts (one per station and horizon); the producer split is counted over variable entries, so it is roughly four times larger and is not comparable to the scored column.

### Recent pipeline events (last 15 min)
- `watchdog_check` at 2026-10-05T05:43:29.795227Z
- `watchdog_check` at 2026-10-05T05:48:29.668520Z
- `watchdog_check` at 2026-10-05T05:53:29.795179Z

### File freshness (minutes since last write)
- `mqtt_live_predictions.json`: 1.5 min
- `observation_audit.jsonl`: 1.5 min
- `prediction_audit.jsonl`: 1.5 min
- `prediction_verification.jsonl`: 5.4 min
