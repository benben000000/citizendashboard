## Pipeline status -- 2026-10-05T02:58:04.965214Z

**OK**

- scored: **82**   pending: 3275   duplicate publishes excluded: 1288
- stations verified: **17**   predictions published: 10780

| horizon | scored forecasts | stations | variable entries by producer |
|---|---|---|---|
| +1h | 10565 | 17 | persistence 38250, lln 2940, unknown 1053 |
| +3h | 7994 | 17 | persistence 28062, lln 2824, unknown 1073 |
| +6h | 5024 | 15 | lln 11054, persistence 8874, unknown 167 |
| +12h | 1559 | 15 | persistence 3116, lln 3062, unknown 4 |
| +24h | 1 | 1 | persistence 3, lln 1 |

Counts are forecasts (one per station and horizon); the producer split is counted over variable entries, so it is roughly four times larger and is not comparable to the scored column.

### Recent pipeline events (last 15 min)
- `watchdog_check` at 2026-10-05T02:43:30.928219Z
- `watchdog_check` at 2026-10-05T02:48:30.584560Z
- `watchdog_check` at 2026-10-05T02:53:30.752945Z

### File freshness (minutes since last write)
- `mqtt_live_predictions.json`: 0.0 min
- `observation_audit.jsonl`: 0.0 min
- `prediction_audit.jsonl`: 0.1 min
- `prediction_verification.jsonl`: 5.4 min
