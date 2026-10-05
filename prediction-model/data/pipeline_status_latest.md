## Pipeline status -- 2026-10-05T11:29:27.107597Z

**ATTENTION** -- live cache stale: 155 min old; observations stopped: 155 min old; predictions stopped: 156 min old

- scored: **48**   pending: 1414   duplicate publishes excluded: 731
- stations verified: **17**   predictions published: 2593

| horizon | scored forecasts | stations | variable entries by producer |
|---|---|---|---|
| +1h | 11317 | 17 | persistence 40350, lln 3848, unknown 1053 |
| +3h | 8401 | 17 | persistence 29690, lln 2824, unknown 1073 |
| +6h | 5024 | 15 | lln 11054, persistence 8874, unknown 167 |
| +12h | 1559 | 15 | persistence 3116, lln 3062, unknown 4 |
| +24h | 312 | 14 | persistence 936, lln 312 |

Counts are forecasts (one per station and horizon); the producer split is counted over variable entries, so it is roughly four times larger and is not comparable to the scored column.

### Recent pipeline events (last 15 min)
- `upstream_silence` at 2026-10-05T11:29:26.219195Z
- `mqtt_error` at 2026-10-05T11:29:26.314751Z
- `watchdog_restart` at 2026-10-05T11:29:29.905998Z

### File freshness (minutes since last write)
- `mqtt_live_predictions.json`: 155.2 min
- `observation_audit.jsonl`: 155.2 min
- `prediction_audit.jsonl`: 156.0 min
- `prediction_verification.jsonl`: 156.8 min
