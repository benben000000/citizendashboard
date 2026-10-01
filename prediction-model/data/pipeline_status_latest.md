## Pipeline status -- 2026-10-01T02:28:04.032736Z

**OK**

- scored: **2591**   pending: 0   duplicate publishes excluded: 7916
- stations verified: **15**   predictions published: 7916

| horizon | scored forecasts | stations | variable entries by producer |
|---|---|---|---|
| +1h | 1140 | 15 | lln 2940, persistence 840, unknown 780 |
| +3h | 906 | 15 | lln 2824, unknown 800 |
| +6h | 545 | 15 | lln 2180 |
| +12h | 0 | 0 | no matured records yet |
| +24h | 0 | 0 | no matured records yet |

Counts are forecasts (one per station and horizon); the producer split is counted over variable entries, so it is roughly four times larger and is not comparable to the scored column.

### Recent pipeline events (last 15 min)
- `watchdog_check` at 2026-10-01T02:13:30.213983Z
- `watchdog_check` at 2026-10-01T02:18:30.218088Z
- `watchdog_check` at 2026-10-01T02:23:30.635992Z
- `watchdog_check` at 2026-10-01T02:26:20.774132Z

### File freshness (minutes since last write)
- `mqtt_live_predictions.json`: 0.0 min
- `observation_audit.jsonl`: 0.0 min
- `prediction_audit.jsonl`: 0.1 min
- `prediction_verification.jsonl`: 0.0 min
