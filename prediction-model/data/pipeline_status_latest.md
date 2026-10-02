## Pipeline status -- 2026-10-01T23:58:04.487341Z

**OK**

- scored: **80**   pending: 896   duplicate publishes excluded: 440
- stations verified: **17**   predictions published: 1704

| horizon | scored forecasts | stations | variable entries by producer |
|---|---|---|---|
| +1h | 3036 | 17 | persistence 8424, lln 2940, unknown 780 |
| +3h | 2332 | 15 | persistence 5704, lln 2824, unknown 800 |
| +6h | 1476 | 15 | lln 4042, persistence 1862 |
| +12h | 171 | 15 | lln 342, persistence 342 |
| +24h | 0 | 0 | no matured records yet |

Counts are forecasts (one per station and horizon); the producer split is counted over variable entries, so it is roughly four times larger and is not comparable to the scored column.

### Recent pipeline events (last 15 min)
- `watchdog_check` at 2026-10-01T23:43:29.925577Z
- `sequence_reset` KT-6CBD47DC5194 at 2026-10-01T23:44:07.383561Z
- `sequence_reset` KT-8CEE47DC5194 at 2026-10-01T23:48:24.958027Z
- `watchdog_check` at 2026-10-01T23:48:29.987672Z
- `watchdog_check` at 2026-10-01T23:53:29.937767Z
- `sequence_reset` KT-6CBD47DC5194 at 2026-10-01T23:54:07.493076Z
- `sequence_reset` KT-8CEE47DC5194 at 2026-10-01T23:56:24.915414Z

### File freshness (minutes since last write)
- `mqtt_live_predictions.json`: 0.0 min
- `observation_audit.jsonl`: 0.0 min
- `prediction_audit.jsonl`: 0.4 min
- `prediction_verification.jsonl`: 5.4 min
