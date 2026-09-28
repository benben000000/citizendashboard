# Multi-Source Implementation Status

## Baseline Reference

| Field | Value |
|---|---|
| Baseline commit | `9b58618900940d43ef65b093ca9527e2e6ff4d43` |
| Source mode | Local-only |
| External sources in production | None |
| Test suite count | 85 |
| Inference policy version | 1.0.0 |

## Current Operational Routing

| Target | h1 | h3 | h6 | h12 | h24 |
|---|---|---|---|---|---|
| Temperature | persistence | persistence | **learned** | **learned** | persistence |
| Humidity | persistence | persistence | persistence | persistence | persistence |
| Pressure | persistence | **learned** | persistence | persistence | persistence |
| Wind speed | persistence | persistence | persistence | **learned** | persistence |
| Wind direction | persistence | persistence | persistence | persistence | persistence |
| Heat index | derived | derived | derived | derived | derived |
| Rain | blend 0.8/0.2 | blend 0.6/0.4 | blend 0.45/0.55 | blend 0.45/0.55 | blend 0.4/0.6 |
| UV | quarantined | quarantined | quarantined | quarantined | quarantined |
| Luminosity | limited/beta | limited/beta | limited/beta | limited/beta | limited/beta |

## Information-Limited Targets

- **Wind direction**: Persistence fallback at all horizons.
- **Humidity**: Persistence fallback at all horizons.
- **UV index**: Quarantined pending sensor calibration audit.
- **Luminosity**: Limited/beta pending daylight validation.

## External Source Status

| Source | Registry Status | Decision | Notes |
|---|---|---|---|
| Nearby stations (project-owned) | Registered | `UNKNOWN_BLOCKED` | Awaiting license review |
| Radar/QPE | Registered | `UNKNOWN_BLOCKED` | Product-specific rights unconfirmed |
| Himawari-9 derived | Registered | `UNKNOWN_BLOCKED` | Product distribution rights unconfirmed |
| PAGASA NWP | Registered | `UNKNOWN_BLOCKED` | Usage rights unconfirmed |
| RainViewer | Registered | `UNKNOWN_BLOCKED` | Not automatically acceptable for commercial use |

## Phase Progress

- [x] Phase 0 — Baseline freeze and documentation
- [x] Phase 1 — Source registry and license gate
- [x] Phase 2 — Source manifests and cache contracts
- [x] Phase 3 — Source audit and license decisions
- [ ] Phase 4 — Nearby station integration (BLOCKED: no approved source)
- [ ] Phase 5 — Radar/QPE integration (BLOCKED: no approved source)
- [ ] Phase 6 — Himawari-9 integration (BLOCKED: no approved source)
- [ ] Phase 7 — NWP integration (BLOCKED: no approved source)
- [x] Phase 8 — Causal feature cube (source-aware extension)
- [ ] Phase 9 — Target-specific refinement (partial: vector wind, regime temp exist)
- [ ] Phase 10 — Evaluation and scorecard regeneration
- [x] Phase 11 — Provenance, release, and rollback
- [x] CI/Test additions — Release gates added

## Change Log

| Date | Phase | Change |
|---|---|---|
| 2026-09-27 | 0 | Baseline freeze recorded |
| 2026-09-27 | 1 | Source registry schema and license gate implemented |
| 2026-09-27 | 2 | Source manifests and cache contracts implemented |
| 2026-09-27 | 3 | All candidate sources registered as UNKNOWN_BLOCKED |
| 2026-09-27 | 8,11,CI | Provenance extended, tests added |
