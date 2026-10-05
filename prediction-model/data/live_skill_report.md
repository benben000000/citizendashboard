# Live verified skill against persistence

Generated 2026-10-02T00:23:29.978391Z

Measured on forecasts the production system actually published, scored
against later station observations. No held-out split, no simulation: the
forecast existed before the observation did.

Persistence is the station's own reading at the forecast's origin time,
within 30 minutes. Rows without that
reading are dropped rather than defaulted.

| horizon | variable | n | served MAE | persistence MAE | skill | verdict |
|---|---|---|---|---|---|---|
| +1h | temperature (degC) | 3033 | 0.6251 | 0.6251 | -0.0% | ties persistence |
| +1h | humidity (%RH) | 3033 | 2.1853 | 2.1854 | +0.0% | ties persistence |
| +1h | pressure (hPa) | 3036 | 0.3777 | 0.3777 | +0.0% | ties persistence |
| +1h | wind_speed (m/s) | 3036 | 0.7036 | 0.7040 | +0.1% | ties persistence |
| +3h | temperature (degC) | 2358 | 1.7553 | 1.7553 | +0.0% | ties persistence |
| +3h | humidity (%RH) | 2358 | 5.4004 | 5.4000 | -0.0% | ties persistence |
| +3h | pressure (hPa) | 2358 | 1.0061 | 1.0061 | -0.0% | ties persistence |
| +3h | wind_speed (m/s) | 2358 | 0.9176 | 0.9176 | +0.0% | ties persistence |
| +6h | temperature (degC) | 1476 | 2.9922 | 3.3555 | +10.8% | beats persistence |
| +6h | humidity (%RH) | 1476 | 10.5999 | 10.6001 | +0.0% | ties persistence |
| +6h | pressure (hPa) | 1476 | 1.6951 | 1.6951 | -0.0% | ties persistence |
| +6h | wind_speed (m/s) | 1476 | 0.9632 | 1.1636 | +17.2% | beats persistence |
| +12h | temperature (degC) | 171 | 2.1863 | 6.1017 | +64.2% | beats persistence |
| +12h | humidity (%RH) | 171 | 20.0600 | 20.0579 | -0.0% | ties persistence |
| +12h | pressure (hPa) | 171 | 1.8961 | 1.8960 | -0.0% | ties persistence |
| +12h | wind_speed (m/s) | 171 | 1.4689 | 2.5823 | +43.1% | beats persistence |
| +24h | temperature | 0 | - | - | - | no rows |
| +24h | humidity | 0 | - | - | - | no rows |
| +24h | pressure | 0 | - | - | - | no rows |
| +24h | wind_speed | 0 | - | - | - | no rows |

## Summary

- cells with computable evidence (n>=30): **16 of 20**
- beats persistence by >=1%: **4**
- ties persistence (within 1%): **12**
- worse than persistence: **0**
- insufficient rows: 0

