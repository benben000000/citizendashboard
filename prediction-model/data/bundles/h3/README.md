# Garcia Weather Telemetry Forecast Bundle: +3h Horizon

- **Bundle Version**: 1.0.0
- **Horizon**: +3 hour(s)
- **Model Family**: GarciaWeatherLNN (PyTorch CfCCell continuous-time recurrent)
- **Implementation Commit**: `b72b16ae960be6627b92dd2445dac10d84b99bef`
- **Artifact Commit**: `b72b16ae960be6627b92dd2445dac10d84b99bef`
- **Checkpoint SHA-256**: `bdb7e7422340a6026eaf49df3bd075150bb39860a8a47c5659c85fe847f71dba`
- **Policy SHA-256**: `9d543af1f5bbba7a3f03dadfb5f40ac62b0d0da4261332552857cadf392ac079`

## Operational Status
- **Surface Weather**: Production Operational (Persistence hybrid blending & frozen skill-gate threshold)
- **Weather Uncertainty**: UNAVAILABLE (conformal calibration pending tropical validation)
- **Water Level Gauge**: BETA RESEARCH ONLY (not for life-safety or flood routing)

## Contents
1. `checkpoint.pt`: PyTorch weights and trained architecture for +3h lead time.
2. `inference_policy.json`: Frozen operational source selection and blending weights.
3. `bundle_manifest.json`: Cryptographic integrity hashes, provenance, and normalization schema.
