# Garcia Weather Telemetry Forecast Bundle: +12h Horizon

- **Bundle Version**: 1.0.0
- **Horizon**: +12 hour(s)
- **Model Family**: GarciaWeatherLNN (PyTorch CfCCell continuous-time recurrent)
- **Implementation Commit**: `b72b16ae960be6627b92dd2445dac10d84b99bef`
- **Artifact Commit**: `b72b16ae960be6627b92dd2445dac10d84b99bef`
- **Checkpoint SHA-256**: `73697f9d10427e7b64a567a6ffb243dcf969e03a5f9eb5538eb5ee01d80fb4e2`
- **Policy SHA-256**: `f3aac1d7a000e72fcd12cf0a88f185933c8ddc08416edf5959bab518796faf44`

## Operational Status
- **Surface Weather**: Production Operational (Persistence hybrid blending & frozen skill-gate threshold)
- **Weather Uncertainty**: UNAVAILABLE (conformal calibration pending tropical validation)
- **Water Level Gauge**: BETA RESEARCH ONLY (not for life-safety or flood routing)

## Contents
1. `checkpoint.pt`: PyTorch weights and trained architecture for +12h lead time.
2. `inference_policy.json`: Frozen operational source selection and blending weights.
3. `bundle_manifest.json`: Cryptographic integrity hashes, provenance, and normalization schema.
