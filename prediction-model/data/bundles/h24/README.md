# Garcia Weather Telemetry Forecast Bundle: +24h Horizon

- **Bundle Version**: 1.0.0
- **Horizon**: +24 hour(s)
- **Model Family**: GarciaWeatherLNN (PyTorch CfCCell continuous-time recurrent)
- **Implementation Commit**: `b72b16ae960be6627b92dd2445dac10d84b99bef`
- **Artifact Commit**: `b72b16ae960be6627b92dd2445dac10d84b99bef`
- **Checkpoint SHA-256**: `632b5336c08da1a9c57e62af67003919f2af5512c345fd9ddecd8f7f0250879c`
- **Policy SHA-256**: `f3aac1d7a000e72fcd12cf0a88f185933c8ddc08416edf5959bab518796faf44`

## Operational Status
- **Surface Weather**: Production Operational (Persistence hybrid blending & frozen skill-gate threshold)
- **Weather Uncertainty**: UNAVAILABLE (conformal calibration pending tropical validation)
- **Water Level Gauge**: BETA RESEARCH ONLY (not for life-safety or flood routing)

## Contents
1. `checkpoint.pt`: PyTorch weights and trained architecture for +24h lead time.
2. `inference_policy.json`: Frozen operational source selection and blending weights.
3. `bundle_manifest.json`: Cryptographic integrity hashes, provenance, and normalization schema.
