# Garcia Weather Telemetry Forecast Bundle: +12h Horizon

- **Bundle Version**: 1.0.0
- **Horizon**: +12 hour(s)
- **Model Family**: GarciaWeatherLNN (PyTorch CfCCell continuous-time recurrent)
- **Implementation Commit**: `b38231c0527ef78cf09804ab290af6ba4073d9ff`
- **Artifact Commit**: `b38231c0527ef78cf09804ab290af6ba4073d9ff`
- **Checkpoint SHA-256**: `73697f9d10427e7b64a567a6ffb243dcf969e03a5f9eb5538eb5ee01d80fb4e2`
- **Policy SHA-256**: `0fb5174035eba419723d1ee4b9bf658d5914772a13cb65aa7de96a1e20e6cfb5`

## Operational Status
- **Surface Weather**: Production Operational (Persistence hybrid blending & frozen skill-gate threshold)
- **Weather Uncertainty**: UNAVAILABLE (conformal calibration pending tropical validation)
- **Water Level Gauge**: BETA RESEARCH ONLY (not for life-safety or flood routing)

## Contents
1. `checkpoint.pt`: PyTorch weights and trained architecture for +12h lead time.
2. `inference_policy.json`: Frozen operational source selection and blending weights.
3. `bundle_manifest.json`: Cryptographic integrity hashes, provenance, and normalization schema.
