# Garcia Weather Telemetry Forecast Bundle: +24h Horizon

- **Bundle Version**: 1.0.0
- **Horizon**: +24 hour(s)
- **Model Family**: GarciaWeatherLNN (PyTorch CfCCell continuous-time recurrent)
- **Implementation Commit**: `b38231c0527ef78cf09804ab290af6ba4073d9ff`
- **Artifact Commit**: `b38231c0527ef78cf09804ab290af6ba4073d9ff`
- **Checkpoint SHA-256**: `632b5336c08da1a9c57e62af67003919f2af5512c345fd9ddecd8f7f0250879c`
- **Policy SHA-256**: `0fb5174035eba419723d1ee4b9bf658d5914772a13cb65aa7de96a1e20e6cfb5`

## Operational Status
- **Surface Weather**: Production Operational (Persistence hybrid blending & frozen skill-gate threshold)
- **Weather Uncertainty**: UNAVAILABLE (conformal calibration pending tropical validation)
- **Water Level Gauge**: BETA RESEARCH ONLY (not for life-safety or flood routing)

## Contents
1. `checkpoint.pt`: PyTorch weights and trained architecture for +24h lead time.
2. `inference_policy.json`: Frozen operational source selection and blending weights.
3. `bundle_manifest.json`: Cryptographic integrity hashes, provenance, and normalization schema.
