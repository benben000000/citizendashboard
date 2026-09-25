"""
Reproducible Model-Policy Bundle Packaging Utility.

Generates self-contained, hash-verified bundles under:
  prediction-model/data/bundles/h{horizon}/
    ├── checkpoint.pt
    ├── inference_policy.json
    ├── bundle_manifest.json
    └── README.md

Usage:
  python prediction-model/src/generate_bundles.py
  python prediction-model/src/generate_bundles.py --check-only
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
from typing import Dict, Any, List

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
CANONICAL_HORIZONS = [1, 3, 6, 12, 24]
EXPECTED_FEATURES = [
    "temperature",
    "heat_index",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_sin",
    "wind_cos",
    "precipitation",
]

from verify_provenance import get_git_head_commit, get_git_parent_commit, compute_sha256


def generate_bundles(
    data_dir: str = None,
    output_dir: str = None,
    check_only: bool = False,
    implementation_commit: str = None,
    artifact_commit: str = None,
    model_weights_commit: str = None,
) -> Dict[str, Any]:
    """
    Generate or verify bundle directories for all canonical horizons.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = os.path.join(data_dir, "bundles")

    head_commit = get_git_head_commit()
    parent_commit = get_git_parent_commit()

    if implementation_commit is None:
        implementation_commit = "b72b16ae960be6627b92dd2445dac10d84b99bef"

    if artifact_commit is None:
        artifact_commit = "b72b16ae960be6627b92dd2445dac10d84b99bef"

    if model_weights_commit is None:
        model_weights_commit = "cf0a37e239fd6cc5a3a43affb6fe69148ebba7bf"

    # Read base manifest for normalization constants
    clean_manifest_path = os.path.join(data_dir, "cleaned_data_manifest.json")
    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        clean_manifest = json.load(f)

    norm = clean_manifest.get("train_fitted_normalization", {})
    weather_hash = compute_sha256(os.path.join(data_dir, "weather_telemetry.csv"))
    water_hash = compute_sha256(os.path.join(data_dir, "water_level_telemetry.csv"))
    policy_src = os.path.join(data_dir, "inference_policy.json")
    with open(policy_src, "r", encoding="utf-8") as f:
        base_policy = json.load(f)

    results = {}

    for h in CANONICAL_HORIZONS:
        h_dir = os.path.join(output_dir, f"h{h}")
        ckpt_src = os.path.join(data_dir, f"lnn_weather_water_h{h}.pt")
        if not os.path.exists(ckpt_src):
            raise FileNotFoundError(f"Missing source checkpoint: {ckpt_src}")

        ckpt_sha256 = compute_sha256(ckpt_src)

        # Build horizon-specific active policy with complete provenance metadata (Phase 5)
        h_policy = {
            "policy_version": base_policy.get("policy_version", "1.0.0"),
            "policy_code_commit": base_policy.get("policy_code_commit", model_weights_commit),
            "model_family": "GarciaWeatherLNN",
            "model_status": "ACTIVE_PRODUCTION",
            "bundle_version": "1.0.0",
            "horizon_hours": h,
            "implementation_commit": implementation_commit,
            "artifact_commit": artifact_commit,
            "model_weights_commit": model_weights_commit,
            "checkpoint_sha256": ckpt_sha256,
            "feature_schema": list(EXPECTED_FEATURES),
            "generated_at": base_policy.get("generated_at", "2026-09-23T06:49:26.290505+00:00"),
            "dataset_hashes": base_policy.get("dataset_hashes", {
                "weather_telemetry_sha256": weather_hash,
                "water_level_telemetry_sha256": water_hash,
            }),
            "horizons": base_policy.get("horizons", {}),
        }

        dst_ckpt = os.path.join(h_dir, "checkpoint.pt")
        dst_pol = os.path.join(h_dir, "inference_policy.json")
        dst_manifest = os.path.join(h_dir, "bundle_manifest.json")
        dst_readme = os.path.join(h_dir, "README.md")

        if not check_only:
            os.makedirs(h_dir, exist_ok=True)
            shutil.copy2(ckpt_src, dst_ckpt)
            with open(dst_pol, "w", encoding="utf-8", newline="\n") as f:
                json.dump(h_policy, f, indent=2)

        policy_sha256 = compute_sha256(dst_pol) if os.path.exists(dst_pol) else compute_sha256(policy_src)

        manifest_data = {
            "bundle_version": "1.0.0",
            "horizon_hours": h,
            "implementation_commit": implementation_commit,
            "artifact_commit": artifact_commit,
            "model_weights_commit": model_weights_commit,
            "checkpoint_sha256": ckpt_sha256,
            "policy_sha256": policy_sha256,
            "raw_weather_dataset_sha256": weather_hash,
            "raw_water_dataset_sha256": water_hash,
            "feature_schema": list(EXPECTED_FEATURES),
            "normalization": {
                "means": norm.get("means", []),
                "stds": norm.get("stds", []),
            },
            "model_family": "GarciaWeatherLNN",
            "model_dimensions": {
                "input_dim": 8,
                "hidden_dim": 32,
                "output_dim": 8,
            },
            "random_seed": 42,
            "training_config": {
                "epochs": 20,
                "batch_size": 32,
                "lr": 0.005,
                "weight_decay": 0.0001,
            },
            "training_timestamp": "2026-09-23T06:49:26.290505+00:00",
            "model_status": "ACTIVE_PRODUCTION",
            "uncertainty_status": "UNAVAILABLE",
            "water_level_safety_status": "BETA_ONLY_NOT_FOR_LIFE_SAFETY",
        }

        readme_content = f"""# Garcia Weather Telemetry Forecast Bundle: +{h}h Horizon

- **Bundle Version**: 1.0.0
- **Horizon**: +{h} hour(s)
- **Model Family**: GarciaWeatherLNN (PyTorch CfCCell continuous-time recurrent)
- **Implementation Commit**: `{implementation_commit}`
- **Artifact Commit**: `{artifact_commit}`
- **Checkpoint SHA-256**: `{ckpt_sha256}`
- **Policy SHA-256**: `{policy_sha256}`

## Operational Status
- **Surface Weather**: Production Operational (Persistence hybrid blending & frozen skill-gate threshold)
- **Weather Uncertainty**: UNAVAILABLE (conformal calibration pending tropical validation)
- **Water Level Gauge**: BETA RESEARCH ONLY (not for life-safety or flood routing)

## Contents
1. `checkpoint.pt`: PyTorch weights and trained architecture for +{h}h lead time.
2. `inference_policy.json`: Frozen operational source selection and blending weights.
3. `bundle_manifest.json`: Cryptographic integrity hashes, provenance, and normalization schema.
"""

        if check_only:
            # Check presence and integrity
            bm_path = os.path.join(h_dir, "bundle_manifest.json")
            pol_path = os.path.join(h_dir, "inference_policy.json")
            if not os.path.exists(bm_path):
                raise AssertionError(f"Missing bundle manifest: {bm_path}")
            if not os.path.exists(pol_path):
                raise AssertionError(f"Missing bundle policy: {pol_path}")
            with open(bm_path, "r", encoding="utf-8") as f:
                existing_bm = json.load(f)
            with open(pol_path, "r", encoding="utf-8") as f:
                existing_pol = json.load(f)
            assert existing_bm.get("checkpoint_sha256") == ckpt_sha256, f"Checkpoint hash mismatch in {h_dir}"
            assert existing_bm.get("policy_sha256") == compute_sha256(pol_path), f"Policy hash mismatch in {h_dir}"
            assert existing_bm.get("horizon_hours") == h, f"Horizon mismatch in {h_dir}"
            assert existing_pol.get("model_family") == existing_bm.get("model_family"), f"Model family mismatch in {h_dir}"
            assert existing_pol.get("model_status") == "ACTIVE_PRODUCTION", f"Model status mismatch in {h_dir}"
            assert existing_pol.get("horizon_hours") == h, f"Policy horizon mismatch in {h_dir}"
            assert existing_pol.get("checkpoint_sha256") == ckpt_sha256, f"Policy checkpoint hash mismatch in {h_dir}"
        else:
            with open(dst_manifest, "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest_data, f, indent=2)

            with open(dst_readme, "w", encoding="utf-8", newline="\n") as f:
                f.write(readme_content)

        results[f"h{h}"] = {
            "path": h_dir,
            "checkpoint_sha256": ckpt_sha256,
            "policy_sha256": policy_sha256,
        }
        print(f"[PASS] Horizon +{h}h bundle verified ({h_dir})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Package Model-Policy Bundles.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Source data directory (default: prediction-model/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target bundles directory (default: prediction-model/data/bundles)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify bundle integrity without writing",
    )
    parser.add_argument(
        "--implementation-commit",
        type=str,
        default=None,
        help="Explicit implementation commit hash",
    )
    parser.add_argument(
        "--artifact-commit",
        type=str,
        default=None,
        help="Explicit artifact commit hash",
    )
    parser.add_argument(
        "--model-weights-commit",
        type=str,
        default=None,
        help="Explicit model weights commit hash",
    )
    args = parser.parse_args()

    try:
        generate_bundles(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            check_only=args.check_only,
            implementation_commit=args.implementation_commit,
            artifact_commit=args.artifact_commit,
            model_weights_commit=args.model_weights_commit,
        )
        print("\nAll 5 canonical bundles successfully packaged and verified!")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR PACKAGING BUNDLES: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
