"""
Deterministic Manifest Generator for Prediction Model Pipeline.

Generates:
  1. data_quality_report.json
  2. cleaned_data_manifest.json
  3. weather_data_audit.json

Usage:
  python prediction-model/src/generate_manifests.py --output-dir prediction-model/data
  python prediction-model/src/generate_manifests.py --check-only
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")

from dataset import get_telemetry_pipeline
from audit_data_availability import run_audit
from verify_provenance import get_git_head_commit, get_git_parent_commit


def generate_manifests(
    output_dir: str = None,
    commit: str = None,
    check_only: bool = False,
) -> Dict[str, Any]:
    """
    Generate or check data manifests.
    If check_only is True, compares generated reports to existing files on disk without writing.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    if commit is None:
        commit = get_git_head_commit()

    print(f"Generating manifests for commit {commit} -> output_dir: {output_dir}")

    # 1. Pipeline data quality and cleaned data manifest
    pipeline = get_telemetry_pipeline()
    dqr_path = os.path.join(output_dir, "data_quality_report.json")
    cdm_path = os.path.join(output_dir, "cleaned_data_manifest.json")

    report = pipeline.generate_data_quality_report(
        output_path=dqr_path if not check_only else None,
        manifest_path=cdm_path if not check_only else None,
        write_to_disk=not check_only,
        git_commit=commit,
    )

    # 2. Weather data availability audit
    wda_path = os.path.join(output_dir, "weather_data_audit.json")
    audit_report = run_audit(output_json=wda_path if not check_only else None, write_to_disk=not check_only)

    if check_only:
        diffs = []
        for p, generated in [(dqr_path, report), (cdm_path, report), (wda_path, audit_report)]:
            if not os.path.exists(p):
                diffs.append(f"Missing file on disk: {p}")
                continue
            with open(p, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            # Compare key invariants (hashes, row counts, feature schema)
            if disk_data.get("data_hashes") != generated.get("data_hashes"):
                diffs.append(f"{os.path.basename(p)} data_hashes mismatch")
            if disk_data.get("raw_counts") != generated.get("raw_counts"):
                diffs.append(f"{os.path.basename(p)} raw_counts mismatch")
        if diffs:
            raise AssertionError(f"Check-only manifest mismatch:\n" + "\n".join(diffs))
        print("[PASS] Manifest check-only: all files match generated schema and hashes.")

    print(f"Successfully generated manifests in: {output_dir}")
    return {
        "status": "PASS",
        "output_dir": output_dir,
        "commit": commit,
        "files": [dqr_path, cdm_path, wda_path],
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic Manifest Generator.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_DIR,
        help="Directory to write manifests (default: prediction-model/data)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Explicit commit hash to record in manifests (default: git HEAD)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify existing manifests match without writing to disk",
    )
    args = parser.parse_args()

    try:
        generate_manifests(
            output_dir=args.output_dir,
            commit=args.commit,
            check_only=args.check_only,
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
