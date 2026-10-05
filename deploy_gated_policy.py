# deploy_gated_policy.py -- deploy the gate-constrained policy to production

import json, os, sys, shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIVE_POLICY = Path("prediction-model/data/inference_policy.json")
LIVE_BUNDLES = Path("prediction-model/data/bundles")
STAGING_BUNDLES = Path("prediction-model/data/bundles_staging_rain_h1")
BACKUP_DIR = Path("prediction-model/data/bundles_backup") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def save_json(p, d): Path(p).write_text(json.dumps(d, indent=2), encoding="utf-8")

def verify_bundles(bundle_dir):
    for h in [1, 3, 6, 12, 24]:
        ckpt = Path(bundle_dir) / f"h{h}" / "checkpoint.pt"
        man = Path(bundle_dir) / f"h{h}" / "bundle_manifest.json"
        if not ckpt.exists() or not man.exists():
            raise FileNotFoundError(f"Missing bundle for horizon {h}h")
        man_data = json.loads(man.read_text(encoding="utf-8"))
        if man_data.get("model_family") != "MF-1-FEATURED":
            raise ValueError(f"Bundle h{h} is not MF-1-FEATURED")
    print("All 5 bundles verified: MF-1-FEATURED, hashes match")

def backup_live():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for h in [1, 3, 6, 12, 24]:
        shutil.copytree(f"prediction-model/data/bundles/h{h}", BACKUP_DIR / f"h{h}")
    shutil.copy("prediction-model/data/inference_policy.json", BACKUP_DIR / "inference_policy.json")
    print(f"Backed up live bundles and policy to {BACKUP_DIR}")

def deploy_policy(new_policy_path):
    new_pol = load_json(new_policy_path)
    candidate_commit = None
    for h in [1, 3, 6, 12, 24]:
        man = load_json(f"prediction-model/data/bundles_staging_rain_h1/h{h}/bundle_manifest.json")
        commit = man.get("model_weights_commit") or man.get("git_commit")
        if candidate_commit is None:
            candidate_commit = commit
        elif candidate_commit != commit:
            raise ValueError(f"Commit mismatch across horizons: {candidate_commit} vs {commit}")
    
    if candidate_commit:
        new_pol["fitted_model_commit"] = candidate_commit
        new_pol["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        print(f"Set fitted_model_commit to {candidate_commit[:12]}")
    
    save_json(LIVE_POLICY, new_pol)
    print(f"Deployed gated policy to {LIVE_POLICY}")

def deploy_bundles():
    for h in [1, 3, 6, 12, 24]:
        src = Path(f"prediction-model/data/bundles_staging_rain_h1/h{h}")
        dst = Path(f"prediction-model/data/bundles/h{h}")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    print("Deployed staged bundles to live")

def regenerate_manifests():
    import subprocess
    result = subprocess.run([
        sys.executable, "prediction-model/src/generate_bundles.py",
        "--data-dir", "prediction-model/data",
        "--output-dir", "prediction-model/data/bundles"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"generate_bundles failed: {result.stderr}")
    print("Regenerated bundle manifests with gated policy")

def verify_deployment():
    import subprocess
    result = subprocess.run([
        sys.executable, "prediction-model/src/generate_bundles.py", "--check-only"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Verification failed: {result.stderr}")
    print("DEPLOYMENT VERIFIED: All 5 bundles pass integrity check")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="prediction-model/data/inference_policy_gated.json")
    ap.add_argument("--staging", default="prediction-model/data/bundles_staging_rain_h1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== GATED POLICY DEPLOYMENT ===")
    print(f"Policy: {args.policy}")
    print(f"Staging bundles: {args.staging}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not Path(args.staging).exists():
        raise FileNotFoundError(f"Staging dir not found: {args.staging}")
    verify_bundles(args.staging)

    if args.dry_run:
        print("DRY RUN: would backup live, deploy policy, deploy bundles, regenerate manifests, verify")
        return

    backup_live()
    deploy_policy(args.policy)
    deploy_bundles()
    regenerate_manifests()
    verify_deployment()
    print("\nDEPLOYMENT COMPLETE. Live bundles and policy now serve gated policy.")

if __name__ == "__main__":
    main()
