"""
Repeated-seed comparison: does the leaky floor's effect exceed run-to-run noise?

WHY THIS RUN EXISTS
-------------------
The controlled single-seed comparison showed the leaky floor changing MAE by
roughly -8% to +2% on wind and consistently hurting temperature. Before drawing a
conclusion from numbers that small, the same comparison has to be repeated with
several seeds. If the seed-to-seed spread is comparable to the effect size, then
neither floor is distinguishable from the other and no promotion decision can rest
on it.

Both arms train on the SAME current corpus (weather_telemetry_current.csv) and
the SAME test split. The only difference is which wind output floor the model
code applies. Because the floor lives in model.py rather than in the training
script, each relu arm is produced by running this from a temporary checkout of
model.py with the floor reverted -- see run_arm() below. That keeps the arms
honest: identical everything except the one line under test.

Output is one candidate directory per (floor, seed) so no run can overwrite
another, plus a machine-readable summary.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "prediction-model", "data")
MODEL_PY = os.path.join(ROOT, "prediction-model", "src", "model.py")

CURRENT_CSV = os.path.join(DATA, "weather_telemetry_current.csv")

# The leaky floor, and the relu expression it replaced. Reverting is a literal
# string substitution so the two arms differ by exactly one call and nothing else.
LEAKY_MARKER = "pred_ws = _nonnegative(ws_orig + d_ws, max_value=250.0)"
RELU_MARKER = "pred_ws = torch.clamp(F.relu(ws_orig + d_ws), min=0.0, max=250.0)"

HORIZONS = "1,3,6,12,24"


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=30
                              ).stdout.strip()
    except Exception:
        return "unknown"


def arm_model_py(floor: str, work_dir: str) -> str:
    """
    Produce the model.py for one arm.

    "leaky" is the working tree as-is. "relu" restores the pre-fix floor. A copy
    is returned so the training subprocess can import it without mutating the
    repo. Both arms otherwise share every byte of every other source file.
    """
    src = open(MODEL_PY, encoding="utf-8").read()

    if floor == "leaky":
        if LEAKY_MARKER not in src:
            raise SystemExit(
                f"expected the leaky floor at the featured output; not found.\n"
                f"Looked for: {LEAKY_MARKER}")
    else:
        if LEAKY_MARKER not in src:
            raise SystemExit(f"cannot build relu arm; {LEAKY_MARKER!r} not found")
        src = src.replace(LEAKY_MARKER, RELU_MARKER, 1)

    dst_dir = os.path.join(work_dir, f"src_{floor}")
    os.makedirs(dst_dir, exist_ok=True)
    # The training script imports its siblings by module name, so the whole src
    # tree is needed. Symlinking would be fragile on Windows; copy instead.
    for entry in os.listdir(os.path.dirname(MODEL_PY)):
        if entry.endswith(".py") and not entry.startswith("tmp_"):
            shutil.copy2(os.path.join(os.path.dirname(MODEL_PY), entry),
                         os.path.join(dst_dir, entry))
    dst = os.path.join(dst_dir, "model.py")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    return dst


def run_arm(floor: str, seed: int, out_dir: str, epochs: int, patience: int,
            log_path: str, work_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    # The trainer must be launched FROM THE ARM DIRECTORY. `model.py` is imported
    # as a sibling module by absolute filename, so running the repo's own
    # prediction-model/src/train_predictive_quality.py would import the repo's
    # model.py -- the leaky one -- no matter what this script wrote to disk. That
    # mistake produced two "different" arms whose checkpoints scored identically
    # to six decimal places; the sweep measured nothing.
    arm_src = arm_model_py(floor, work_dir)
    trainer = os.path.join(os.path.dirname(arm_src), "train_predictive_quality.py")
    if not os.path.exists(trainer):
        raise SystemExit(f"arm trainer missing: {trainer}")

    cmd = [
        os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
        trainer,
        "--horizons", HORIZONS,
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--seed", str(seed),
        "--weather-csv", CURRENT_CSV,
        "--output-dir", out_dir,
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=14400)
    return {
        "floor": floor, "seed": seed, "output_dir": out_dir,
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - t0, 1),
        "log": log_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--work-dir", default=os.path.join(
        ROOT, "scratch", "seed_sweep"))
    ap.add_argument("--summary", default=os.path.join(
        DATA, "seed_sweep_summary.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.work_dir, exist_ok=True)

    print("=" * 78)
    print("SEED SWEEP: leaky floor vs relu floor, identical corpus, isolated effect")
    print("=" * 78)
    print(f"  corpus   : {os.path.basename(CURRENT_CSV)}")
    print(f"  horizons : {HORIZONS}")
    print(f"  seeds    : {seeds}")
    print(f"  epochs   : {args.epochs} (patience {args.patience})")
    print(f"  commit   : {git_head()[:12]}")

    # Verify both arms are constructible before spending an hour of compute.
    for floor in ("leaky", "relu"):
        p = arm_model_py(floor, args.work_dir)
        text = open(p, encoding="utf-8").read()
        want = LEAKY_MARKER if floor == "leaky" else RELU_MARKER
        other = RELU_MARKER if floor == "leaky" else LEAKY_MARKER
        assert want in text, f"{floor} arm missing its marker"
        assert other not in text, f"{floor} arm still contains the other floor"
        print(f"  arm ok   : {floor}")

    runs = []
    for floor in ("relu", "leaky"):
        for seed in seeds:
            tag = f"{floor}_s{seed}"
            out = os.path.join(DATA, "seed_sweep", tag)
            log = os.path.join(args.work_dir, f"{tag}.log")
            print(f"\n-> {tag}  (output: {os.path.relpath(out, ROOT)})", flush=True)
            res = run_arm(floor, seed, out, args.epochs, args.patience, log,
                          args.work_dir)
            res["tag"] = tag
            # Repo-relative paths only. This file lives under prediction-model/data
            # and is covered by the provenance path-hygiene gate, which rejects
            # absolute machine paths ("C:\...") so artifacts stay portable and
            # cannot leak a developer's directory layout.
            res["output_dir"] = os.path.relpath(out, ROOT).replace("\\", "/")
            res["log"] = os.path.relpath(log, ROOT).replace("\\", "/")
            runs.append(res)
            status = "ok" if res["returncode"] == 0 else f"FAILED rc={res['returncode']}"
            print(f"   {status}  in {res['elapsed_s']:.0f}s  log={os.path.basename(log)}",
                  flush=True)

    summary = {
        "generated_at_epoch": int(time.time()),
        "commit": git_head(),
        "corpus": os.path.relpath(CURRENT_CSV, ROOT).replace("\\", "/"),
        "corpus_sha_note": "recorded in each candidate manifest",
        "horizons": HORIZONS,
        "seeds": seeds,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "runs": runs,
    }
    with open(args.summary, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written: {args.summary}")
    bad = [r["tag"] for r in runs if r["returncode"] != 0]
    if bad:
        print(f"WARNING: failed runs: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
