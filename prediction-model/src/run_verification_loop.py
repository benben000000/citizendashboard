"""
Run the prediction verifier on a recurring interval.

WHY THIS EXISTS
---------------
The verification layer worked the whole time. Nothing ran it.

`prediction_verification_summary.json` sat at scored=0 / pending=81 while
`observation_audit.jsonl` accumulated 19,413 records -- because `verify_predictions.py`
was never invoked by anything. The ingestor writes predictions and observations;
nothing reconciled them. So the project had no production accuracy figure and,
worse, the summary file said so in a way that read like "nothing has matured yet"
rather than "nothing has looked".

The first manual run scored 812 predictions across 15 stations:

    horizon   n   temperature   humidity   pressure   wind_speed
      1h    615        0.2510      1.2746     0.2096       0.6032
      3h    197        0.6494      2.2976     0.2525       0.8082

Longer horizons fill in as predictions age past their target time, which is why
this runs on a schedule rather than once: at 30 minutes it costs nothing and it
is the only mechanism that grows the longer-horizon sample.

WHAT IT DOES NOT DO
-------------------
It never mutates the audit trail, the policy, or any checkpoint. It reads
predictions, matches them to later observations within the tolerance, and writes
a verification JSONL plus a summary. It is safe to run concurrently with the
ingestor.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
VERIFIER = os.path.join(HERE, "verify_predictions.py")


def run_once():
    started = time.time()
    proc = subprocess.run([PY, VERIFIER], capture_output=True, text=True,
                          cwd=ROOT, timeout=1800)
    ok = proc.returncode == 0
    tail = (proc.stdout or "").strip().splitlines()
    summary = next((ln for ln in reversed(tail) if "scored" in ln.lower()), "")
    print(f"[{datetime.now(timezone.utc):%H:%M:%SZ}] rc={proc.returncode} "
          f"{time.time()-started:.1f}s  {summary[:120]}")
    if not ok:
        print("  stderr:", (proc.stderr or "").strip().splitlines()[-3:])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-minutes", type=int, default=30,
                    help="How often to reconcile predictions against "
                         "observations. 30 minutes keeps longer horizons growing "
                         "without meaningful CPU cost.")
    ap.add_argument("--once", action="store_true",
                    help="Run a single pass and exit.")
    args = ap.parse_args()

    if args.once:
        return 0 if run_once() else 1

    period = max(60, args.interval_minutes * 60)
    print(f"verification loop: every {args.interval_minutes} min "
          f"(Ctrl-C to stop)", flush=True)
    run_once()
    while True:
        try:
            time.sleep(period)
        except KeyboardInterrupt:
            print("\nstopped")
            return 0
        try:
            run_once()
        except Exception as exc:  # a loop that dies silently is the bug we had
            print(f"  loop error (continuing): {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())