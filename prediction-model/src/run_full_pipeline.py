"""
Full Prediction-Model Pipeline Orchestrator.

WHY THIS EXISTS
---------------
Getting a change from "idea" to "measured" currently requires a person to know
and correctly sequence eight steps by hand:

    1. fetch_current_telemetry.py    refresh the corpus (16 requests, 1/station)
    2. train_predictive_quality.py   train the candidate for 5 horizons
    3. benchmark_vs_nwp.py           score the candidate against real NWP
    4. capture_release_baseline.py   record/compare the incumbent
    5. generate_bundles.py           package model+policy into data/bundles/h*
    6. verify_provenance.py          prove the packaged artifacts are the ones claimed
    7. release_gate.py               the promotion gate
    8. pytest                        the test suite

Every one of those has been got wrong at least once this cycle, in two specific
ways that a plain shell script cannot catch:

  * A step FAILED SILENTLY. ``benchmark_vs_nwp.py`` exited 0 while printing
    ``n/a`` for all seven NWP models and still emitted a rank. ``train_predictive_
    quality.py`` exited rc=1 at the very end (promotion audit bug) while leaving
    five valid checkpoints on disk. Checking only the exit code misses the first;
    checking only "did the files appear" misses the second. This orchestrator
    checks BOTH for every stage and, when they disagree, says so explicitly
    rather than picking whichever answer is more convenient.

  * A step was SKIPPED. Provenance verification has to pass before anything is
    packaged; the pipeline enforces that as a preflight inside the bundle stage,
    so it holds even when ``--stage bundle`` is run on its own. The release gate
    is the other one: it needs a promotion-evidence bundle this pipeline refuses
    to fabricate, so without ``--gate-inputs`` the stage is SKIPPED loudly and
    the run is reported as incomplete rather than green.

It also never deletes a data file, and it records a SHA-256 for every artifact it
consumes or produces, so a run is reproducible and diffable after the fact.

Usage
-----
    python prediction-model/src/run_full_pipeline.py --stage all
    python prediction-model/src/run_full_pipeline.py --stage benchmark --dry-run
    python prediction-model/src/run_full_pipeline.py --stage bundle --force
    python prediction-model/src/run_full_pipeline.py --stage gate \\
        --gate-inputs prediction-model/data/gate_evidence.json

TWO THINGS TO KNOW BEFORE YOU RELY ON THIS
------------------------------------------
1. STEP 5 packages the wrong weights for a promotion workflow.
   ``generate_bundles.py`` packages ``data/lnn_weather_water_h{h}.pt`` -- the
   PRODUCTION checkpoints -- not the ``candidate_h{h}h.pt`` files this pipeline
   just trained and benchmarked. Training a candidate and running the bundle stage
   therefore does NOT ship that candidate. The pipeline reports this as a WARNING
   on every bundle run instead of quietly implying a promotion it did not perform.

2. STEP 7 cannot be auto-fed by this pipeline.
   ``release_gate.py`` takes ``--inputs <evidence.json>`` carrying per-cell,
   per-row, paired VALIDATION-split statistics, and it structurally refuses to
   decide on test data. ``benchmark_vs_nwp.py`` emits aggregate TEST-split
   metrics, which cannot be relabelled into that bundle honestly. So the gate
   runs when you hand it evidence, and is skipped with a loud banner when you
   do not. Do not "fix" that by relabelling the test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PIPELINE_VERSION = "1.0.0"
MANIFEST_VERSION = "1.0.0"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)
DEFAULT_DATA_DIR = os.path.join(MODEL_DIR, "data")
DEFAULT_LOG_ROOT = os.path.join(MODEL_DIR, "logs", "pipeline")

CANONICAL_HORIZONS = [1, 3, 6, 12, 24]

# The scripts the pipeline drives. Kept as one table so a rename is a one-line
# change rather than a hunt through the stage bodies.
SCRIPTS = {
    "fetch": "fetch_current_telemetry.py",
    "train": "train_predictive_quality.py",
    "benchmark": "benchmark_vs_nwp.py",
    "baseline": "capture_release_baseline.py",
    "bundle": "generate_bundles.py",
    "provenance": "verify_provenance.py",
    "gate": "release_gate.py",
}

# benchmark_vs_nwp.py --help, verbatim: the NWP display names it can emit.
NWP_DISPLAY_NAMES = [
    "ECMWF IFS",
    "NOAA GFS",
    "DWD ICON",
    "Meteo-France",
    "JMA",
    "UK Met Office",
    "ECCC GEM",
]

# The four scored channels benchmark_vs_nwp.py reports per horizon.
BENCHMARK_VARIABLES = ["temperature", "humidity", "pressure", "wind_speed"]

BUNDLE_FILES = [
    "checkpoint.pt",
    "inference_policy.json",
    "bundle_manifest.json",
    "README.md",
]

STAGE_ORDER = [
    "fetch",
    "train",
    "benchmark",
    "baseline",
    "bundle",
    "verify",
    "gate",
    "test",
]

STAGE_DESCRIPTIONS = {
    "fetch": "Refresh the telemetry corpus (one request per station).",
    "train": "Train the candidate model for all 5 canonical horizons.",
    "benchmark": "Score the candidate against real NWP on identical rows.",
    "baseline": "Record/compare the incumbent production baseline.",
    "bundle": "Package model+policy bundles, gated on provenance.",
    "verify": "Full provenance verification + the provenance test suite.",
    "gate": "Release gate (skipped loudly if release_gate.py is absent).",
    "test": "The test suite.",
}

SEV_ERROR = "error"
SEV_WARN = "warning"
SEV_INFO = "info"

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"        # the stage ran and declined to run (no gate impl)
STATUS_DRY_RUN = "dry_run"
STATUS_BLOCKED = "blocked"        # refused by a safety gate before doing the work
STATUS_HALTED = "halted"          # never ran, because an earlier stage failed


# --------------------------------------------------------------------------- #
# Result / config primitives
# --------------------------------------------------------------------------- #

@dataclass
class CommandResult:
    """One subprocess invocation and everything the manifest needs about it."""

    argv: List[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    log_path: Optional[str] = None
    timed_out: bool = False
    launched: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "argv": list(self.argv),
            "command_line": " ".join(_quote(a) for a in self.argv),
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 3),
            "log_path": self.log_path,
            "timed_out": self.timed_out,
            "launched": self.launched,
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
        }


@dataclass
class Check:
    """A single substantive verification, independent of any exit code."""

    name: str
    ok: bool
    detail: str
    severity: str = SEV_ERROR

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class StageResult:
    name: str
    status: str
    duration_seconds: float = 0.0
    commands: List[CommandResult] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)
    produced: Dict[str, Any] = field(default_factory=dict)
    consumed: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def blocking_failures(self) -> List[str]:
        return [c.detail for c in self.checks if not c.ok and c.severity == SEV_ERROR]

    @property
    def warnings(self) -> List[str]:
        return [c.detail for c in self.checks if not c.ok and c.severity == SEV_WARN]

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": STAGE_DESCRIPTIONS.get(self.name, ""),
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
            "commands": [c.to_json() for c in self.commands],
            "checks": [c.to_json() for c in self.checks],
            "failures": list(self.failures),
            "notes": list(self.notes),
            "artifacts_produced": self.produced,
            "artifacts_consumed": self.consumed,
        }


@dataclass
class PipelineConfig:
    """Every path and knob the stages read. Built from argv by ``parse_args``."""

    data_dir: str
    run_dir: str
    manifest_path: str = ""
    stage: str = "all"
    python: str = field(default_factory=lambda: sys.executable)
    force: bool = False
    dry_run: bool = False
    require_gate: bool = False

    # Step 1 - fetch
    corpus: str = ""
    fetch_start: str = "2026-06-20"
    fetch_end: str = ""
    fetch_take: int = 5000
    fetch_delay: float = 3.5
    fetch_stations: Optional[str] = None
    fetch_filter_outliers: bool = False
    min_fetch_stations: int = 8

    # Step 2 - train
    candidate_dir: str = ""
    horizons: str = "1,3,6,12,24"
    epochs: int = 60
    patience: int = 12
    lr: float = 1e-3
    seed: Optional[int] = None
    train_commit: Optional[str] = None

    # Step 3 - benchmark
    benchmark_out: str = ""
    benchmark_label: str = "LNN (this project)"
    min_nwp_scored: int = 1
    require_full_nwp: bool = False

    # Step 4 - baseline
    baseline_out: str = ""
    bundles_dir: str = ""

    # Step 6 - provenance
    allow_commit: Optional[str] = None
    provenance_test: str = "test_provenance.py"

    # Step 7 - gate
    gate_inputs: str = ""
    gate_verdict_out: str = ""
    gate_args: List[str] = field(default_factory=list)

    # Step 8 - tests
    pytest_target: str = ""
    pytest_args: List[str] = field(default_factory=lambda: ["-q"])

    # Behaviour
    timeout: Optional[float] = None
    keep_going: bool = False

    def horizon_list(self) -> List[int]:
        return [int(x) for x in self.horizons.split(",") if x.strip()]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in str(arg) or not str(arg)) else str(arg)


def _tail(text: str, limit: int = 4000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return "...[truncated %d chars]...\n" % (len(text) - limit) + text[-limit:]


def _wrap(text: str, width: int = 74) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(path: str) -> Optional[Dict[str, Any]]:
    """Fingerprint a file or a whole directory. Never raises."""
    try:
        if os.path.isfile(path):
            return {
                "type": "file",
                "sha256": sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }
        if os.path.isdir(path):
            files = {}
            for root, _dirs, names in os.walk(path):
                for n in sorted(names):
                    fp = os.path.join(root, n)
                    try:
                        files[os.path.relpath(fp, path).replace("\\", "/")] = {
                            "sha256": sha256_file(fp),
                            "size_bytes": os.path.getsize(fp),
                        }
                    except OSError:
                        continue
            return {"type": "dir", "file_count": len(files), "files": files}
    except OSError:
        return None
    return None


_PYTEST_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warnings?)\b"
)


def parse_pytest_summary(stdout: str) -> Tuple[str, Dict[str, int], List[str]]:
    """Pull the trailing pytest summary line apart.

    pytest prints e.g. ``12 passed, 1 skipped in 4.2s`` or
    ``=== 1 failed, 11 passed in 4.2s ===``. A collection error prints
    ``!!!! Interrupted: 2 errors during collection !!!!`` with no counts. We
    treat "exited 0 but reported failures" as a failure, same as a non-zero exit.
    """
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    summary = ""
    for ln in reversed(lines):
        if re.search(r"\bin\s+[\d.]+s\b", ln) or "!!!" in ln or re.search(
            r"\b(no tests ran)\b", ln
        ):
            summary = ln
            break
    if not summary:
        summary = lines[-1] if lines else ""

    counts: Dict[str, int] = {}
    for m in _PYTEST_COUNT_RE.finditer(summary):
        key = m.group(2)
        if key.startswith("warning"):
            key = "warning"
        if key.startswith("error"):
            key = "error"
        counts[key] = counts.get(key, 0) + int(m.group(1))

    problems: List[str] = []
    if "failed" in counts and counts["failed"]:
        problems.append(f"{counts['failed']} test(s) failed")
    if "error" in counts and counts["error"]:
        problems.append(f"{counts['error']} test error(s)")
    if not counts:
        problems.append(f"no pytest result counts in summary line: {summary!r}")
    if "no tests ran" in summary.lower():
        problems.append("pytest collected no tests")
    return summary, counts, problems


def bundle_dir_name(horizon: int) -> str:
    return f"h{horizon}"


# --------------------------------------------------------------------------- #
# The subprocess seam
# --------------------------------------------------------------------------- #

Runner = Callable[[List[str], str], CommandResult]


def make_subprocess_runner(
    python_executable: str,
    cwd: str,
    log_dir: str,
    timeout: Optional[float] = None,
) -> Runner:
    """Build the real runner. Tests substitute their own two-arg callable."""

    def _run(argv: List[str], stage_name: str) -> CommandResult:
        os.makedirs(log_dir, exist_ok=True)
        idx = 1
        while os.path.exists(os.path.join(log_dir, f"{stage_name}-{idx}.log")):
            idx += 1
        log_path = os.path.join(log_dir, f"{stage_name}-{idx}.log")
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            rc, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as exc:
            rc = 124
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            err = (exc.stderr if isinstance(exc.stderr, str) else "") + (
                f"\n[timed out after {timeout}s]"
            )
            timed_out = True
        except FileNotFoundError as exc:
            rc, out, err, timed_out = 127, "", str(exc), False
        duration = time.time() - started
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("$ " + " ".join(_quote(a) for a in argv) + "\n\n")
            fh.write(f"[exit {rc} in {duration:.2f}s]\n\n--- stdout ---\n{out}\n")
            fh.write(f"\n--- stderr ---\n{err}\n")
        return CommandResult(
            argv=list(argv),
            exit_code=rc,
            stdout=out or "",
            stderr=err or "",
            duration_seconds=duration,
            log_path=log_path,
            timed_out=timed_out,
        )

    return _run


class StageContext:
    """Per-run state handed to every stage."""

    def __init__(self, config: PipelineConfig, runner: Runner):
        self.config = config
        self.runner = runner
        self.provenance_verified = False
        self.provenance_detail = ""

    def script(self, key: str) -> str:
        return os.path.join(SRC_DIR, SCRIPTS[key])

    def run(self, argv: List[str], stage: str) -> CommandResult:
        if self.config.dry_run:
            # Planned, never launched. Recorded in the manifest like any other
            # command so --dry-run output is the real execution plan.
            return CommandResult(
                argv=list(argv),
                exit_code=0,
                stdout="",
                stderr="",
                duration_seconds=0.0,
                log_path=None,
                launched=False,
            )
        return self.runner(argv, stage)

    def precondition(self, name: str, ok: bool, detail: str) -> Check:
        """An input-existence precondition.

        Under --dry-run a missing input is reported as a WARNING, not a
        failure: the point of a dry run is to show the plan and flag what will
        break, not to claim the run failed before it started. A real run still
        refuses to proceed.
        """
        return Check(name, ok, detail, SEV_WARN if self.config.dry_run else SEV_ERROR)


# --------------------------------------------------------------------------- #
# Stage 1 - fetch
# --------------------------------------------------------------------------- #

def _verify_corpus(corpus: str, min_stations: int) -> List[Check]:
    checks: List[Check] = []
    if not os.path.isfile(corpus):
        checks.append(Check("corpus_exists", False,
                           f"corpus was not written: {corpus}"))
        return checks
    size = os.path.getsize(corpus)
    checks.append(Check("corpus_exists", size > 0,
                        f"{corpus} ({size:,} bytes)" if size else
                        f"{corpus} exists but is EMPTY (0 bytes)"))

    stations: set = set()
    rows = 0
    bad_header = None
    try:
        with open(corpus, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            for needed in ("station_id", "recorded_at"):
                if needed not in fields:
                    bad_header = needed
                    break
            if bad_header is None:
                for rec in reader:
                    rows += 1
                    sid = (rec.get("station_id") or "").strip()
                    if sid:
                        stations.add(sid)
    except (OSError, UnicodeDecodeError) as exc:
        checks.append(Check("corpus_readable", False,
                           f"could not read {corpus}: {exc}"))
        return checks

    if bad_header is not None:
        checks.append(Check("corpus_header", False,
                           f"corpus header is missing required column "
                           f"{bad_header!r}; the pipeline stages downstream expect "
                           f"the fetch_current_telemetry.py schema"))
        return checks
    checks.append(Check("corpus_header", True,
                       f"header carries station_id/recorded_at ({len(fields)} columns)"))
    checks.append(Check("corpus_rows", rows > 0,
                       f"{rows:,} data rows"
                       if rows else "corpus has a header and ZERO data rows"))
    checks.append(Check(
        "corpus_station_coverage", len(stations) >= min_stations,
        f"{len(stations)} distinct station_id(s); required >= {min_stations}"
        + ("" if len(stations) >= min_stations else
           " -- a fetch that collapsed most stations looks like the "
           "timestamp-keyed merge bug this corpus has hit before"),
    ))
    return checks


def stage_fetch(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("fetch", STATUS_PASSED)
    res.consumed["corpus.pre"] = hash_path(cfg.corpus)

    if os.path.exists(cfg.corpus) and not cfg.force:
        msg = (f"refusing to overwrite existing corpus {cfg.corpus}. Pass --force to "
               f"replace it, or point --corpus at a new path. The pipeline never "
               f"deletes or silently clobbers data files.")
        res.checks.append(ctx.precondition("corpus_overwrite_guard", False, msg))
        if cfg.dry_run:
            res.status = STATUS_DRY_RUN
            res.note(msg)
        else:
            res.status = STATUS_FAILED
            res.fail(msg)
        return res
    res.checks.append(Check("corpus_overwrite_guard", True,
                            "no pre-existing corpus, or --force given"))

    argv = [cfg.python, ctx.script("fetch"),
            "--start", cfg.fetch_start,
            "--end", cfg.fetch_end or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "--out", cfg.corpus,
            "--take", str(cfg.fetch_take),
            "--delay", str(cfg.fetch_delay)]
    if cfg.fetch_stations:
        argv += ["--stations", cfg.fetch_stations]
    if cfg.fetch_filter_outliers:
        argv.append("--filter-outliers")

    cmd = ctx.run(argv, "fetch")
    res.commands.append(cmd)
    res.duration_seconds = cmd.duration_seconds
    res.checks.append(Check("exit_code", cmd.exit_code == 0,
                            f"fetch_current_telemetry.py exited {cmd.exit_code}"))

    if cfg.dry_run:
        res.status = STATUS_DRY_RUN
        res.checks.append(Check("corpus_content", True,
                                "skipped: --dry-run did not write a corpus",
                                SEV_INFO))
        res.produced["corpus"] = {"path": cfg.corpus, "status": "not_written_dry_run"}
        return res

    res.checks.extend(_verify_corpus(cfg.corpus, cfg.min_fetch_stations))
    res.produced["corpus"] = hash_path(cfg.corpus)
    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 2 - train
# --------------------------------------------------------------------------- #

def train_artifacts(candidate_dir: str, horizons: Sequence[int]) -> List[str]:
    names = [
        "baseline_manifest.json",
        "predictive_quality_scorecard.json",
        "model_comparison_report.json",
        "uncertainty_report.json",
        "anomaly_report.json",
        "information_ceiling_report.json",
        "model_selection_report.json",
        "major_improvement_baseline_freeze.json",
        "champion_challenger_report.json",
        "candidate_manifest_summary.json",
    ]
    for h in horizons:
        names += [
            f"candidate_h{h}h.pt",
            f"candidate_h{h}h_manifest.json",
            f"candidate_h{h}h_calibration.json",
            f"candidate_h{h}h_predictions.csv",
        ]
    return [os.path.join(candidate_dir, n) for n in names]


def stage_train(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("train", STATUS_PASSED)
    horizons = cfg.horizon_list()
    res.consumed["corpus"] = hash_path(cfg.corpus)

    blocked = None
    if not os.path.isfile(cfg.corpus):
        blocked = (f"training corpus does not exist: {cfg.corpus}. Run --stage fetch "
                   f"first, or point --corpus at an existing corpus.")
        res.checks.append(ctx.precondition("corpus_present", False, blocked))
        if not cfg.dry_run:
            res.status = STATUS_FAILED
            res.fail(blocked)
            return res

    argv = [cfg.python, ctx.script("train"),
            "--weather-csv", cfg.corpus,
            "--output-dir", cfg.candidate_dir,
            "--horizons", cfg.horizons,
            "--epochs", str(cfg.epochs),
            "--patience", str(cfg.patience),
            "--lr", str(cfg.lr)]
    if cfg.seed is not None:
        argv += ["--seed", str(cfg.seed)]
    if cfg.train_commit:
        argv += ["--commit", cfg.train_commit]

    cmd = ctx.run(argv, "train")
    res.commands.append(cmd)
    res.duration_seconds = cmd.duration_seconds

    if cfg.dry_run:
        res.status = STATUS_DRY_RUN
        if blocked:
            res.note(blocked)
        res.checks.append(Check("exit_code", True, "skipped: --dry-run", SEV_INFO))
        res.checks.append(Check("artifacts", True, "skipped: --dry-run", SEV_INFO))
        res.produced["planned"] = [
            os.path.relpath(p, REPO_ROOT).replace("\\", "/")
            for p in train_artifacts(cfg.candidate_dir, horizons)]
        return res

    # --- BOTH halves of the check, and the discrepancy between them ---------
    expected = train_artifacts(cfg.candidate_dir, horizons)
    missing = [p for p in expected if not os.path.isfile(p)]
    present = [p for p in expected if os.path.isfile(p)]
    empty = [p for p in present if os.path.getsize(p) == 0]

    exit_ok = cmd.exit_code == 0
    artifacts_ok = not missing and not empty

    res.checks.append(Check(
        "exit_code", exit_ok,
        f"train_predictive_quality.py exited {cmd.exit_code}"))
    if empty:
        res.checks.append(Check("artifacts", False,
                                f"{len(empty)} candidate artifact(s) exist but are "
                                f"0 bytes: {[os.path.basename(p) for p in empty]}"))
    elif missing:
        res.checks.append(Check(
            "artifacts", False,
            f"{len(missing)} of {len(expected)} expected candidate artifacts are "
            f"missing from {cfg.candidate_dir}: "
            f"{[os.path.basename(p) for p in missing]}"))
    else:
        res.checks.append(Check(
            "artifacts", True,
            f"all {len(expected)} expected candidate artifacts present and non-empty "
            f"(checkpoints for horizons {horizons}, plus the 10 audit reports)"))

    # The exact historical failure: rc=1 at the very end (promotion audit) while
    # five perfectly good checkpoints sit on disk. Do not let a green artifact
    # check launder a red exit code -- and do not let a red exit code hide a
    # usable artifact set. Say which one it is.
    if not exit_ok and artifacts_ok:
        msg = (
            f"DISCREPANCY: train_predictive_quality.py exited {cmd.exit_code} but "
            f"all {len(expected)} candidate artifacts are present and non-empty. "
            f"Checkpoints exist, so the artifacts are usable, but the script's own "
            f"final step failed and the run is NOT a success. Treating as FAILURE. "
            f"stderr tail: {_tail(cmd.stderr, 500).strip() or '(empty)'}"
        )
        res.checks.append(Check("exit_artifact_discrepancy", False, msg))
        res.fail(msg)
    elif exit_ok and not artifacts_ok:
        msg = (
            f"DISCREPANCY: train_predictive_quality.py exited 0 but the artifact set "
            f"is incomplete. A zero exit is not evidence of a completed training run."
        )
        res.checks.append(Check("exit_artifact_discrepancy", False, msg))
        res.fail(msg)
    elif exit_ok and artifacts_ok:
        res.checks.append(Check("exit_artifact_discrepancy", True,
                                "exit code and artifact set agree"))

    for h in horizons:
        p = os.path.join(cfg.candidate_dir, f"candidate_h{h}h.pt")
        fp = hash_path(p)
        if fp:
            res.produced[os.path.basename(p)] = fp
    res.produced["candidate_dir"] = hash_path(cfg.candidate_dir)
    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 3 - benchmark
# --------------------------------------------------------------------------- #

def _metric_ok(m: Any) -> bool:
    return (
        isinstance(m, dict)
        and isinstance(m.get("mae"), (int, float))
        and m.get("n", 0) > 0
    )


def verify_benchmark_results(
    results_path: str,
    label: str,
    horizons: Sequence[int],
    min_nwp_scored: int = 1,
    require_full_nwp: bool = False,
) -> List[Check]:
    """Substantive verification of benchmark_vs_nwp.py output.

    Exit 0 is not evidence. This rejects the exact hollow run that shipped: all
    seven NWP models printed ``n/a`` and the script still emitted a rank.
    """
    checks: List[Check] = []
    if not os.path.isfile(results_path):
        checks.append(Check("results_file", False,
                           f"benchmark results JSON was not written: {results_path}"))
        return checks
    size = os.path.getsize(results_path)
    checks.append(Check("results_file", size > 0,
                        f"{results_path} ({size:,} bytes)" if size else
                        f"{results_path} is EMPTY"))

    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (ValueError, OSError) as exc:
        checks.append(Check("results_parse", False,
                           f"benchmark results are not valid JSON: {exc}"))
        return checks
    checks.append(Check("results_parse", True, "results JSON parses"))

    results = doc.get("results")
    if not isinstance(results, dict) or not results:
        checks.append(Check("results_present", False,
                           "results JSON has no non-empty 'results' object; "
                           "benchmark_vs_nwp.py --out wrote a shell"))
        return checks
    checks.append(Check("results_present", True,
                        f"{len(results)} horizon key(s): {sorted(results)}"))

    missing_h = [h for h in horizons if f"h{h}" not in results]
    checks.append(Check(
        "horizons_present", not missing_h,
        "all requested horizons present: " + ", ".join(f"h{h}" for h in horizons)
        if not missing_h else
        f"requested horizons missing from results: "
        f"{[f'h{h}' for h in missing_h]} (benchmark_vs_nwp.py skips a horizon "
        f"whose test split is empty -- that is a data problem, not a pass)"))

    hollow = []
    zero_n = []
    missing_ours = []
    nwp_hist: Dict[str, int] = {}
    for h in horizons:
        key = f"h{h}"
        block = results.get(key)
        if not isinstance(block, dict):
            continue
        for var in BENCHMARK_VARIABLES:
            row = block.get(var)
            if not isinstance(row, dict):
                hollow.append(f"{key}/{var}: channel block missing")
                continue
            ours = row.get(label)
            if not _metric_ok(ours):
                missing_ours.append(
                    f"{key}/{var}: model under test ({label!r}) has no score "
                    f"(value={ours!r})")
                continue
            if _metric_ok(ours) and not _metric_ok(row.get("persistence")):
                hollow.append(f"{key}/{var}: persistence baseline unscored, so the "
                              f"skill column would be meaningless")
            pers_n = (row.get("persistence") or {}).get("n")
            if isinstance(pers_n, int) and pers_n != ours.get("n"):
                zero_n.append(f"{key}/{var}: N={ours.get('n')} for the model under "
                              f"test but N={pers_n} for persistence -- different rows")
            scored = [nm for nm, mm in row.items() if nm in NWP_DISPLAY_NAMES and _metric_ok(mm)]
            nwp_hist[f"h{h}"] = max(nwp_hist.get(f"h{h}", 0), len(scored))
            required = len(NWP_DISPLAY_NAMES) if require_full_nwp else min_nwp_scored
            if len(scored) < required:
                hollow.append(
                    f"{key}/{var}: only {len(scored)} of {len(NWP_DISPLAY_NAMES)} NWP "
                    f"models produced a score (need >= {required}); "
                    f"scored={scored or 'none'} -- this is the silent n/a run")

        rain = block.get("_rain_brier")
        if not isinstance(rain, dict):
            hollow.append(f"{key}/_rain_brier: rain occurrence block missing")
        else:
            for needed in (label, "persistence", "climatology (constant)"):
                v = rain.get(needed)
                if not isinstance(v, (int, float)) or v != v:
                    hollow.append(f"{key}/_rain_brier: {needed!r} is not a number "
                                  f"(value={v!r})")

    checks.append(Check(
        "model_under_test_scored", not missing_ours,
        f"model under test {label!r} scored on every channel at every horizon"
        if not missing_ours else "; ".join(missing_ours)))

    total_nwp = max(nwp_hist.values()) if nwp_hist else 0
    checks.append(Check(
        "nwp_models_scored", not hollow,
        f"scored channels complete; best horizon had {total_nwp} of "
        f"{len(NWP_DISPLAY_NAMES)} NWP models producing numbers"
        if not hollow else "; ".join(hollow)))

    checks.append(Check(
        "test_window_count", not zero_n,
        "model under test and persistence scored on identical window counts"
        if not zero_n else "; ".join(zero_n)))

    return checks


def stage_benchmark(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("benchmark", STATUS_PASSED)
    res.consumed["corpus"] = hash_path(cfg.corpus)
    for h in cfg.horizon_list():
        res.consumed[f"candidate_h{h}h.pt"] = hash_path(
            os.path.join(cfg.candidate_dir, f"candidate_h{h}h.pt"))
    blocked = None
    if not os.path.isdir(cfg.candidate_dir):
        blocked = (f"candidate directory does not exist: {cfg.candidate_dir}. "
                   f"Run --stage train first.")
        res.checks.append(ctx.precondition("candidate_dir", False, blocked))
        if not cfg.dry_run:
            res.status = STATUS_FAILED
            res.fail(blocked)
            return res

    argv = [cfg.python, ctx.script("benchmark"),
            "--weather-csv", cfg.corpus,
            "--candidate-dir", cfg.candidate_dir,
            "--horizons", cfg.horizons,
            "--out", cfg.benchmark_out,
            "--label", cfg.benchmark_label]
    cmd = ctx.run(argv, "benchmark")
    res.commands.append(cmd)
    res.duration_seconds = cmd.duration_seconds
    res.checks.append(Check("exit_code", cmd.exit_code == 0,
                            f"benchmark_vs_nwp.py exited {cmd.exit_code}"))

    if cfg.dry_run:
        res.status = STATUS_DRY_RUN
        if blocked:
            res.note(blocked)
        res.checks.append(Check("results_content", True, "skipped: --dry-run", SEV_INFO))
        return res

    res.checks.extend(verify_benchmark_results(
        cfg.benchmark_out, cfg.benchmark_label, cfg.horizon_list(),
        cfg.min_nwp_scored, cfg.require_full_nwp))
    res.produced[os.path.basename(cfg.benchmark_out)] = hash_path(cfg.benchmark_out)
    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 4 - baseline
# --------------------------------------------------------------------------- #

def verify_baseline_markdown(path: str, corpus: str) -> List[Check]:
    """capture_release_baseline.py exits 0 on an EMPTY scoreboard.

    If the benchmark JSON it was handed has no row matching the model label it
    looks for, it emits a well-formed markdown file containing no measurements.
    That is a hollow success, so we require actual rows and require the corpus
    hash it recorded to be the corpus we asked about.
    """
    checks: List[Check] = []
    if not os.path.isfile(path):
        checks.append(Check("baseline_file", False,
                           f"baseline markdown was not written: {path}"))
        return checks
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        checks.append(Check("baseline_file", False, f"could not read {path}: {exc}"))
        return checks

    checks.append(Check("baseline_file", len(text) > 0,
                        f"{path} ({len(text):,} chars)"))
    checks.append(Check("baseline_header", "# Release baseline" in text,
                        "markdown carries the expected header"
                        if "# Release baseline" in text else
                        f"{path} does not look like a release baseline capture"))

    rows = [ln for ln in text.splitlines() if re.match(r"^\|\s*\+\d+h\s*\|", ln)]
    n_horizons = len(set(re.findall(r"^\|\s*\+(\d+)h\s*\|", text, re.M)))
    checks.append(Check(
        "baseline_rows", bool(rows),
        f"{len(rows)} scoreboard row(s) across {n_horizons} horizon(s)"
        if rows else
        f"{path} contains a scoreboard header but ZERO measurements -- the "
        f"benchmark JSON had no row matching the model label, so this is an "
        f"empty capture presented as a success"))

    if os.path.isfile(corpus):
        want = sha256_file(corpus)
        m = re.search(r"corpus sha256:\s*`([0-9a-f]{64})`", text)
        got = m.group(1) if m else None
        checks.append(Check(
            "baseline_corpus_binding", got == want,
            f"baseline records the corpus we scored ({want[:16]}...)"
            if got == want else
            f"baseline corpus hash {got!r} != {os.path.basename(corpus)} hash "
            f"{want!r}: the baseline was captured against a different corpus"))
    else:
        checks.append(Check("baseline_corpus_binding", False,
                           f"corpus {corpus} missing, cannot confirm the baseline "
                           f"was captured against it", SEV_WARN))
    return checks


def stage_baseline(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("baseline", STATUS_PASSED)
    res.consumed["corpus"] = hash_path(cfg.corpus)
    res.consumed["benchmark"] = hash_path(cfg.benchmark_out)
    res.consumed["bundles_dir"] = hash_path(cfg.bundles_dir)

    blocked = None
    if not os.path.isfile(cfg.benchmark_out):
        blocked = (f"benchmark results not found: {cfg.benchmark_out}. "
                   f"capture_release_baseline.py reads a benchmark JSON and exits 1 "
                   f"without one; run --stage benchmark first.")
        res.checks.append(ctx.precondition("benchmark_present", False, blocked))
        if not cfg.dry_run:
            res.status = STATUS_FAILED
            res.fail(blocked)
            return res

    argv = [cfg.python, ctx.script("baseline"),
            "--benchmark", cfg.benchmark_out,
            "--corpus", cfg.corpus,
            "--bundle-dir", cfg.bundles_dir,
            "--out", cfg.baseline_out]
    cmd = ctx.run(argv, "baseline")
    res.commands.append(cmd)
    res.duration_seconds = cmd.duration_seconds
    res.checks.append(Check("exit_code", cmd.exit_code == 0,
                            f"capture_release_baseline.py exited {cmd.exit_code}"))

    if cfg.dry_run:
        res.status = STATUS_DRY_RUN
        if blocked:
            res.note(blocked)
        res.checks.append(Check("baseline_content", True, "skipped: --dry-run", SEV_INFO))
        return res

    res.checks.extend(verify_baseline_markdown(cfg.baseline_out, cfg.corpus))
    res.produced[os.path.basename(cfg.baseline_out)] = hash_path(cfg.baseline_out)
    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 5 - bundle (provenance-gated)
# --------------------------------------------------------------------------- #

def overwrite_conflicts(cfg: PipelineConfig) -> List[Dict[str, str]]:
    """Targets the bundle stage would clobber. Each needs an explicit --force.

    Covers both named hazards: an existing bundle directory, and a promoted
    policy. The per-horizon ``inference_policy.json`` inside a bundle *is* the
    promoted policy for that horizon, so it is listed explicitly as well as
    being covered by its directory.
    """
    conflicts: List[Dict[str, str]] = []
    for h in CANONICAL_HORIZONS:
        hdir = os.path.join(cfg.bundles_dir, bundle_dir_name(h))
        pol = os.path.join(hdir, "inference_policy.json")
        if os.path.isdir(hdir) and any(
            os.path.isfile(os.path.join(hdir, f)) for f in BUNDLE_FILES
        ):
            conflicts.append({
                "kind": "bundle_dir",
                "path": hdir,
                "detail": f"bundle directory for horizon +{h}h already exists",
            })
        if os.path.isfile(pol):
            conflicts.append({
                "kind": "promoted_policy",
                "path": pol,
                "detail": f"a promoted inference policy already exists for +{h}h",
            })
    promoted = os.path.join(cfg.data_dir, "inference_policy.json")
    if os.path.isfile(promoted):
        conflicts.append({
            "kind": "promoted_policy",
            "path": promoted,
            "detail": "a promoted inference_policy.json is in place; repackaging "
                      "bundles changes what that policy points at",
        })
    return conflicts


def verify_bundles(bundles_dir: str, horizons: Sequence[int]) -> List[Check]:
    checks: List[Check] = []
    missing: List[str] = []
    empty: List[str] = []
    hash_mismatch: List[str] = []
    horizon_mismatch: List[str] = []
    for h in horizons:
        hdir = os.path.join(bundles_dir, bundle_dir_name(h))
        for fname in BUNDLE_FILES:
            fp = os.path.join(hdir, fname)
            if not os.path.isfile(fp):
                missing.append(f"{bundle_dir_name(h)}/{fname}")
            elif os.path.getsize(fp) == 0:
                empty.append(f"{bundle_dir_name(h)}/{fname}")
        man_p = os.path.join(hdir, "bundle_manifest.json")
        if os.path.isfile(man_p):
            try:
                with open(man_p, "r", encoding="utf-8") as fh:
                    man = json.load(fh)
            except (ValueError, OSError) as exc:
                hash_mismatch.append(f"{bundle_dir_name(h)}: manifest unreadable ({exc})")
                continue
            if man.get("horizon_hours") != h:
                horizon_mismatch.append(
                    f"{bundle_dir_name(h)}: manifest says horizon_hours="
                    f"{man.get('horizon_hours')!r}")
            ck = os.path.join(hdir, "checkpoint.pt")
            claimed = man.get("checkpoint_sha256")
            if os.path.isfile(ck) and isinstance(claimed, str) and claimed:
                actual = sha256_file(ck)
                if actual != claimed:
                    hash_mismatch.append(
                        f"{bundle_dir_name(h)}: checkpoint.pt sha256 {actual[:12]}... "
                        f"!= manifest {claimed[:12]}...")
    checks.append(Check("bundle_files_present", not missing,
                        f"all {len(horizons)}x{len(BUNDLE_FILES)} bundle files present"
                        if not missing else
                        f"missing bundle file(s): {missing}"))
    checks.append(Check("bundle_files_non_empty", not empty,
                        "no 0-byte bundle files" if not empty else
                        f"0-byte bundle file(s): {empty}"))
    checks.append(Check("bundle_horizon_labels", not horizon_mismatch,
                        "every manifest labels its own horizon correctly"
                        if not horizon_mismatch else "; ".join(horizon_mismatch)))
    checks.append(Check("bundle_checkpoint_hashes", not hash_mismatch,
                        "every bundle_manifest.json checkpoint_sha256 matches the "
                        "checkpoint it ships"
                        if not hash_mismatch else "; ".join(hash_mismatch)))
    return checks


def stage_bundle(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("bundle", STATUS_PASSED)
    res.consumed["data_dir"] = {"type": "dir_ref", "path": cfg.data_dir}

    # --- gate 1: never clobber without --force -----------------------------
    conflicts = overwrite_conflicts(cfg)
    if conflicts and not cfg.force:
        listing = "; ".join(f"[{c['kind']}] {c['path']} ({c['detail']})"
                            for c in conflicts)
        msg = (f"refusing to overwrite existing release artifacts without --force. "
               f"{len(conflicts)} conflict(s): {listing}. The pipeline never "
               f"deletes data files and never silently replaces a promoted policy.")
        res.checks.append(Check("overwrite_guard", False, msg))
        if cfg.dry_run:
            res.status = STATUS_DRY_RUN
            res.note(msg)
            return res
        res.status = STATUS_BLOCKED
        res.fail(msg)
        return res
    res.checks.append(Check("overwrite_guard", True,
                            f"{len(conflicts)} pre-existing target(s); --force given"
                            if conflicts else "no pre-existing bundle dirs or "
                                            "promoted policies"))

    # --- gate 2: provenance must pass BEFORE generate_bundles.py runs -------
    # This is a preflight inside the stage rather than a separate ordering
    # requirement, so it holds when --stage bundle is run on its own.
    prov_argv = [cfg.python, ctx.script("provenance")]
    if cfg.allow_commit:
        prov_argv += ["--allow-commit", cfg.allow_commit]
    if ctx.provenance_verified:
        res.note("provenance already verified earlier in this run; re-running the "
                 "gate anyway because bundle contents may have changed")
    prov = ctx.run(prov_argv, "bundle-provenance")
    res.commands.append(prov)
    prov_ok = prov.exit_code == 0
    res.checks.append(Check(
        "provenance_preflight", prov_ok,
        f"verify_provenance.py exited {prov.exit_code}"
        + ("" if prov_ok else
           " -- generate_bundles.py was NOT run. Packaging artifacts that fail "
           "the provenance gate is exactly the failure this pipeline exists to "
           "prevent. Use --allow-commit <sha> if the gate is rejecting a "
           "legitimate historical commit.")))

    if not cfg.dry_run and not prov_ok:
        res.status = STATUS_BLOCKED
        res.fail("provenance verification did not pass; generate_bundles.py was "
                 "not invoked")
        return res

    argv = [cfg.python, ctx.script("bundle"),
            "--data-dir", cfg.data_dir,
            "--output-dir", cfg.bundles_dir]
    cmd = ctx.run(argv, "bundle")
    res.commands.append(cmd)
    res.duration_seconds += cmd.duration_seconds
    res.checks.append(Check("exit_code", cmd.exit_code == 0,
                            f"generate_bundles.py exited {cmd.exit_code}"))

    # Independent re-verification: the same script, --check-only, re-reads what
    # is on disk and re-derives every hash. Belt and braces over the writer.
    check_cmd = ctx.run(argv + ["--check-only"], "bundle-check")
    res.commands.append(check_cmd)
    res.checks.append(Check("check_only", check_cmd.exit_code == 0,
                            f"generate_bundles.py --check-only exited "
                            f"{check_cmd.exit_code}"))

    if cfg.dry_run:
        res.status = STATUS_DRY_RUN
        res.checks.append(Check("bundle_content", True, "skipped: --dry-run", SEV_INFO))
        res.produced["planned"] = [
            os.path.relpath(os.path.join(cfg.bundles_dir, bundle_dir_name(h), f),
                            REPO_ROOT).replace("\\", "/")
            for h in CANONICAL_HORIZONS for f in BUNDLE_FILES]
        return res

    res.checks.extend(verify_bundles(cfg.bundles_dir, CANONICAL_HORIZONS))
    res.produced["bundles_dir"] = hash_path(cfg.bundles_dir)

    # The interface surprise, stated every time rather than buried: these bundles
    # wrap the PRODUCTION lnn_weather_water_h*.pt checkpoints, not the candidate
    # this pipeline just trained.
    candidate_hashes = set()
    for h in CANONICAL_HORIZONS:
        cp = os.path.join(cfg.candidate_dir, f"candidate_h{h}h.pt")
        if os.path.isfile(cp):
            candidate_hashes.add(sha256_file(cp))
    shipped_candidate = False
    for h in CANONICAL_HORIZONS:
        ck = os.path.join(cfg.bundles_dir, bundle_dir_name(h), "checkpoint.pt")
        if os.path.isfile(ck) and sha256_file(ck) in candidate_hashes:
            shipped_candidate = True
            break
    res.checks.append(Check(
        "bundles_package_the_trained_candidate", shipped_candidate,
        "bundles package a candidate_h*.pt checkpoint"
        if shipped_candidate else
        "NOTE: these bundles package the PRODUCTION lnn_weather_water_h*.pt "
        "checkpoints, not the candidate_h*.pt trained in this run. "
        "generate_bundles.py has no candidate input. This run measured a "
        "candidate but did not ship it; promoting it is a separate, manual step.",
        SEV_WARN if not shipped_candidate else SEV_ERROR))
    if not shipped_candidate:
        res.note("candidate was benchmarked but not packaged -- see the "
                 "bundles_package_the_trained_candidate warning")

    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 6 - verify
# --------------------------------------------------------------------------- #

def stage_verify(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("verify", STATUS_PASSED)
    for h in CANONICAL_HORIZONS:
        res.consumed[f"bundle_h{h}"] = hash_path(
            os.path.join(cfg.bundles_dir, bundle_dir_name(h)))

    argv = [cfg.python, ctx.script("provenance")]
    if cfg.allow_commit:
        argv += ["--allow-commit", cfg.allow_commit]
    cmd = ctx.run(argv, "verify-provenance")
    res.commands.append(cmd)
    res.duration_seconds += cmd.duration_seconds
    ok = cmd.exit_code == 0
    res.checks.append(Check("exit_code", ok,
                            f"verify_provenance.py exited {cmd.exit_code}"
                            + ("" if ok else
                               " -- the post-bundle provenance gate failed")))
    if ok and not cfg.dry_run:
        ctx.provenance_verified = True
        ctx.provenance_detail = "verify stage"

    test_path = os.path.join(SRC_DIR, cfg.provenance_test)
    if not os.path.isfile(test_path):
        res.checks.append(Check("provenance_test_present", False,
                                f"provenance test suite not found: {test_path}"))
    else:
        targv = [cfg.python, "-m", "pytest", test_path, "-q"]
        tcmd = ctx.run(targv, "verify-provenance-tests")
        res.commands.append(tcmd)
        res.duration_seconds += tcmd.duration_seconds
        if cfg.dry_run:
            res.checks.append(Check("provenance_tests", True,
                                    "skipped: --dry-run", SEV_INFO))
        else:
            summary, counts, problems = parse_pytest_summary(tcmd.stdout)
            res.checks.append(Check(
                "provenance_tests", tcmd.exit_code == 0 and not problems,
                f"{cfg.provenance_test}: {summary} [{counts}]"
                if tcmd.exit_code == 0 and not problems else
                f"{cfg.provenance_test} FAILED (exit {tcmd.exit_code}): {summary}"
                + (f" -- {problems}" if problems else "")))

    res.produced["verification"] = {
        "provenance_exit_code": cmd.exit_code,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if cfg.dry_run:
        res.status = STATUS_DRY_RUN
    else:
        res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 7 - gate
# --------------------------------------------------------------------------- #

# release_gate.py exit codes, from its own module (EXIT_MEANING). rc=0 is the
# ONLY code that means "the gate ran and handed you promotions"; 1/2/3 all mean
# no promotion is being made, and 3 means the gate did not run at all.
GATE_EXIT_OK = 0
GATE_EXIT_MEANING = {
    0: "gate ran; decision complete; act on the listed promotions",
    1: "BLOCKED: the gate could not make a decision",
    2: "the gate decided and the challenger lost every cell",
    3: "UNUSABLE_INPUT: the gate could not run",
}


def _verify_gate_verdict(path: str, gate_rc: int) -> List[Check]:
    """The gate is the step whose whole purpose is producing a decision.

    An exit code with no verdict document on disk is the failure mode
    release_gate.py was written to prevent, so we check for the artefact and not
    only the process status.
    """
    checks: List[Check] = []
    if not os.path.isfile(path):
        checks.append(Check("gate_verdict_written", False,
                           f"release gate exited {gate_rc} but wrote no verdict "
                           f"document at {path}: a decision procedure that "
                           f"reports success without producing a decision"))
        return checks
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (ValueError, OSError) as exc:
        checks.append(Check("gate_verdict_written", False,
                           f"gate verdict at {path} is not valid JSON: {exc}"))
        return checks
    checks.append(Check("gate_verdict_written", True, path))

    rec = doc.get("recommendation")
    counts = doc.get("counts") or {}
    checks.append(Check(
        "gate_recommendation", rec in ("PROMOTE_ALL", "SPLIT"),
        f"gate recommendation {rec!r} with counts {counts}"
        if rec in ("PROMOTE_ALL", "SPLIT") else
        f"gate recommendation is {rec!r} (counts {counts}); PROMOTE_ALL/SPLIT is "
        f"the only outcome that releases anything"))

    integrity = doc.get("integrity") or {}
    bad = sorted(k for k, v in integrity.items()
                 if isinstance(v, dict) and v.get("status") not in ("PASS", None))
    checks.append(Check(
        "gate_integrity", not bad,
        "all gate integrity checks PASS" if not bad else
        f"gate integrity blockers: "
        f"{[(k, (integrity[k] or {}).get('reason')) for k in bad]}"))
    return checks


def stage_gate(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("gate", STATUS_PASSED)
    script = ctx.script("gate")

    def _skip(reason: str) -> StageResult:
        res.commands = []
        res.status = STATUS_SKIPPED
        res.fail(reason)
        # A skipped stage has not FAILED -- it declined to run, and the stage
        # status plus the manifest note are what carry that. Keeping the check
        # at warning severity stops "skipped" being counted as a hard failure,
        # which is how a dry run or a diagnostic run stays honest about rc.
        res.checks.append(Check("gate_ran", False, reason, SEV_WARN))
        res.note("SKIPPED: the release gate did not run and did not pass")
        print("\n" + "!" * 78)
        print("!! RELEASE GATE SKIPPED -- it did not run and it did not pass.")
        for ln in _wrap(reason, 74):
            print("!! " + ln)
        print("!! This run is INCOMPLETE. Do not report it as a green pipeline.")
        print("!" * 78 + "\n")
        if cfg.require_gate:
            res.status = STATUS_FAILED
            res.checks.append(Check(
                "gate_required", False,
                "--require-gate was given and the gate did not produce a verdict"))
        return res

    if not os.path.isfile(script):
        return _skip(
            f"release_gate.py is not present at {script}. The release gate step "
            f"was SKIPPED, not passed. This pipeline does not invent a gate: a run "
            f"with no gate is a run with one unverified step.")

    res.checks.append(Check("gate_script_present", True, script))

    # release_gate.py requires --inputs (a JSON bundle of per-cell paired
    # validation-split evidence) and --out. The pipeline deliberately does NOT
    # synthesise that bundle: benchmark_vs_nwp.py reports aggregate TEST-split
    # metrics, while the gate decides on validation-split paired per-row evidence
    # and structurally refuses to decide on test data. Faking a validation label
    # over test numbers would be the exact over-claim this repository keeps
    # warning about, so the gate is skipped until a real evidence bundle exists.
    if not cfg.gate_inputs:
        return _skip(
            "release_gate.py requires --inputs <evidence.json> and --out <verdict.json>, "
            "and no evidence bundle was supplied (--gate-inputs). The pipeline will "
            "not fabricate one: benchmark_vs_nwp.py emits aggregate TEST-split metrics, "
            "while the gate decides on validation-split paired per-row evidence and "
            "structurally refuses to decide on test data. Supply --gate-inputs with a "
            "real promotion-evidence bundle to run the gate.")

    if not os.path.isfile(cfg.gate_inputs):
        msg = f"--gate-inputs was given but the file does not exist: {cfg.gate_inputs}"
        res.checks.append(ctx.precondition("gate_inputs_present", False, msg))
        if cfg.dry_run:
            res.status = STATUS_DRY_RUN
            return res
        res.status = STATUS_FAILED
        res.fail(msg)
        return res

    argv = [cfg.python, script,
            "--inputs", cfg.gate_inputs,
            "--out", cfg.gate_verdict_out] + list(cfg.gate_args)
    cmd = ctx.run(argv, "gate")
    res.commands.append(cmd)
    res.duration_seconds += cmd.duration_seconds

    meaning = GATE_EXIT_MEANING.get(cmd.exit_code, "unrecognised exit code")
    res.checks.append(Check(
        "exit_code", cmd.exit_code == GATE_EXIT_OK,
        f"release_gate.py exited {cmd.exit_code} ({meaning})"
        + ("" if cmd.exit_code == GATE_EXIT_OK else
           " -- only exit 0 means the gate produced promotions. 1/2 mean the "
           "challenger was blocked or lost; 3 means the gate never ran.")))

    if cfg.dry_run:
        res.checks.append(Check("gate_verdict_written", True,
                                "skipped: --dry-run", SEV_INFO))
        res.status = STATUS_DRY_RUN
        return res

    res.checks.extend(_verify_gate_verdict(cfg.gate_verdict_out, cmd.exit_code))
    res.produced["gate_verdict"] = hash_path(cfg.gate_verdict_out)
    res.produced["gate_stdout_tail"] = _tail(cmd.stdout, 1500)
    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #
# Stage 8 - test
# --------------------------------------------------------------------------- #

def stage_test(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    res = StageResult("test", STATUS_PASSED)
    argv = [cfg.python, "-m", "pytest", cfg.pytest_target] + list(cfg.pytest_args)
    cmd = ctx.run(argv, "test")
    res.commands.append(cmd)
    res.duration_seconds = cmd.duration_seconds

    res.checks.append(Check(
        "exit_code", cmd.exit_code == 0,
        f"pytest exited {cmd.exit_code}"))

    if cfg.dry_run:
        res.checks.append(Check("test_results", True,
                                "skipped: --dry-run", SEV_INFO))
        res.status = STATUS_DRY_RUN
        return res

    summary, counts, problems = parse_pytest_summary(cmd.stdout)
    res.checks.append(Check(
        "test_results", not problems,
        f"{summary} [{counts}]" if not problems else
        f"pytest summary reports problems: {problems} (summary: {summary!r})"))
    res.produced["pytest_summary"] = {"summary": summary, "counts": counts}
    res.status = STATUS_PASSED if not res.blocking_failures else STATUS_FAILED
    return res


# --------------------------------------------------------------------------- #

STAGE_FUNCS: Dict[str, Callable[[StageContext], StageResult]] = {
    "fetch": stage_fetch,
    "train": stage_train,
    "benchmark": stage_benchmark,
    "baseline": stage_baseline,
    "bundle": stage_bundle,
    "verify": stage_verify,
    "gate": stage_gate,
    "test": stage_test,
}


def run_pipeline(
    config: PipelineConfig,
    runner: Optional[Runner] = None,
    manifest_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], int]:
    """Execute the selected stages. Returns (manifest, exit_code)."""
    emit = log or (lambda m: print(m))
    log_dir = os.path.join(config.run_dir, "logs")
    os.makedirs(config.run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if runner is None:
        runner = make_subprocess_runner(
            config.python, REPO_ROOT, log_dir, config.timeout)

    selected = STAGE_ORDER if config.stage == "all" else [config.stage]
    ctx = StageContext(config, runner)

    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": os.path.basename(config.run_dir.rstrip("\\/")) or "adhoc",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "duration_seconds": 0.0,
        "python": config.python,
        "cwd": REPO_ROOT,
        "paths": {
            "repo_root": REPO_ROOT,
            "src_dir": SRC_DIR,
            "data_dir": config.data_dir,
            "run_dir": config.run_dir,
            "corpus": config.corpus,
            "candidate_dir": config.candidate_dir,
            "benchmark_out": config.benchmark_out,
            "baseline_out": config.baseline_out,
            "bundles_dir": config.bundles_dir,
        },
        "invocation": {
            "stage": config.stage,
            "force": config.force,
            "dry_run": config.dry_run,
            "require_gate": config.require_gate,
            "keep_going": config.keep_going,
        },
        "options": {
            "horizons": config.horizons,
            "epochs": config.epochs,
            "patience": config.patience,
            "lr": config.lr,
            "seed": config.seed,
            "benchmark_label": config.benchmark_label,
            "min_nwp_scored": config.min_nwp_scored,
            "require_full_nwp": config.require_full_nwp,
            "min_fetch_stations": config.min_fetch_stations,
            "fetch_start": config.fetch_start,
            "fetch_end": config.fetch_end or "<today>",
            "fetch_take": config.fetch_take,
            "fetch_delay": config.fetch_delay,
            "pytest_target": config.pytest_target,
            "pytest_args": list(config.pytest_args),
            "gate_inputs": config.gate_inputs,
            "gate_verdict_out": config.gate_verdict_out,
            "gate_args": list(config.gate_args),
            "allow_commit": config.allow_commit,
        },
        "stages": [],
        "status": STATUS_PASSED,
        "notes": [],
    }

    overall_started = time.time()
    exit_code = 0
    aborted = False
    gate_skipped = False

    emit("=" * 78)
    emit("PREDICTION MODEL -- FULL PIPELINE")
    emit(f"  run dir   : {config.run_dir}")
    emit(f"  stage     : {config.stage}")
    emit(f"  dry run   : {config.dry_run}")
    emit(f"  force     : {config.force}")
    emit(f"  stages    : {', '.join(selected)}")
    emit("=" * 78)

    for name in selected:
        if aborted:
            rec = StageResult(name, STATUS_HALTED)
            rec.note("HALTED: an earlier stage failed and the pipeline is "
                     "fail-fast by design. This stage did not run and its result "
                     "is unknown, not passed.")
            manifest["stages"].append(rec.to_json())
            continue
        emit("")
        emit(f"--- [{name.upper()}] {STAGE_DESCRIPTIONS.get(name, '')}")
        if config.dry_run:
            emit(f"    (dry run: commands are printed, never executed)")
        stage_started = time.time()
        result = STAGE_FUNCS[name](ctx)
        if not result.duration_seconds:
            result.duration_seconds = time.time() - stage_started
        for c in result.commands:
            tag = "would run" if not c.launched else f"exit {c.exit_code}"
            emit(f"    $ {' '.join(_quote(a) for a in c.argv)}   [{tag}]")
        for c in result.checks:
            mark = "ok  " if c.ok else ("WARN" if c.severity == SEV_WARN else "FAIL")
            emit(f"    [{mark}] {c.name}: {c.detail}")
        for w in result.warnings:
            emit(f"    [WARN] {w}")
        for f in result.failures:
            emit(f"    [FAIL] {f}")
        emit(f"    => {name}: {result.status.upper()} "
             f"({result.duration_seconds:.1f}s)")
        manifest["stages"].append(result.to_json())

        fatal = result.status in (STATUS_FAILED, STATUS_BLOCKED) and \
            not config.keep_going
        if result.status in (STATUS_FAILED, STATUS_BLOCKED):
            # keep_going suppresses the HALT, never the non-zero exit code. A
            # diagnosis run that ends rc=0 reads as a pass to every caller.
            exit_code = 1
        if result.status == STATUS_SKIPPED:
            manifest["notes"].append(
                f"stage '{name}' was SKIPPED, not passed: "
                + (result.failures[0] if result.failures else "unspecified"))
            gate_skipped = True
            if config.require_gate:
                exit_code = 1
                fatal = True
        if fatal:
            aborted = True
            emit("")
            emit(f"!!! {name.upper()} FAILED -- halting. Remaining stages were not "
                 f"run and must not be assumed to have run.")

    # Status is decided once, at the end, from what actually happened. Deciding
    # it inside the loop made a run that halted at the bundle stage report
    # "gate skipped" purely because later stages were marked skipped.
    if exit_code:
        manifest["status"] = STATUS_FAILED
    elif gate_skipped:
        manifest["status"] = "gate_skipped"
    else:
        manifest["status"] = STATUS_PASSED

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_seconds"] = round(time.time() - overall_started, 3)
    manifest["summary"] = {
        s["name"]: s["status"] for s in manifest["stages"]
    }

    out_path = manifest_path or config.manifest_path or os.path.join(
        config.run_dir, "run_manifest.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    emit("")
    emit("=" * 78)
    emit("PIPELINE SUMMARY")
    for s in manifest["stages"]:
        emit(f"  {s['name']:<10} {s['status']:<10} {s['duration_seconds']:>8.1f}s")
    emit(f"  manifest  {out_path}")
    if manifest["status"] == STATUS_PASSED:
        emit("  RESULT: all requested stages passed.")
    elif manifest["status"] == "gate_skipped":
        emit("  RESULT: INCOMPLETE -- the release gate was SKIPPED, not passed. "
             "Do not report this run as a green pipeline.")
    else:
        emit("  RESULT: FAILED. See the stage detail above.")
    emit("=" * 78)

    return manifest, exit_code


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", default="all", choices=STAGE_ORDER + ["all"],
                   help="Run one stage or the whole pipeline (default: all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command and every check without executing "
                        "anything or writing anything.")
    p.add_argument("--force", action="store_true",
                   help="Required to overwrite an existing corpus, an existing "
                        "bundle directory, or a promoted policy.")
    p.add_argument("--require-gate", action="store_true",
                   help="Treat a missing release_gate.py as a hard failure "
                        "instead of a loud skip.")
    p.add_argument("--keep-going", action="store_true",
                   help="Do not halt on the first failing stage (for diagnosis; "
                        "the exit code is still non-zero).")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--run-dir", default=None,
                   help="Where the manifest and stage logs go "
                        "(default: prediction-model/logs/pipeline/<utc-timestamp>).")
    p.add_argument("--manifest", default=None,
                   help="Explicit manifest path (default: <run-dir>/run_manifest.json).")
    p.add_argument("--timeout", type=float, default=None,
                   help="Per-command timeout in seconds (default: none).")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used for every subprocess.")

    g = p.add_argument_group("stage 1: fetch")
    g.add_argument("--corpus", default=None,
                   help="Telemetry corpus CSV (default: "
                        "<data-dir>/weather_telemetry_current.csv).")
    g.add_argument("--fetch-start", default="2026-06-20")
    g.add_argument("--fetch-end", default="")
    g.add_argument("--fetch-take", type=int, default=5000)
    g.add_argument("--fetch-delay", type=float, default=3.5)
    g.add_argument("--fetch-stations", default=None,
                   help="Comma-separated station ids (default: all 16).")
    g.add_argument("--fetch-filter-outliers", action="store_true")
    g.add_argument("--min-fetch-stations", type=int, default=8)

    g = p.add_argument_group("stage 2: train")
    g.add_argument("--candidate-dir", default=None)
    g.add_argument("--horizons", default="1,3,6,12,24")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--patience", type=int, default=12)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--train-commit", default=None)

    g = p.add_argument_group("stage 3/4: benchmark and baseline")
    g.add_argument("--benchmark-out", default=None)
    g.add_argument("--benchmark-label", default="LNN (this project)")
    g.add_argument("--min-nwp-scored", type=int, default=1,
                   help="Minimum NWP models that must produce a number per "
                        "channel per horizon (default: 1).")
    g.add_argument("--require-full-nwp", action="store_true",
                   help="Require all 7 NWP models scored in every channel.")
    g.add_argument("--baseline-out", default=None)
    g.add_argument("--bundles-dir", default=None)

    g = p.add_argument_group("stages 6-8: verify, gate, test")
    g.add_argument("--allow-commit", default=None)
    g.add_argument("--provenance-test", default="test_provenance.py")
    g.add_argument("--gate-inputs", default=None,
                   help="Promotion-evidence JSON for release_gate.py --inputs. "
                        "The gate needs per-cell paired VALIDATION-split evidence, "
                        "which this pipeline does not fabricate from test-split "
                        "benchmark output. Without it the gate stage is SKIPPED "
                        "loudly, not passed.")
    g.add_argument("--gate-verdict-out", default=None,
                   help="Where release_gate.py writes its verdict "
                        "(default: <run-dir>/release_gate_verdict.json).")
    g.add_argument("--gate-args", default="",
                   help="Extra arguments for release_gate.py (space separated).")
    g.add_argument("--pytest-target", default=None)
    g.add_argument("--pytest-args", default="-q",
                   help="Extra pytest arguments (default: '-q').")

    args = p.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    run_dir = args.run_dir or os.path.join(
        DEFAULT_LOG_ROOT,
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = os.path.abspath(run_dir)

    cfg = PipelineConfig(
        data_dir=data_dir,
        run_dir=run_dir,
        stage=args.stage,
        python=args.python,
        force=args.force,
        dry_run=args.dry_run,
        require_gate=args.require_gate,
        min_fetch_stations=args.min_fetch_stations,
        corpus=os.path.abspath(args.corpus) if args.corpus else os.path.join(
            data_dir, "weather_telemetry_current.csv"),
        fetch_start=args.fetch_start,
        fetch_end=args.fetch_end,
        fetch_take=args.fetch_take,
        fetch_delay=args.fetch_delay,
        fetch_stations=args.fetch_stations,
        fetch_filter_outliers=args.fetch_filter_outliers,
        candidate_dir=os.path.abspath(args.candidate_dir) if args.candidate_dir
        else os.path.join(data_dir, "candidate_artifacts"),
        horizons=args.horizons,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
        train_commit=args.train_commit,
        benchmark_out=os.path.abspath(args.benchmark_out) if args.benchmark_out
        else os.path.join(run_dir, "nwp_benchmark_results.json"),
        benchmark_label=args.benchmark_label,
        min_nwp_scored=args.min_nwp_scored,
        require_full_nwp=args.require_full_nwp,
        baseline_out=os.path.abspath(args.baseline_out) if args.baseline_out
        else os.path.join(run_dir, "release_baseline.md"),
        bundles_dir=os.path.abspath(args.bundles_dir) if args.bundles_dir
        else os.path.join(data_dir, "bundles"),
        allow_commit=args.allow_commit,
        provenance_test=args.provenance_test,
        gate_inputs=os.path.abspath(args.gate_inputs) if args.gate_inputs else "",
        gate_verdict_out=os.path.abspath(args.gate_verdict_out)
        if args.gate_verdict_out
        else os.path.join(run_dir, "release_gate_verdict.json"),
        gate_args=args.gate_args.split() if args.gate_args else [],
        pytest_target=os.path.abspath(args.pytest_target) if args.pytest_target
        else SRC_DIR,
        pytest_args=args.pytest_args.split() if args.pytest_args else [],
        timeout=args.timeout,
        keep_going=args.keep_going,
        manifest_path=(os.path.abspath(args.manifest) if args.manifest
                       else os.path.join(run_dir, "run_manifest.json")),
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    if cfg.dry_run:
        print("[dry-run] no subprocess will be launched and no artifact written "
              "except this manifest.")
    _manifest, rc = run_pipeline(cfg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
