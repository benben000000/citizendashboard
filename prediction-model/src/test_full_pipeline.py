"""
Tests for the full-pipeline orchestrator.

WHAT IS MOCKED AND WHAT IS NOT
------------------------------
Only the subprocess layer is mocked. Every verification here runs against a REAL
temporary filesystem holding artifacts shaped exactly like the real scripts
produce, so these tests exercise the actual checks rather than a stubbed version
of them. A test that mocked the verifier would only prove the mock agrees with
itself.

The behaviours under test are the ones that have actually bitten this project:

  * the exact argv each stage must construct, so a renamed or added flag breaks
    a test instead of a 40-minute training run
  * a failing stage halting the pipeline
  * exit-0-with-hollow-output being a FAILURE (benchmark_vs_nwp.py printing n/a
    for all seven NWP models and still emitting a rank)
  * exit-1-with-complete-artifacts being a FAILURE (train_predictive_quality.py
    raising in its promotion audit after writing five good checkpoints)
  * --dry-run launching nothing
  * missing artifacts being detected
  * --force being required before overwriting a bundle dir or a promoted policy
  * the release gate being SKIPPED loudly rather than silently passed

Usage:
    PYTHONPATH=src python src/test_full_pipeline.py
    python -m pytest prediction-model/src/test_full_pipeline.py -q
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import run_full_pipeline as rfp  # noqa: E402

LABEL = "LNN (this project)"
NWP = list(rfp.NWP_DISPLAY_NAMES)
VARS = list(rfp.BENCHMARK_VARIABLES)
HORIZONS = list(rfp.CANONICAL_HORIZONS)


# --------------------------------------------------------------------------- #
# Artifact builders -- these mirror the real scripts' output shapes.
# --------------------------------------------------------------------------- #

def write_corpus(path, stations=16, rows_per_station=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("station_id,recorded_at,temperature,humidity,pressure,"
                 "wind_speed,wind_direction,precipitation\n")
        for s in range(stations):
            sid = f"STATION{s:02d}"
            for r in range(rows_per_station):
                fh.write(f"{sid},2026-09-30T0{r}:00:00.000Z,"
                         f"30.{r},60,1010.{r},3.5,180,0.0\n")
    return path


def write_candidate_artifacts(candidate_dir, horizons=HORIZONS, complete=True):
    os.makedirs(candidate_dir, exist_ok=True)
    names = ["baseline_manifest.json", "predictive_quality_scorecard.json",
             "model_comparison_report.json", "uncertainty_report.json",
             "anomaly_report.json", "information_ceiling_report.json",
             "model_selection_report.json",
             "major_improvement_baseline_freeze.json",
             "champion_challenger_report.json", "candidate_manifest_summary.json"]
    for h in horizons:
        names += [f"candidate_h{h}h.pt", f"candidate_h{h}h_manifest.json",
                  f"candidate_h{h}h_calibration.json",
                  f"candidate_h{h}h_predictions.csv"]
    if not complete:
        # Drop one artifact per horizon: the "files are missing" case.
        names = [n for n in names if n != "candidate_h6h_calibration.json"]
    for n in names:
        with open(os.path.join(candidate_dir, n), "w", encoding="utf-8") as fh:
            fh.write(f'{{"artifact": "{n}"}}\n')
    return candidate_dir


def _metric(mae, n=3209, rmse=None, bias=0.1):
    return {"mae": mae, "rmse": rmse if rmse is not None else mae * 1.4,
            "bias": bias, "n": n, "ci95_mae": [mae * 0.95, mae * 1.05]}


def write_benchmark_results(path, label=LABEL, horizons=HORIZONS, nwp_scored=None,
                            n=3209, hollow=False, drop_horizon=None):
    """Build a benchmark_vs_nwp.py results JSON.

    ``nwp_scored=None`` means every NWP model scored. A list means exactly those
    NWP display names scored and every other NWP entry is ``None`` -- which is
    what the script writes when the lookup misses, and is what it also writes
    when it prints ``n/a`` for all seven.
    """
    if hollow:
        nwp_scored = []
    results = {}
    for h in horizons:
        if drop_horizon is not None and h == drop_horizon:
            continue
        key = f"h{h}"
        block = {}
        for v in VARS:
            row = {label: _metric(0.80, n=n), "persistence": _metric(0.85, n=n)}
            for name in NWP:
                scored = nwp_scored is None or name in nwp_scored
                row[name] = _metric(0.90, n=n) if scored else None
            block[v] = row
        block["_rain_brier"] = {label: 0.11, "persistence": 0.13,
                                "climatology (constant)": 0.12}
        results[key] = block
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)
    return path


def write_baseline_md(path, corpus_sha, rows=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Release baseline: incumbent production model",
        "",
        f"- corpus: `weather_telemetry_current.csv`",
        f"- corpus sha256: `{corpus_sha}`",
        "",
        "## Scoreboard",
        "",
    ]
    for i in range(rows):
        lines.append(f"| +{HORIZONS[i % len(HORIZONS)]}h | 0.800 | 0.850 | "
                     f"+5.9% | 0.812 | 0.905 | 1/9 |")
    lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


def write_bundles(bundles_dir, source_ckpt=None, horizons=HORIZONS, tag="v1"):
    """Write bundle dirs whose manifests hash the checkpoints they ship.

    ``tag`` makes the checkpoint bytes differ between calls so a test can tell a
    repackaged bundle from the one that was already there.
    """
    for h in horizons:
        hdir = os.path.join(bundles_dir, f"h{h}")
        os.makedirs(hdir, exist_ok=True)
        if source_ckpt:
            shutil.copy2(source_ckpt, os.path.join(hdir, "checkpoint.pt"))
        else:
            with open(os.path.join(hdir, "checkpoint.pt"), "wb") as fh:
                fh.write(f"fake-checkpoint-h{h}-{tag}".encode())
        with open(os.path.join(hdir, "inference_policy.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"horizon_hours": h, "model_status": "ACTIVE_PRODUCTION",
                       "model_family": "GarciaWeatherLNN",
                       "checkpoint_sha256": ""}, fh, indent=2)
        with open(os.path.join(hdir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# +{h}h\n")
        ck_sha = rfp.sha256_file(os.path.join(hdir, "checkpoint.pt"))
        pol_path = os.path.join(hdir, "inference_policy.json")
        with open(pol_path, "r", encoding="utf-8") as fh:
            pol = json.load(fh)
        pol["checkpoint_sha256"] = ck_sha
        with open(pol_path, "w", encoding="utf-8") as fh:
            json.dump(pol, fh, indent=2)
        with open(os.path.join(hdir, "bundle_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"bundle_version": "1.0.0", "horizon_hours": h,
                       "implementation_commit": "b72b16ae",
                       "artifact_commit": "b72b16ae",
                       "model_weights_commit": "cf0a37e2",
                       "checkpoint_sha256": ck_sha,
                       "policy_sha256": rfp.sha256_file(pol_path),
                       "raw_weather_dataset_sha256": "a" * 64,
                       "raw_water_dataset_sha256": "b" * 64,
                       "feature_schema": ["temperature"], "model_family": "GarciaWeatherLNN"},
                      fh, indent=2)
    return bundles_dir


# --------------------------------------------------------------------------- #
# The mocked subprocess layer
# --------------------------------------------------------------------------- #

class FakeRunner:
    """Stands in for ``make_subprocess_runner``.

    Records every argv it is handed and, by default, writes the artifacts the
    real script would have written so the substantive checks have something real
    to inspect.
    """

    def __init__(self, env, exit_codes=None, stdout=None, write=True):
        self.env = env
        self.calls = []          # list of (stage_name, argv)
        self.exit_codes = exit_codes or {}
        self.stdout = stdout or {}
        self.write = write

    # -- helpers used by the assertions -------------------------------------
    @property
    def argvs(self):
        return [argv for _stage, argv in self.calls]

    def for_script(self, script_name):
        return [argv for _s, argv in self.calls
                if any(a.endswith(script_name) for a in argv)]

    def ran(self, script_name):
        return bool(self.for_script(script_name))

    # -- the seam -----------------------------------------------------------
    def __call__(self, argv, stage_name):
        self.calls.append((stage_name, list(argv)))
        joined = " ".join(argv)
        rc = 0
        for needle, code in self.exit_codes.items():
            if needle in joined:
                rc = code
                break
        script = os.path.basename(argv[1]) if len(argv) > 1 else ""
        out = self.stdout.get(script, "")
        if not out:
            for needle, text in self.stdout.items():
                if needle in joined:
                    out = text
                    break
        if not out and "-m pytest" in joined:
            out = "42 passed in 3.21s\n"
        if self.write:
            self._materialize(argv, rc)
        return rfp.CommandResult(argv=list(argv), exit_code=rc, stdout=out,
                                 stderr="", duration_seconds=0.01,
                                 log_path=os.path.join(self.env["run_dir"], "fake.log"))

    def _materialize(self, argv, rc):
        """Write what the real script would have written.

        Deliberately independent of ``rc``. A script that writes five valid
        checkpoints and *then* raises is precisely the failure this pipeline
        exists to catch, so the fake reproduces that ordering rather than
        assuming nothing was written whenever the process failed.
        """
        e = self.env
        script = os.path.basename(argv[1]) if len(argv) > 1 else ""
        if script in e.get("do_not_write", ()):
            return
        if script == "fetch_current_telemetry.py":
            write_corpus(e["corpus"], rows_per_station=e.get("fetch_rows", 7))
        elif script == "train_predictive_quality.py":
            write_candidate_artifacts(e["candidate_dir"],
                                      complete=e.get("train_complete", True))
            zero = e.get("zero_byte")
            if zero:
                open(os.path.join(e["candidate_dir"], zero), "w").close()
        elif script == "benchmark_vs_nwp.py":
            write_benchmark_results(_flag(argv, "--out"), label=_flag(argv, "--label"),
                                    nwp_scored=e.get("nwp_scored"),
                                    hollow=e.get("hollow_benchmark", False),
                                    n=e.get("benchmark_n", 3209),
                                    drop_horizon=e.get("drop_benchmark_horizon"))
        elif script == "capture_release_baseline.py":
            # Bind to the corpus as it is NOW, not as it was in setUp: an
            # earlier fetch stage legitimately rewrites it.
            write_baseline_md(_flag(argv, "--out"),
                              rfp.sha256_file(_flag(argv, "--corpus")),
                              rows=e.get("baseline_rows", 5))
        elif script == "generate_bundles.py":
            bd = write_bundles(_flag(argv, "--output-dir"),
                               tag=e.get("bundle_tag", "regenerated"),
                               horizons=e.get("bundle_horizons", HORIZONS))
            tamper = e.get("tamper_bundle_horizon")
            if tamper is not None:
                man = os.path.join(bd, f"h{tamper}", "bundle_manifest.json")
                with open(man, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                doc["checkpoint_sha256"] = "c" * 64
                with open(man, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh, indent=2)
        elif script == "release_gate.py":
            out = _flag(argv, "--out")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(e.get("gate_verdict", {
                    "recommendation": "PROMOTE_ALL",
                    "counts": {"PROMOTE": 4, "REJECT": 0,
                               "INSUFFICIENT_EVIDENCE": 0},
                    "integrity": {"baseline": {"status": "PASS"}},
                }), fh, indent=2)


def _flag(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


# --------------------------------------------------------------------------- #
# Base fixture
# --------------------------------------------------------------------------- #

class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pl_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.data_dir = os.path.join(self.tmp, "data")
        self.run_dir = os.path.join(self.tmp, "run")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

        self.corpus = os.path.join(self.data_dir, "weather_telemetry_current.csv")
        self.candidate_dir = os.path.join(self.data_dir, "candidate_artifacts")
        self.bundles_dir = os.path.join(self.data_dir, "bundles")
        self.benchmark_out = os.path.join(self.run_dir, "nwp_benchmark_results.json")
        self.baseline_out = os.path.join(self.run_dir, "release_baseline.md")
        self.manifest = os.path.join(self.run_dir, "run_manifest.json")

        write_corpus(self.corpus)
        self.corpus_sha = rfp.sha256_file(self.corpus)
        # Production checkpoint source, the way generate_bundles.py works.
        self.production_ckpt = os.path.join(self.data_dir, "lnn_weather_water_h1.pt")
        with open(self.production_ckpt, "wb") as fh:
            fh.write(b"production-weights")
        with open(os.path.join(self.data_dir, "inference_policy.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"policy_version": "1.0.0"}, fh)

        self.env = {
            "run_dir": self.run_dir,
            "data_dir": self.data_dir,
            "corpus": self.corpus,
            "corpus_sha": self.corpus_sha,
            "candidate_dir": self.candidate_dir,
            "bundles_dir": self.bundles_dir,
        }

    def cfg(self, stage="all", **kw):
        params = dict(
            data_dir=self.data_dir,
            run_dir=self.run_dir,
            manifest_path=self.manifest,
            stage=stage,
            python="PYTHON",
            corpus=self.corpus,
            candidate_dir=self.candidate_dir,
            benchmark_out=self.benchmark_out,
            baseline_out=self.baseline_out,
            bundles_dir=self.bundles_dir,
            gate_verdict_out=os.path.join(self.run_dir, "release_gate_verdict.json"),
            pytest_target=os.path.join(self.tmp, "nonexistent_tests"),
        )
        params.update(kw)
        return rfp.PipelineConfig(**params)

    def run_pipeline(self, cfg=None, runner=None, write=True):
        cfg = cfg or self.cfg()
        runner = runner or FakeRunner(self.env, write=write)
        quiet = lambda _m: None  # noqa: E731
        manifest, rc = rfp.run_pipeline(cfg, runner=runner, log=quiet)
        return manifest, rc, runner


# --------------------------------------------------------------------------- #
# 1. Command construction -- the exact argv, per stage
# --------------------------------------------------------------------------- #

class TestCommandConstruction(PipelineTestCase):
    def test_fetch_argv_matches_the_real_flag_names(self):
        _, _, runner = self.run_pipeline(self.cfg(stage="fetch", force=True))
        self.assertEqual(len(runner.argvs), 1)
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "fetch_current_telemetry.py"),
            "--start", "2026-06-20",
            "--end", rfp.datetime.now(rfp.timezone.utc).strftime("%Y-%m-%d"),
            "--out", self.corpus,
            "--take", "5000",
            "--delay", "3.5",
        ])

    def test_fetch_passes_stations_and_filter_only_when_asked(self):
        _, _, runner = self.run_pipeline(self.cfg(
            stage="fetch", force=True, fetch_stations="AAA,BBB",
            fetch_filter_outliers=True, fetch_take=999, fetch_delay=0.0,
            fetch_start="2026-01-01", fetch_end="2026-02-02"))
        argv = runner.argvs[0]
        self.assertEqual(_flag(argv, "--stations"), "AAA,BBB")
        self.assertIn("--filter-outliers", argv)
        self.assertEqual(_flag(argv, "--start"), "2026-01-01")
        self.assertEqual(_flag(argv, "--end"), "2026-02-02")
        self.assertEqual(_flag(argv, "--take"), "999")
        self.assertEqual(_flag(argv, "--delay"), "0.0")

    def test_train_argv_uses_weather_csv_and_horizons(self):
        _, _, runner = self.run_pipeline(self.cfg(stage="train"))
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "train_predictive_quality.py"),
            "--weather-csv", self.corpus,
            "--output-dir", self.candidate_dir,
            "--horizons", "1,3,6,12,24",
            "--epochs", "60",
            "--patience", "12",
            "--lr", "0.001",
        ])

    def test_train_passes_seed_and_commit_only_when_given(self):
        _, _, runner = self.run_pipeline(self.cfg(
            stage="train", seed=7, train_commit="abc1234"))
        argv = runner.argvs[0]
        self.assertEqual(_flag(argv, "--seed"), "7")
        self.assertEqual(_flag(argv, "--commit"), "abc1234")

    def test_benchmark_argv_uses_candidate_dir_and_out(self):
        write_candidate_artifacts(self.candidate_dir)
        _, _, runner = self.run_pipeline(self.cfg(stage="benchmark"))
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "benchmark_vs_nwp.py"),
            "--weather-csv", self.corpus,
            "--candidate-dir", self.candidate_dir,
            "--horizons", "1,3,6,12,24",
            "--out", self.benchmark_out,
            "--label", LABEL,
        ])

    def test_benchmark_label_override_is_propagated(self):
        write_candidate_artifacts(self.candidate_dir)
        _, _, runner = self.run_pipeline(
            self.cfg(stage="benchmark", benchmark_label="LNN windfix"))
        self.assertEqual(_flag(runner.argvs[0], "--label"), "LNN windfix")

    def test_baseline_argv_points_at_the_benchmark_json_we_produced(self):
        write_benchmark_results(self.benchmark_out)
        _, _, runner = self.run_pipeline(self.cfg(stage="baseline"))
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "capture_release_baseline.py"),
            "--benchmark", self.benchmark_out,
            "--corpus", self.corpus,
            "--bundle-dir", self.bundles_dir,
            "--out", self.baseline_out,
        ])

    def test_bundle_argv_and_the_check_only_second_pass(self):
        _, _, runner = self.run_pipeline(
            self.cfg(stage="bundle", force=True))
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "verify_provenance.py"),
        ])
        self.assertEqual(runner.argvs[1], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "generate_bundles.py"),
            "--data-dir", self.data_dir,
            "--output-dir", self.bundles_dir,
        ])
        self.assertEqual(runner.argvs[2], runner.argvs[1] + ["--check-only"])

    def test_bundle_passes_allow_commit_through_to_the_provenance_gate(self):
        _, _, runner = self.run_pipeline(
            self.cfg(stage="bundle", force=True, allow_commit="deadbee"))
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "verify_provenance.py"),
            "--allow-commit", "deadbee",
        ])

    def test_verify_argv_runs_provenance_then_the_provenance_test_suite(self):
        _, _, runner = self.run_pipeline(self.cfg(stage="verify"))
        self.assertEqual(runner.argvs[0], [
            "PYTHON", os.path.join(rfp.SRC_DIR, "verify_provenance.py")])
        self.assertEqual(runner.argvs[1], [
            "PYTHON", "-m", "pytest",
            os.path.join(rfp.SRC_DIR, "test_provenance.py"), "-q"])

    def test_test_stage_argv(self):
        cfg = self.cfg(stage="test", pytest_args=["-q", "--tb=short"])
        _, _, runner = self.run_pipeline(cfg)
        self.assertEqual(runner.argvs[0], [
            "PYTHON", "-m", "pytest", cfg.pytest_target, "-q", "--tb=short"])

    def test_no_flag_is_passed_that_the_scripts_do_not_accept(self):
        """Guard against the classic failure: a flag the script rejects.

        Every flag the pipeline can emit is checked against the parser of the
        script it targets, so adding a flag upstream without wiring it here (or
        vice versa) fails a test rather than a 40-minute run.
        """
        allowed = {
            "fetch_current_telemetry.py": {
                "--start", "--end", "--out", "--base", "--interval", "--take",
                "--delay", "--stations", "--filter-outliers"},
            "train_predictive_quality.py": {
                "--epochs", "--patience", "--horizons", "--lr", "--seed",
                "--output-dir", "--weather-csv", "--commit"},
            "benchmark_vs_nwp.py": {
                "--weather-csv", "--candidate-dir", "--horizons", "--out", "--label"},
            "capture_release_baseline.py": {
                "--benchmark", "--corpus", "--bundle-dir", "--out", "--check"},
            "generate_bundles.py": {
                "--data-dir", "--output-dir", "--check-only",
                "--implementation-commit", "--artifact-commit",
                "--model-weights-commit"},
            "verify_provenance.py": {"--allow-commit"},
        }
        _, _, runner = self.run_pipeline(
            self.cfg(stage="all", force=True, seed=1, allow_commit="abc",
                     fetch_stations="A,B"))
        for _stage, argv in runner.calls:
            script = os.path.basename(argv[1]) if len(argv) > 1 else ""
            if script not in allowed:
                continue
            i = 2
            while i < len(argv):
                tok = argv[i]
                if tok.startswith("--"):
                    self.assertIn(
                        tok, allowed[script],
                        f"{script} does not accept {tok} (from argv {argv})")
                    i += 2
                else:
                    i += 1


# --------------------------------------------------------------------------- #
# 2. Fail-fast: a failing stage stops the pipeline
# --------------------------------------------------------------------------- #

class TestFailFast(PipelineTestCase):
    def test_nonzero_train_exit_halts_every_later_stage(self):
        runner = FakeRunner(self.env, exit_codes={
            "train_predictive_quality.py": 1})
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="all"), runner=runner)
        self.assertEqual(rc, 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(runner.ran("benchmark_vs_nwp.py"))
        self.assertFalse(runner.ran("generate_bundles.py"))
        self.assertFalse(runner.ran("verify_provenance.py"))
        self.assertEqual(manifest["summary"]["benchmark"], rfp.STATUS_HALTED)
        self.assertEqual(manifest["summary"]["test"], rfp.STATUS_HALTED)

    def test_a_halted_stage_is_marked_halted_not_skipped(self):
        """A halted stage is UNKNOWN, not a clean skip and not a pass."""
        runner = FakeRunner(self.env, exit_codes={"train_predictive_quality.py": 1})
        manifest, _rc, _ = self.run_pipeline(self.cfg(stage="all"), runner=runner)
        verify = [s for s in manifest["stages"] if s["name"] == "verify"][0]
        self.assertEqual(verify["status"], rfp.STATUS_HALTED)
        self.assertIn("did not run", " ".join(verify["notes"]))

    def test_keep_going_runs_later_stages_but_still_exits_nonzero(self):
        runner = FakeRunner(self.env, exit_codes={"train_predictive_quality.py": 1})
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="all", keep_going=True), runner=runner)
        self.assertEqual(rc, 1)
        self.assertTrue(runner.ran("benchmark_vs_nwp.py"))
        self.assertEqual(manifest["status"], "failed")

    def test_provenance_failure_blocks_packaging_even_standalone(self):
        """--stage bundle alone must still refuse to package."""
        write_bundles(self.bundles_dir)
        runner = FakeRunner(self.env, exit_codes={"verify_provenance.py": 1})
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle", force=True), runner=runner)
        self.assertEqual(rc, 1)
        self.assertFalse(runner.ran("generate_bundles.py"),
                         "generate_bundles.py ran despite a failed provenance gate")
        self.assertEqual(manifest["summary"]["bundle"], rfp.STATUS_BLOCKED)

    def test_missing_training_corpus_stops_before_launching_anything(self):
        os.remove(self.corpus)
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="train"), runner=runner)
        self.assertEqual(rc, 1)
        self.assertEqual(runner.calls, [])
        self.assertEqual(manifest["summary"]["train"], rfp.STATUS_FAILED)


# --------------------------------------------------------------------------- #
# 3. Exit 0 with hollow output is a FAILURE
# --------------------------------------------------------------------------- #

class TestHollowOutputIsFailure(PipelineTestCase):
    def test_benchmark_where_every_nwp_model_is_na_fails_despite_exit_0(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["hollow_benchmark"] = True
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="benchmark"), runner=runner)
        # The command really did exit 0 -- that is the whole point of the test.
        exit_codes = [c["exit_code"] for s in manifest["stages"]
                      for c in s["commands"]]
        self.assertEqual(exit_codes, [0])
        self.assertEqual(rc, 1, "an all-n/a benchmark was accepted as a pass")
        self.assertEqual(manifest["summary"]["benchmark"], rfp.STATUS_FAILED)
        names = [c["name"] for s in manifest["stages"] for c in s["checks"]]
        self.assertIn("nwp_models_scored", names)

    def test_hollow_detection_message_names_the_silent_n_a_run(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["hollow_benchmark"] = True
        manifest, _rc, _ = self.run_pipeline(self.cfg(stage="benchmark"))
        detail = " ".join(c["detail"] for s in manifest["stages"]
                          for c in s["checks"] if c["name"] == "nwp_models_scored")
        self.assertIn("NWP models produced a score", detail)
        self.assertIn("0 of 7", detail)

    def test_partial_nwp_coverage_below_the_minimum_fails(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["nwp_scored"] = ["ECMWF IFS", "NOAA GFS"]
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="benchmark", min_nwp_scored=3))
        self.assertEqual(rc, 1)
        self.assertEqual(manifest["summary"]["benchmark"], rfp.STATUS_FAILED)

    def test_partial_nwp_coverage_at_the_minimum_passes(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["nwp_scored"] = ["ECMWF IFS", "NOAA GFS"]
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="benchmark", min_nwp_scored=2))
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["summary"]["benchmark"], rfp.STATUS_PASSED)

    def test_require_full_nwp_demands_all_seven(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["nwp_scored"] = NWP[:6]
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="benchmark", require_full_nwp=True))
        self.assertEqual(rc, 1)
        self.assertEqual(manifest["summary"]["benchmark"], rfp.STATUS_FAILED)

    def test_benchmark_with_zero_test_windows_fails(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["hollow_benchmark"] = True
        self.env["benchmark_n"] = 0
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="benchmark"), runner=runner)
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for s in manifest["stages"]
                  for c in s["checks"]}
        self.assertFalse(checks["model_under_test_scored"])

    def test_baseline_with_a_wellformed_but_empty_scoreboard_fails(self):
        """capture_release_baseline.py exits 0 on an empty scoreboard."""
        write_benchmark_results(self.benchmark_out)
        self.env["baseline_rows"] = 0
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="baseline"))
        exit_codes = [c["exit_code"] for s in manifest["stages"]
                      for c in s["commands"]]
        self.assertEqual(exit_codes, [0], "the script exited 0; only the content is wrong")
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for s in manifest["stages"]
                  for c in s["checks"]}
        self.assertFalse(checks["baseline_rows"])

    def test_baseline_bound_to_a_different_corpus_fails(self):
        """A baseline captured against another corpus is a hollow success."""
        write_benchmark_results(self.benchmark_out)

        class WrongCorpus(FakeRunner):
            def _materialize(self, argv, rc):
                if argv[1].endswith("capture_release_baseline.py"):
                    write_baseline_md(_flag(argv, "--out"), "0" * 64, rows=5)
                    return
                super()._materialize(argv, rc)

        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="baseline"), runner=WrongCorpus(self.env))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for s in manifest["stages"]
                  for c in s["checks"]}
        self.assertFalse(checks["baseline_corpus_binding"])

    def test_train_exit_1_with_complete_artifacts_is_reported_as_a_discrepancy(self):
        """The historic promotion-audit bug: rc=1, five valid checkpoints."""
        runner = FakeRunner(self.env, exit_codes={
            "train_predictive_quality.py": 1})
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="train"), runner=runner)
        self.assertEqual(rc, 1, "rc=1 with complete artifacts was accepted")
        self.assertEqual(manifest["summary"]["train"], rfp.STATUS_FAILED)
        # The artifacts really were there -- the check that must not have passed.
        checks = {c["name"]: c["ok"] for c in
                  manifest["stages"][0]["checks"]}
        self.assertTrue(checks["artifacts"],
                        "the artifact check should pass; only the exit disagrees")
        self.assertFalse(checks["exit_code"])
        self.assertFalse(checks["exit_artifact_discrepancy"])
        detail = " ".join(c["detail"] for c in
                          manifest["stages"][0]["checks"]
                          if c["name"] == "exit_artifact_discrepancy")
        self.assertIn("DISCREPANCY", detail)
        self.assertIn("NOT a success", detail)

    def test_train_exit_0_with_missing_artifacts_is_reported_as_a_discrepancy(self):
        self.env["train_complete"] = False
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="train"))
        self.assertEqual(rc, 1, "exit 0 with an incomplete artifact set was accepted")
        detail = " ".join(c["detail"] for c in
                          manifest["stages"][0]["checks"]
                          if c["name"] == "exit_artifact_discrepancy")
        self.assertIn("DISCREPANCY", detail)

    def test_pytest_exit_0_with_failures_in_the_summary_fails(self):
        runner = FakeRunner(self.env, stdout={
            "-m pytest": "F\n=== 1 failed, 41 passed in 12.30s ===\n"})
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="test"), runner=runner)
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertTrue(checks["exit_code"])
        self.assertFalse(checks["test_results"])

    def test_a_clean_pytest_summary_passes(self):
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="test"))
        self.assertEqual(rc, 0)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertTrue(checks["exit_code"])
        self.assertTrue(checks["test_results"])
        self.assertEqual(manifest["summary"]["test"], rfp.STATUS_PASSED)

    def test_gate_exit_0_without_a_verdict_document_fails(self):
        gate_inputs = os.path.join(self.tmp, "gate_inputs.json")
        with open(gate_inputs, "w", encoding="utf-8") as fh:
            json.dump({"challenger": {}, "incumbent": {}}, fh)

        class NoVerdict(FakeRunner):
            def _materialize(self, argv, rc):
                return  # deliberately write nothing

        cfg = self.cfg(stage="gate", gate_inputs=gate_inputs)
        manifest, rc, _ = self.run_pipeline(cfg, runner=NoVerdict(self.env))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertTrue(checks["exit_code"])
        self.assertFalse(checks["gate_verdict_written"])


# --------------------------------------------------------------------------- #
# 4. --dry-run launches nothing
# --------------------------------------------------------------------------- #

class TestDryRun(PipelineTestCase):
    def test_dry_run_launches_no_subprocess(self):
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="all", dry_run=True, force=True), runner=runner)
        self.assertEqual(runner.calls, [],
                         "--dry-run reached the subprocess layer")
        self.assertEqual(rc, 0)

    def test_dry_run_still_plans_every_command(self):
        runner = FakeRunner(self.env)
        manifest, _rc, _ = self.run_pipeline(
            self.cfg(stage="all", dry_run=True, force=True), runner=runner)
        planned = [c for s in manifest["stages"] for c in s["commands"]]
        self.assertTrue(planned, "a dry run must still show the plan")
        self.assertTrue(all(c["launched"] is False for c in planned))
        scripts = {os.path.basename(c["argv"][1]) for c in planned
                   if len(c["argv"]) > 1}
        for expected in ("fetch_current_telemetry.py", "train_predictive_quality.py",
                         "benchmark_vs_nwp.py", "capture_release_baseline.py",
                         "generate_bundles.py", "verify_provenance.py"):
            self.assertIn(expected, scripts)

    def test_dry_run_writes_no_artifacts(self):
        runner = FakeRunner(self.env)
        self.run_pipeline(self.cfg(stage="all", dry_run=True, force=True),
                          runner=runner)
        self.assertFalse(os.path.exists(self.benchmark_out))
        self.assertFalse(os.path.isdir(self.candidate_dir))
        self.assertFalse(os.path.isdir(self.bundles_dir))

    def test_dry_run_records_every_stage_and_reports_no_failure(self):
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="all", dry_run=True, force=True))
        self.assertEqual(rc, 0)
        self.assertEqual(set(manifest["summary"]),
                         set(rfp.STAGE_ORDER))
        for s in manifest["stages"]:
            self.assertNotIn(rfp.STATUS_FAILED, [s["status"]])
            for c in s["checks"]:
                self.assertFalse(
                    c["severity"] == rfp.SEV_ERROR and not c["ok"],
                    f"{s['name']}/{c['name']} reported a hard failure in a dry run")

    def test_dry_run_does_not_mutate_a_real_data_directory(self):
        """A dry run against the live repo must touch nothing."""
        before = sorted(os.listdir(self.data_dir))
        self.run_pipeline(self.cfg(stage="all", dry_run=True))
        self.assertEqual(sorted(os.listdir(self.data_dir)), before)

    def test_a_single_stage_dry_run_also_launches_nothing(self):
        for stage in rfp.STAGE_ORDER:
            runner = FakeRunner(self.env)
            self.run_pipeline(self.cfg(stage=stage, dry_run=True, force=True),
                              runner=runner)
            self.assertEqual(runner.calls, [], f"--stage {stage} --dry-run ran")

    def test_single_stage_runs_only_that_stage(self):
        for stage in rfp.STAGE_ORDER:
            runner = FakeRunner(self.env)
            manifest, _rc, _ = self.run_pipeline(
                self.cfg(stage=stage, force=True), runner=runner)
            self.assertEqual([s["name"] for s in manifest["stages"]], [stage])

    def test_dry_run_manifest_is_written(self):
        self.run_pipeline(self.cfg(stage="all", dry_run=True, force=True))
        self.assertTrue(os.path.isfile(self.manifest))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertTrue(doc["invocation"]["dry_run"])
        planned = [c for s in doc["stages"] for c in s["commands"]]
        self.assertTrue(planned)
        self.assertTrue(all(c["launched"] is False for c in planned))

    def test_dry_run_lists_the_artifacts_each_stage_would_write(self):
        manifest, _rc, _ = self.run_pipeline(
            self.cfg(stage="all", dry_run=True, force=True))
        by_name = {s["name"]: s for s in manifest["stages"]}
        planned = by_name["train"]["artifacts_produced"]["planned"]
        self.assertTrue(any(p.endswith("candidate_h6h.pt") for p in planned))
        self.assertTrue(any(p.endswith("champion_challenger_report.json")
                            for p in planned))
        bundle_planned = by_name["bundle"]["artifacts_produced"]["planned"]
        self.assertEqual(len(bundle_planned), 5 * len(rfp.BUNDLE_FILES))
        self.assertTrue(any(p.endswith("h24/checkpoint.pt")
                            for p in bundle_planned))


# --------------------------------------------------------------------------- #
# 5. Missing artifacts are detected
# --------------------------------------------------------------------------- #

class TestMissingArtifacts(PipelineTestCase):
    def test_missing_candidate_checkpoint_fails_the_train_stage(self):
        self.env["do_not_write"] = {"train_predictive_quality.py"}
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="train"))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["artifacts"])

    def test_zero_byte_candidate_checkpoint_fails(self):
        self.env["zero_byte"] = "candidate_h12h.pt"
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="train"))
        self.assertEqual(rc, 1)
        detail = " ".join(c["detail"] for c in manifest["stages"][0]["checks"]
                          if c["name"] == "artifacts")
        self.assertIn("0 bytes", detail)

    def test_absent_benchmark_json_fails_the_benchmark_stage(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["do_not_write"] = {"benchmark_vs_nwp.py"}
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="benchmark"))
        self.assertEqual(rc, 1)
        detail = " ".join(c["detail"] for c in manifest["stages"][0]["checks"])
        self.assertIn("was not written", detail)

    def test_empty_benchmark_json_fails(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["do_not_write"] = {"benchmark_vs_nwp.py"}
        os.makedirs(self.run_dir, exist_ok=True)
        open(self.benchmark_out, "w").close()
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="benchmark"))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["results_file"])

    def test_unparseable_benchmark_json_fails(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["do_not_write"] = {"benchmark_vs_nwp.py"}
        with open(self.benchmark_out, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="benchmark"))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["results_parse"])

    def test_benchmark_with_zero_test_windows_fails(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["benchmark_n"] = 0
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="benchmark"))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["model_under_test_scored"])

    def test_benchmark_missing_a_requested_horizon_fails(self):
        write_candidate_artifacts(self.candidate_dir)
        self.env["drop_benchmark_horizon"] = 12
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="benchmark"))
        self.assertEqual(rc, 1)
        detail = " ".join(c["detail"] for c in manifest["stages"][0]["checks"]
                          if c["name"] == "horizons_present")
        self.assertIn("h12", detail)

    def test_incomplete_bundle_set_fails(self):
        self.env["bundle_horizons"] = [1, 3, 6, 12]   # +24h never written
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle", force=True))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["bundle_files_present"])

    def test_bundle_manifest_checkpoint_hash_mismatch_fails(self):
        self.env["tamper_bundle_horizon"] = 6
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle", force=True))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["bundle_checkpoint_hashes"])

    def test_absent_provenance_test_suite_fails_the_verify_stage(self):
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="verify", provenance_test="test_does_not_exist.py"))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["provenance_test_present"])

    def test_provenance_test_suite_reporting_failures_fails_the_verify_stage(self):
        runner = FakeRunner(self.env, stdout={
            "test_provenance.py": "F\n=== 2 failed, 18 passed in 4.10s ===\n"})
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="verify"), runner=runner)
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertTrue(checks["exit_code"])
        self.assertFalse(checks["provenance_tests"])

    def test_bundle_check_only_failure_fails_the_stage(self):
        runner = FakeRunner(self.env, exit_codes={"--check-only": 1})
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle", force=True), runner=runner)
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertTrue(checks["exit_code"])
        self.assertFalse(checks["check_only"])

    def test_a_failed_run_still_writes_a_complete_manifest(self):
        runner = FakeRunner(self.env, exit_codes={"train_predictive_quality.py": 1})
        self.run_pipeline(self.cfg(stage="all", force=True), runner=runner)
        self.assertTrue(os.path.isfile(self.manifest))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["status"], "failed")
        self.assertTrue(doc["finished_at"])
        self.assertEqual(len(doc["stages"]), len(rfp.STAGE_ORDER))
        train = [s for s in doc["stages"] if s["name"] == "train"][0]
        self.assertEqual([c["exit_code"] for c in train["commands"]], [1])


# --------------------------------------------------------------------------- #
# 6. --force is required before overwriting
# --------------------------------------------------------------------------- #

class TestForceGuards(PipelineTestCase):
    def test_existing_bundle_dir_requires_force(self):
        write_bundles(self.bundles_dir)
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle"), runner=runner)
        self.assertEqual(rc, 1)
        self.assertFalse(runner.ran("generate_bundles.py"))
        self.assertEqual(manifest["summary"]["bundle"], rfp.STATUS_BLOCKED)
        detail = " ".join(c["detail"] for c in manifest["stages"][0]["checks"])
        self.assertIn("--force", detail)
        self.assertIn("bundle_dir", detail)

    def test_existing_promoted_policy_requires_force(self):
        """Even with no bundle dir, a promoted inference_policy.json blocks."""
        os.makedirs(self.bundles_dir, exist_ok=True)
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle"), runner=runner)
        self.assertEqual(rc, 1)
        self.assertFalse(runner.ran("generate_bundles.py"))
        self.assertIn("promoted_policy", " ".join(
            c["detail"] for c in manifest["stages"][0]["checks"]))

    def test_nothing_is_deleted_by_the_guard(self):
        write_bundles(self.bundles_dir, tag="old")
        before = sorted(os.listdir(self.bundles_dir))
        self.run_pipeline(self.cfg(stage="bundle"))
        self.assertEqual(sorted(os.listdir(self.bundles_dir)), before)
        self.assertTrue(os.path.isfile(
            os.path.join(self.bundles_dir, "h24", "checkpoint.pt")))
        self.assertTrue(os.path.isfile(
            os.path.join(self.bundles_dir, "h1", "inference_policy.json")))

    def test_existing_corpus_requires_force_to_overwrite(self):
        before = rfp.sha256_file(self.corpus)
        runner = FakeRunner(self.env)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="fetch"), runner=runner)
        self.assertEqual(rc, 1)
        self.assertEqual(runner.calls, [])
        self.assertEqual(rfp.sha256_file(self.corpus), before)
        self.assertEqual(manifest["summary"]["fetch"], rfp.STATUS_FAILED)

    def test_force_allows_overwriting_and_the_old_artifacts_are_replaced(self):
        write_bundles(self.bundles_dir, tag="old")
        before = rfp.sha256_file(
            os.path.join(self.bundles_dir, "h1", "checkpoint.pt"))
        self.env["bundle_tag"] = "new"
        manifest, rc, runner = self.run_pipeline(self.cfg(stage="bundle", force=True))
        self.assertEqual(rc, 0)
        self.assertTrue(runner.ran("generate_bundles.py"))
        after = rfp.sha256_file(
            os.path.join(self.bundles_dir, "h1", "checkpoint.pt"))
        self.assertNotEqual(before, after,
                            "generate_bundles.py should have rewritten the bundle")
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertTrue(checks["overwrite_guard"])

    def test_force_lets_the_corpus_be_refetched(self):
        before = rfp.sha256_file(self.corpus)
        self.env["fetch_rows"] = 9
        manifest, rc, runner = self.run_pipeline(self.cfg(stage="fetch", force=True))
        self.assertEqual(rc, 0)
        self.assertTrue(runner.ran("fetch_current_telemetry.py"))
        self.assertNotEqual(rfp.sha256_file(self.corpus), before)

    def test_low_station_coverage_fails_even_with_exit_0(self):
        class OneStation(FakeRunner):
            def _materialize(self, argv, rc):
                if argv[1].endswith("fetch_current_telemetry.py"):
                    write_corpus(self.env["corpus"], stations=1, rows_per_station=7)
                    return
                super()._materialize(argv, rc)

        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="fetch", force=True), runner=OneStation(self.env))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["corpus_station_coverage"])


# --------------------------------------------------------------------------- #
# 7. The release gate: skipped loudly, or run and its verdict checked
# --------------------------------------------------------------------------- #

class TestReleaseGate(PipelineTestCase):
    def _gate_inputs(self):
        p = os.path.join(self.tmp, "gate_inputs.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"challenger": {}, "incumbent": {}}, fh)
        return p

    def test_gate_without_evidence_is_skipped_not_passed(self):
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="gate"))
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["summary"]["gate"], rfp.STATUS_SKIPPED)
        self.assertEqual(manifest["status"], "gate_skipped")
        notes = " ".join(manifest["notes"])
        self.assertIn("SKIPPED, not passed", notes)
        self.assertIn("will not fabricate", notes)

    def test_require_gate_turns_the_skip_into_a_failure(self):
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="gate", require_gate=True))
        self.assertEqual(rc, 1)
        self.assertEqual(manifest["status"], "failed")

    def test_gate_runs_when_evidence_is_supplied(self):
        inputs = self._gate_inputs()
        cfg = self.cfg(stage="gate", gate_inputs=inputs)
        manifest, rc, runner = self.run_pipeline(cfg)
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["summary"]["gate"], rfp.STATUS_PASSED)
        argv = runner.argvs[0]
        self.assertEqual(os.path.basename(argv[1]), "release_gate.py")
        self.assertEqual(_flag(argv, "--inputs"), inputs)
        self.assertTrue(_flag(argv, "--out").endswith("release_gate_verdict.json"))

    def test_gate_retain_verdict_is_not_a_pipeline_pass(self):
        inputs = self._gate_inputs()
        self.env["gate_verdict"] = {
            "recommendation": "RETAIN_ALL",
            "counts": {"PROMOTE": 0, "REJECT": 4, "INSUFFICIENT_EVIDENCE": 0},
            "integrity": {"baseline": {"status": "PASS"}}}
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="gate", gate_inputs=inputs))
        self.assertEqual(rc, 1, "a gate that promoted nothing was reported as a pass")
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["gate_recommendation"])

    def test_gate_blocked_verdict_is_not_a_pipeline_pass(self):
        inputs = self._gate_inputs()
        self.env["gate_verdict"] = {
            "recommendation": "BLOCKED",
            "counts": {"PROMOTE": 0, "REJECT": 0,
                       "INSUFFICIENT_EVIDENCE": 4},
            "integrity": {"baseline": {"status": "BLOCK",
                                       "reason": "baseline drifted"}}}
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="gate", gate_inputs=inputs))
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["gate_recommendation"])
        self.assertFalse(checks["gate_integrity"])

    def test_gate_nonzero_exit_fails_the_stage(self):
        inputs = self._gate_inputs()
        runner = FakeRunner(self.env, exit_codes={"release_gate.py": 2})
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="gate", gate_inputs=inputs), runner=runner)
        self.assertEqual(rc, 1)
        checks = {c["name"]: c["ok"] for c in manifest["stages"][0]["checks"]}
        self.assertFalse(checks["exit_code"])

    def test_missing_evidence_file_fails_when_explicitly_named(self):
        cfg = self.cfg(stage="gate", gate_inputs=os.path.join(self.tmp, "nope.json"))
        manifest, rc, runner = self.run_pipeline(cfg)
        self.assertEqual(rc, 1)
        self.assertFalse(runner.ran("release_gate.py"))
        self.assertEqual(manifest["summary"]["gate"], rfp.STATUS_FAILED)


# --------------------------------------------------------------------------- #
# 8. The manifest is the reproducibility record
# --------------------------------------------------------------------------- #

class TestManifest(PipelineTestCase):
    def test_manifest_is_written_to_disk_and_parses(self):
        self.run_pipeline(self.cfg(stage="all", force=True))
        self.assertTrue(os.path.isfile(self.manifest))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["manifest_version"], rfp.MANIFEST_VERSION)
        self.assertEqual(doc["invocation"]["stage"], "all")
        self.assertTrue(doc["invocation"]["force"])
        self.assertTrue(doc["started_at"] and doc["finished_at"])
        self.assertIsInstance(doc["duration_seconds"], float)

    def test_every_stage_records_command_exit_code_and_duration(self):
        self.run_pipeline(self.cfg(stage="all", force=True))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual([s["name"] for s in doc["stages"]], rfp.STAGE_ORDER)
        for s in doc["stages"]:
            self.assertIn("duration_seconds", s)
            self.assertIn("checks", s)
            for c in s["commands"]:
                self.assertIn("argv", c)
                self.assertIsInstance(c["exit_code"], int)
                self.assertIn("duration_seconds", c)
                self.assertIn("command_line", c)

    def test_consumed_artifacts_are_hashed(self):
        self.run_pipeline(self.cfg(stage="all", force=True))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        by_name = {s["name"]: s for s in doc["stages"]}
        # The corpus the later stages consumed is the one fetch produced, not
        # the one setUp wrote; the manifest must record what was actually read.
        current_sha = rfp.sha256_file(self.corpus)
        train = by_name["train"]["artifacts_consumed"]
        self.assertIn("corpus", train)
        self.assertEqual(train["corpus"]["sha256"], current_sha)
        self.assertEqual(train["corpus"]["size_bytes"],
                         os.path.getsize(self.corpus))
        self.assertEqual(by_name["fetch"]["artifacts_produced"]["corpus"]["sha256"],
                         current_sha)
        bench = by_name["benchmark"]["artifacts_consumed"]
        for h in HORIZONS:
            self.assertIn(f"candidate_h{h}h.pt", bench)
            self.assertEqual(len(bench[f"candidate_h{h}h.pt"]["sha256"]), 64)

    def test_produced_artifacts_are_hashed(self):
        self.run_pipeline(self.cfg(stage="all", force=True))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        by_name = {s["name"]: s for s in doc["stages"]}
        self.assertEqual(by_name["train"]["artifacts_produced"]["candidate_h6h.pt"]
                         ["sha256"],
                         rfp.sha256_file(os.path.join(
                             self.candidate_dir, "candidate_h6h.pt")))
        produced = by_name["bundle"]["artifacts_produced"]["bundles_dir"]
        self.assertEqual(produced["type"], "dir")
        self.assertIn("h24/checkpoint.pt", produced["files"])
        self.assertEqual(len(produced["files"]["h24/checkpoint.pt"]["sha256"]), 64)

    def test_manifest_records_the_pipeline_status_and_summary(self):
        self.run_pipeline(self.cfg(stage="all", force=True))
        with open(self.manifest, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertIn(doc["status"], (rfp.STATUS_PASSED, "gate_skipped"))
        self.assertEqual(set(doc["summary"]), set(rfp.STAGE_ORDER))
        self.assertEqual(doc["summary"]["fetch"], rfp.STATUS_PASSED)


# --------------------------------------------------------------------------- #
# 9. End-to-end, and unit tests for the parsers
# --------------------------------------------------------------------------- #

class TestEndToEnd(PipelineTestCase):
    def test_a_healthy_run_passes_everything_except_the_gate(self):
        manifest, rc, _ = self.run_pipeline(self.cfg(stage="all", force=True))
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["status"], "gate_skipped")
        for name in ("fetch", "train", "benchmark", "baseline", "bundle", "verify"):
            self.assertEqual(manifest["summary"][name], rfp.STATUS_PASSED)
        self.assertEqual(manifest["summary"]["gate"], rfp.STATUS_SKIPPED)

    def test_the_candidate_is_never_silently_claimed_as_shipped(self):
        """generate_bundles.py packages production weights, not the candidate."""
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="bundle", force=True))
        self.assertEqual(rc, 0)
        checks = {c["name"]: c for c in manifest["stages"][0]["checks"]}
        warn = checks["bundles_package_the_trained_candidate"]
        self.assertFalse(warn["ok"])
        self.assertEqual(warn["severity"], rfp.SEV_WARN)
        self.assertIn("PRODUCTION", warn["detail"])

    def test_a_full_green_run_when_the_gate_is_satisfied(self):
        inputs = os.path.join(self.tmp, "gate_inputs.json")
        with open(inputs, "w", encoding="utf-8") as fh:
            json.dump({"challenger": {}, "incumbent": {}}, fh)
        manifest, rc, _ = self.run_pipeline(
            self.cfg(stage="all", force=True, gate_inputs=inputs))
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["status"], rfp.STATUS_PASSED)
        for name in rfp.STAGE_ORDER:
            self.assertEqual(manifest["summary"][name], rfp.STATUS_PASSED, name)


class TestParsers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pl_parse_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_pytest_summary_parsing(self):
        cases = [
            ("....\n12 passed in 1.2s\n", {"passed": 12}, []),
            ("\n=== 1 failed, 41 passed in 12.30s ===\n",
             {"failed": 1, "passed": 41}, ["1 test(s) failed"]),
            ("!!! Interrupted: 2 errors during collection !!!\n",
             {"error": 2}, ["2 test error(s)"]),
            ("no tests ran in 0.01s\n", {}, [
                "no pytest result counts in summary line: 'no tests ran in 0.01s'",
                "pytest collected no tests"]),
        ]
        for text, counts, problems in cases:
            summary, got_counts, got_problems = rfp.parse_pytest_summary(text)
            self.assertEqual(got_counts, counts, summary)
            self.assertEqual(got_problems, problems, summary)

    def test_pytest_summary_tolerates_empty_output(self):
        summary, counts, problems = rfp.parse_pytest_summary("")
        self.assertEqual(summary, "")
        self.assertEqual(counts, {})
        self.assertEqual(len(problems), 1)

    def test_overwrite_conflicts_classifies_kinds(self):
        cfg = rfp.PipelineConfig(
            data_dir=tempfile.mkdtemp(prefix="pl_oc_"),
            run_dir=tempfile.mkdtemp(prefix="pl_ocr_"))
        cfg.bundles_dir = os.path.join(cfg.data_dir, "bundles")
        try:
            self.assertEqual(rfp.overwrite_conflicts(cfg), [])
            write_bundles(cfg.bundles_dir, horizons=[1])
            with open(os.path.join(cfg.data_dir, "inference_policy.json"), "w",
                      encoding="utf-8") as fh:
                fh.write("{}")
            kinds = {c["kind"] for c in rfp.overwrite_conflicts(cfg)}
            self.assertEqual(kinds, {"bundle_dir", "promoted_policy"})
            paths = {c["path"] for c in rfp.overwrite_conflicts(cfg)}
            self.assertIn(os.path.join(cfg.bundles_dir, "h1"), paths)
            self.assertIn(os.path.join(cfg.data_dir, "inference_policy.json"), paths)
        finally:
            shutil.rmtree(cfg.data_dir, ignore_errors=True)
            shutil.rmtree(cfg.run_dir, ignore_errors=True)

    def test_cli_accepts_every_documented_stage(self):
        for stage in rfp.STAGE_ORDER + ["all"]:
            cfg = rfp.parse_args(["--stage", stage, "--run-dir", self.tmp])
            self.assertEqual(cfg.stage, stage)

    def test_cli_rejects_an_unknown_stage(self):
        with self.assertRaises(SystemExit):
            rfp.parse_args(["--stage", "promote-everything", "--run-dir", self.tmp])

    def test_cli_defaults_point_inside_the_prediction_model_tree(self):
        cfg = rfp.parse_args(["--run-dir", self.tmp])
        self.assertTrue(cfg.data_dir.endswith(os.path.join("prediction-model", "data")))
        self.assertEqual(cfg.bundles_dir,
                         os.path.join(cfg.data_dir, "bundles"))
        self.assertEqual(cfg.candidate_dir,
                         os.path.join(cfg.data_dir, "candidate_artifacts"))
        self.assertTrue(cfg.corpus.endswith("weather_telemetry_current.csv"))
        self.assertTrue(cfg.gate_verdict_out.endswith("release_gate_verdict.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
