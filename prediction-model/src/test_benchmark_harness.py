"""
Regression tests for the benchmark harness -- the three measurement bugs that
produced confidently-wrong numbers while every artifact on disk looked perfect.

WHAT WENT WRONG
---------------

BUG A -- NWP COVERAGE SILENTLY EMPTY  (benchmark_vs_nwp.load_nwp)
    The cache is keyed ``station|model`` but the same series is present once per
    FETCHED WINDOW, and the loader used to keep only the LONGEST window per
    station|model. The longest cached series ended 2026-08-28; the test split
    ran from 2026-09-11 onward. Longest-wins therefore selected a window that did
    not overlap the evaluation period AT ALL, so every NWP lookup missed,
    `metrics()` got an all-NaN series and returned None, and all seven NWP models
    printed "n/a" -- while the code went on printing a rank for them, as if they
    had been scored. A benchmark with zero NWP coverage still looks like a
    benchmark, which is what makes this class of bug so expensive.
    Now merged on timestamp as a pure union, with an explicit precedence:
    windows are applied shortest-first, and a window only fills timestamps that
    are still MISSING, so a narrow recent re-fetch wins wherever it overlaps;
    among equal spans, the window reaching furthest into the present is applied
    first and wins. Both halves of that contract are load-bearing and each is
    pinned separately -- see
    `test_narrow_recent_window_takes_precedence_over_a_longer_stale_one` (span
    ordering) and
    `test_equal_span_tie_is_broken_by_the_latest_window_end` (tie-break).

BUG B -- WRONG FEATURE SPACE FED TO THE MODEL  (benchmark_vs_nwp._score_candidate)
    Candidates are trained on the NORMALISED tensor `res[0]`. The candidate branch
    of `main()` used to pass `X_raw` -- the de-normalised, clipped tensor that
    the PRODUCTION BUNDLE predictor consumes (`predict_from_observed_sequence`).
    That shifted every feature out of distribution: +3h wind MAE came out at
    0.742 instead of the true 0.498. The number was plausible, formatted to three
    decimals, and wrong.

BUG C -- PROMOTION AUDIT SILENTLY SKIPPED  (train_predictive_quality)
    `train_and_evaluate_all_horizons` builds `_target_routes` with
        for _label, _target, _metric, _block in [ (...3-field tuples...) ]
    Three fields unpacked into FOUR names raises
    `ValueError: not enough values to unpack`. Because the audit runs LAST --
    after every checkpoint is written -- each run exited rc=1 while leaving a
    complete, valid-looking artifact set on disk. The promotion verdict was never
    computed and nobody could tell from the filesystem.

The common shape: a failure that produces NO error the operator can see at the
point of use. These tests are deliberately structural where a numeric threshold
would be fragile (BUG C reads the source with `ast`; BUG B asserts the *identity*
of the tensor handed to the scorer), because the numbers move with the corpus
while the shapes of these mistakes do not.
"""

import ast
import builtins
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import benchmark_vs_nwp as bnv  # noqa: E402
from dataset import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    TelemetryDataPipeline,
    build_forecast_windows,
)

TRAIN_SRC = os.path.join(HERE, "train_predictive_quality.py")
CANDIDATE_DIR = os.path.join(bnv.DATA_DIR, "candidate_artifacts")

# The NWP cache fields load_nwp() merges. Kept as a module constant so a rename
# in production code surfaces here as a loud failure rather than a silent one.
NWP_FIELDS = ("temperature_2m", "relative_humidity_2m", "surface_pressure",
              "wind_speed_10m", "precipitation")

# The same clip the production bundle predictor is fed, copied from main().
# If this ever changes, the de-normalised comparison in TestCandidateFeatureSpace
# stops being the real thing and that test must be revisited.
RAW_CLIP_LO = [10, 10, 10, 900, 0, -1, -1, 0]
RAW_CLIP_HI = [50, 70, 100, 1050, 180, 1, 1, 150]

# Real Central Luzon climate normals, close enough to the pipeline's own stats to
# be plausible if -- and only if -- somebody feeds physical units to a model that
# was trained on z-scores.
FAKE_NORM_MEANS = np.array([27.9, 32.2, 86.7, 1005.9, 1.16, 0.096, -0.0099, 0.528])
FAKE_NORM_STDS = np.array([3.16, 7.08, 12.12, 4.47, 3.16, 0.747, 0.658, 2.907])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _hourly(start, end):
    """Inclusive list of ``YYYY-MM-DDTHH:MM`` stamps, as Open-Meteo stores them."""
    cur, out = datetime.fromisoformat(start), []
    stop = datetime.fromisoformat(end)
    while cur <= stop:
        out.append(cur.strftime("%Y-%m-%dT%H:%M"))
        cur += timedelta(hours=1)
    return out


def _cache_entry(station, model, times, value):
    """One fetched window, shaped exactly like a real nwp_benchmark_cache record."""
    rec = {"station_id": station, "lat": 14.4797, "lon": 120.9412,
           "model": model, "time": list(times)}
    for fld in NWP_FIELDS:
        rec[fld] = [value] * len(times)
    return rec


def _write_cache(monkeypatch, tmp_path, records):
    """Point the module-level CACHE constant at a synthetic cache and load it.

    `load_nwp()` reads the module global `CACHE`, not an argument, so this is the
    only seam available -- it is a real seam: the production call path is
    exercised byte-for-byte, only the file on disk is swapped.
    """
    path = os.path.join(str(tmp_path), "synthetic_nwp_cache.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    monkeypatch.setattr(bnv, "CACHE", path)
    return bnv.load_nwp()


def _longest_window_only(records):
    """Reimplementation of the DELETED longest-wins rule, for the regression test.

    Deliberately kept here, in the test, so the historical failure mode stays
    executable and documented. It is not production code and must never be
    copied back into benchmark_vs_nwp.py.
    """
    grouped = {}
    for v in records.values():
        base = "%s|%s" % (v["station_id"], v["model"])
        cur = grouped.get(base)
        if cur is None or len(v["time"]) > len(cur["time"]):
            grouped[base] = v
    return {base: set(v["time"]) for base, v in grouped.items()}


def _station_timestamps(entry):
    return {t for t in entry["idx"]}


# The evaluation window that Bug A missed: the refetched corpus splits test from
# 2026-09-11 onward, while the stale cached series stopped at 2026-08-28.
EVAL_START = "2026-09-11T00:00"
EVAL_END = "2026-09-20T23:00"
STALE_START = "2026-06-01T00:00"
STALE_END = "2026-08-28T23:00"


@pytest.fixture(scope="module")
def default_pipeline():
    """The corpus the shipped candidate checkpoints were actually trained on.

    candidate_artifacts/candidate_h*h_manifest.json records
    weather_telemetry_sha256 = 86ce906453aa..., which is
    data/weather_telemetry.csv. Scoring on any other corpus compares a model to
    a distribution it was never fitted on and muddies the very effect under test.
    """
    csv = os.path.join(bnv.DATA_DIR, "weather_telemetry.csv")
    if not os.path.exists(csv):
        pytest.skip("data/weather_telemetry.csv is not present")
    return TelemetryDataPipeline(weather_csv=csv)


@pytest.fixture(scope="module")
def current_pipeline():
    """The refetched corpus -- the one whose test split starts 2026-09-11."""
    csv = os.path.join(bnv.DATA_DIR, "weather_telemetry_current.csv")
    if not os.path.exists(csv):
        pytest.skip("data/weather_telemetry_current.csv is not present")
    return TelemetryDataPipeline(weather_csv=csv)


# --------------------------------------------------------------------------- #
# BUG A
# --------------------------------------------------------------------------- #

class TestNwpCacheWindowMerge:
    """load_nwp() must UNION every cached window, never discard one."""

    def test_merge_keeps_timestamps_from_every_cached_window(self, monkeypatch, tmp_path):
        """The direct regression: a stale long window plus a recent short one.

        Longest-wins returned the stale window only, so the entire evaluation
        period vanished and all seven NWP models reported "n/a" while still being
        given a rank.
        """
        stale = _hourly(STALE_START, STALE_END)
        recent = _hourly(EVAL_START, EVAL_END)
        assert len(stale) > len(recent), "stale window must be the LONGER one"
        records = {
            "03pqkGAj|ecmwf_ifs025|2026-06-01_2026-08-28":
                _cache_entry("03pqkGAj", "ecmwf_ifs025", stale, 111.0),
            "03pqkGAj|ecmwf_ifs025|2026-09-11_2026-09-20":
                _cache_entry("03pqkGAj", "ecmwf_ifs025", recent, 222.0),
        }
        merged = _write_cache(monkeypatch, tmp_path, records)

        entry = merged["03pqkGAj|ecmwf_ifs025"]
        have = _station_timestamps(entry)
        missing = (set(stale) | set(recent)) - have
        assert not missing, (
            "%d timestamps were dropped by the merge; load_nwp() must be a union "
            "over cached windows, not a winner-takes-all selection" % len(missing)
        )
        assert entry["rec"]["temperature_2m"][entry["idx"][EVAL_START]] == 222.0, (
            "the recent window's own value was overwritten; the short targeted "
            "re-fetch is exactly the data the evaluation period needs"
        )

    def test_longest_window_alone_would_have_missed_the_evaluation_period(
            self, monkeypatch, tmp_path):
        """Documents the incident numerically, and pins the fix against it.

        This is the whole bug in one assertion: the window the OLD rule selected
        contains ZERO evaluation-period timestamps, and the merged series contains
        all of them.
        """
        stale = _hourly(STALE_START, STALE_END)
        recent = _hourly(EVAL_START, EVAL_END)
        records = {
            "S|ecmwf_ifs025|old": _cache_entry("S", "ecmwf_ifs025", stale, 111.0),
            "S|ecmwf_ifs025|new": _cache_entry("S", "ecmwf_ifs025", recent, 222.0),
        }
        merged = _write_cache(monkeypatch, tmp_path, records)

        old = _longest_window_only(records)["S|ecmwf_ifs025"]
        eval_stamps = set(recent)
        old_hits = old & eval_stamps
        new_hits = _station_timestamps(merged["S|ecmwf_ifs025"]) & eval_stamps

        assert not old_hits, (
            "the longest-window rule is supposed to be BROKEN here; if it now "
            "covers the evaluation period this fixture no longer reproduces the "
            "bug and the regression test is toothless"
        )
        assert len(new_hits) == len(eval_stamps), (
            "the merged series must resolve every evaluation-period timestamp; "
            "this is the lookup main() performs at the target timestamp"
        )

    def test_benchmark_style_lookup_resolves_after_merge(self, monkeypatch, tmp_path):
        """Replays main()'s exact lookup: ``target_timestamp[:13] + ':00'``.

        Bug A did not fail at load time. It failed here, silently, N times per
        horizon -- `metrics()` then received an all-NaN vector and returned None,
        so the row printed "n/a" and the ranking code still assigned a position.
        """
        stale = _hourly(STALE_START, STALE_END)
        recent = _hourly(EVAL_START, EVAL_END)
        records = {
            "03pqkGAj|ecmwf_ifs025|a": _cache_entry("03pqkGAj", "ecmwf_ifs025", stale, 111.0),
            "03pqkGAj|ecmwf_ifs025|b": _cache_entry("03pqkGAj", "ecmwf_ifs025", recent, 222.0),
        }
        merged = _write_cache(monkeypatch, tmp_path, records)
        entry = merged["03pqkGAj|ecmwf_ifs025"]

        # Metadata stamps carry seconds and an offset; the harness truncates.
        for target in recent:
            meta_stamp = target + ":00:00+00:00"
            key = meta_stamp[:13] + ":00"
            j = entry["idx"].get(key)
            assert j is not None, "no NWP value for evaluation target %s" % meta_stamp
            assert entry["rec"]["temperature_2m"][j] is not None

    def test_merged_index_and_records_stay_aligned(self, monkeypatch, tmp_path):
        """`idx` and `rec` are built from different passes; they must agree.

        A merge that appends records without re-indexing them produces
        plausible-looking garbage rather than a missing value, which is worse.
        """
        records = {
            "S|ecmwf_ifs025|a": _cache_entry("S", "ecmwf_ifs025",
                                             _hourly(STALE_START, "2026-07-15T23:00"), 111.0),
            "S|ecmwf_ifs025|b": _cache_entry("S", "ecmwf_ifs025",
                                             _hourly("2026-08-01T00:00", STALE_END), 222.0),
            "S|gfs_seamless|a": _cache_entry("S", "gfs_seamless",
                                            _hourly(EVAL_START, EVAL_END), 333.0),
        }
        merged = _write_cache(monkeypatch, tmp_path, records)

        assert set(merged) == {"S|ecmwf_ifs025", "S|gfs_seamless"}, (
            "windows for one model must not be collapsed into another model's key"
        )
        for key, entry in merged.items():
            stamps = sorted(entry["idx"])
            assert stamps == sorted(set(stamps))
            for fld in NWP_FIELDS:
                assert len(entry["rec"][fld]) == len(stamps), (
                    "%s/%s: record length disagrees with the index" % (key, fld)
                )
            for t, i in entry["idx"].items():
                assert stamps[i] == t, "%s: idx mapping is not the sorted order" % key

    def test_equal_span_ties_go_to_the_later_window(self, monkeypatch, tmp_path):
        """Minimal two-window form of the tie-break.

        With span tied, the window reaching furthest into the present is applied
        FIRST and therefore wins the overlap (the write is fill-if-missing, so
        first-applied takes each timestamp). A targeted re-fetch that happens to
        cover the same number of hours must not be discarded in favour of an
        older one.

        The exhaustive multi-window ordering property is asserted in
        `test_equal_span_tie_is_broken_by_the_latest_window_end`; this one is
        the two-window smoke test that keeps a regression local and legible.
        """
        early = _hourly("2026-07-01T00:00", "2026-09-05T12:00")
        late = _hourly("2026-07-01T01:00", "2026-09-05T13:00")
        assert len(early) == len(late)
        records = {
            "S|ecmwf_ifs025|early": _cache_entry("S", "ecmwf_ifs025", early, 11.0),
            "S|ecmwf_ifs025|late": _cache_entry("S", "ecmwf_ifs025", late, 22.0),
        }
        merged = _write_cache(monkeypatch, tmp_path, records)
        entry = merged["S|ecmwf_ifs025"]
        assert entry["rec"]["temperature_2m"][entry["idx"]["2026-09-01T00:00"]] == 22.0

    def test_equal_span_tie_is_broken_by_the_latest_window_end(self, monkeypatch, tmp_path):
        """Among equal-span windows, the LATEST-ending one wins every overlap.

        This is the second half of the merge contract, and it is the half with no
        guard until now. The two halves are coupled: making the write
        fill-only-if-missing promoted "first window applied" from meaningless to
        authoritative, which silently turned the sort order into the whole
        precedence rule. The tie-break had to be re-derived from that moment --
        `end` had to become DESCENDING. Left ASCENDING, an equal-span pair
        applies the OLDEST window first and the oldest then wins every shared
        timestamp, so a re-fetch that superseded an older fetch of the same length
        is discarded exactly where it was meant to be trusted. Nothing raises;
        the merged series just quietly serves stale values.

        Three windows of identical length, each shifted a month later than the
        last, so the pair that matters is ambiguous from span alone. The
        assertion is stated as a property over EVERY timestamp rather than a
        single probe, because one shared timestamp could be satisfied by an
        accidental ordering that a third window then contradicts:

            at each timestamp, the value must come from the window with the
            LATEST end among those containing it.

        The headline check first: at a timestamp shared by all three, the latest
        window must win. If the oldest does, the sort is ascending again.
        """
        # Derive the three windows from one base by whole-day offsets, so the
        # equal-span precondition is structural rather than a coincidence of
        # month lengths (Jul-Sep is 92 days, Aug-Oct 91, Sep-Nov 90).
        base_start = datetime(2026, 6, 1, 0, 0)
        base_end = datetime(2026, 8, 31, 23, 0)
        offsets = {"early": 0, "mid": 31, "late": 62}
        values = {"early": 11.0, "mid": 22.0, "late": 33.0}
        stamps = {
            name: _hourly((base_start + timedelta(days=off)).isoformat(),
                          (base_end + timedelta(days=off)).isoformat())
            for name, off in offsets.items()
        }
        ends = {name: s[-1] for name, s in stamps.items()}

        spans = {len(s) for s in stamps.values()}
        assert len(spans) == 1, (
            "fixture is broken: the three windows must have identical spans, got %s"
            % sorted(spans)
        )
        assert ends["early"] < ends["mid"] < ends["late"], "fixture is not ordered"

        records = {
            "S|ecmwf_ifs025|%s" % name: _cache_entry("S", "ecmwf_ifs025", s, values[name])
            for name, s in stamps.items()
        }
        merged = _write_cache(monkeypatch, tmp_path, records)
        entry = merged["S|ecmwf_ifs025"]
        idx, rec = entry["idx"], entry["rec"]["temperature_2m"]

        # Headline: a timestamp present in all three windows.
        shared = set(stamps["early"]) & set(stamps["mid"]) & set(stamps["late"])
        assert shared, "fixture is broken: the three windows do not all overlap"
        triple = min(shared)
        assert rec[idx[triple]] == values["late"], (
            "at %s, shared by all three equal-span windows, the winner is %s "
            "(value %s). The latest-ending window must win; the oldest winning "
            "means the tie-break sort went back to ascending end."
            % (triple, {v: k for k, v in values.items()}[rec[idx[triple]]],
               rec[idx[triple]])
        )

        # Exhaustive form: latest end among the windows containing the timestamp.
        for t in idx:
            holders = [name for name in stamps if t in set(stamps[name])]
            expected = values[max(holders, key=lambda n: ends[n])]
            assert rec[idx[t]] == expected, (
                "at %s (held by %s) the merged value is %s but the latest-ending "
                "holder is %r at %s" % (t, holders, rec[idx[t]],
                                        max(holders, key=lambda n: ends[n]), expected)
            )

        # And coverage is still intact: the union of all three windows must be
        # present, and the exclusive head of the OLDEST and exclusive tail of the
        # NEWEST must carry their own values. Otherwise the tie-break would have
        # been bought with lost coverage. (The middle window of three
        # equal-length consecutive windows is fully covered by its neighbours, so
        # exclusivity is only meaningful at the two ends.)
        union = set()
        for s in stamps.values():
            union |= set(s)
        assert not (union - set(idx)), (
            "%d timestamps were dropped by the merge" % len(union - set(idx))
        )
        assert rec[idx[stamps["early"][0]]] == values["early"], (
            "the oldest window's exclusive head was lost or overwritten"
        )
        assert rec[idx[stamps["late"][-1]]] == values["late"], (
            "the newest window's exclusive tail was lost or overwritten"
        )

    def test_narrow_recent_window_takes_precedence_over_a_longer_stale_one(
            self, monkeypatch, tmp_path):
        """A short, recent, targeted re-fetch must win where it overlaps.

        The guaranteed behaviour, matching load_nwp()'s docstring: "Shortest
        windows are applied FIRST and longer ones fill only the timestamps still
        missing, so a narrow, recent re-fetch takes precedence where it overlaps."

        This assertion used to be a strict xfail, because the implementation did
        the exact inverse. Entries are sorted shortest-first, and the write was
        unconditional (`row[fld] = arr[i]`), so the LONGEST window was written
        LAST and won every overlap -- Bug A's original longest-wins rule, now
        smeared across a timestamp map where it was no longer visible to review.
        The write is now fill-only-if-missing, so the FIRST window applied takes
        each timestamp, and shortest-first is the correct order. The precedence
        is now permanent, so this is a plain test.
        """
        long_stale = _hourly(STALE_START, "2026-09-05T23:00")
        short_recent = _hourly("2026-09-01T00:00", "2026-09-05T23:00")
        assert len(long_stale) > len(short_recent)
        records = {
            "S|ecmwf_ifs025|long": _cache_entry("S", "ecmwf_ifs025", long_stale, 111.0),
            "S|ecmwf_ifs025|short": _cache_entry("S", "ecmwf_ifs025", short_recent, 222.0),
        }
        merged = _write_cache(monkeypatch, tmp_path, records)
        entry = merged["S|ecmwf_ifs025"]
        got = entry["rec"]["temperature_2m"][entry["idx"]["2026-09-02T06:00"]]
        assert got == 222.0, (
            "the long stale window (%s) overwrote the short recent one (%s) at an "
            "overlapping timestamp" % (got, 222.0)
        )

    def test_metadata_keys_and_empty_entries_are_ignored(self, monkeypatch, tmp_path):
        """`_meta` bookkeeping and window-less records must not become series."""
        records = {
            "_meta": {"start_date": "2026-06-01", "end_date": "2026-09-30",
                      "source": "Open-Meteo Historical Forecast API"},
            "S|ecmwf_ifs025|good": _cache_entry("S", "ecmwf_ifs025",
                                               _hourly(EVAL_START, EVAL_END), 5.0),
            "S|gfs_seamless|empty": {"station_id": "S", "model": "gfs_seamless",
                                     "time": [], "temperature_2m": []},
        }
        merged = _write_cache(monkeypatch, tmp_path, records)
        assert set(merged) == {"S|ecmwf_ifs025"}, (
            "a '_meta' or empty record became a scored series: %s" % sorted(merged)
        )


class TestRealNwpCacheCoverage:
    """End-to-end: the committed cache must actually cover the test split.

    This is the test that would have failed on the day. The committed cache now
    spans 2026-06-20..2026-09-30, so every one of the seven models resolves for
    every test window; before the re-fetch the hit rate was exactly 0.0 and the
    table printed "n/a" seven times while still ranking the local model.
    """

    def test_every_nwp_model_resolves_for_every_test_window(self, current_pipeline):
        if not os.path.exists(bnv.CACHE):
            pytest.skip("nwp_benchmark_cache.json is not present")
        nwp = bnv.load_nwp()
        res = build_forecast_windows(pipeline=current_pipeline, split="test", horizon=6,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        meta = res[6]
        if not meta:
            pytest.skip("test split is empty for this corpus")

        hits = {m: 0 for m in bnv.NWP_DISPLAY}
        for m in meta:
            for mdl in bnv.NWP_DISPLAY:
                entry = nwp.get("%s|%s" % (m["station_id"], mdl))
                if not entry:
                    continue
                if entry["idx"].get(m["target_timestamp"][:13] + ":00") is not None:
                    hits[mdl] += 1

        empty = sorted(m for m, c in hits.items() if c == 0)
        assert not empty, (
            "no cached NWP value resolved for %s across %d test windows; "
            "`metrics()` will return None and the model will print n/a while "
            "still being assigned a rank" % (empty, len(meta))
        )
        for mdl, c in hits.items():
            assert c == len(meta), (
                "%s resolved for only %d of %d test windows -- partial NWP "
                "coverage is reported as a real score and is not the same as none"
                % (mdl, c, len(meta))
            )

    def test_wind_is_converted_to_station_units_not_quoted_as_kmh(self):
        """Guards the 3.6x wind error that made the LNN look 6x better than ECMWF.

        Adjacent to Bug A because it is the same failure shape: a silently wrong
        number printed to three decimals. NWP wind_speed_10m is km/h; station
        telemetry is m/s. Verified: station mean 1.501 m/s vs NWP 11.19 km/h.
        """
        assert bnv.NWP_TO_STATION_UNITS == {"wind_speed": pytest.approx(1.0 / 3.6)}


# --------------------------------------------------------------------------- #
# BUG B
# --------------------------------------------------------------------------- #

class _FakePipeline:
    """Minimal stand-in for TelemetryDataPipeline for driving main()'s candidate path."""

    test_start = datetime(2026, 9, 11, 14, tzinfo=timezone.utc)
    time_range_max = datetime(2026, 9, 12, 0, tzinfo=timezone.utc)
    norm_means = FAKE_NORM_MEANS
    norm_stds = FAKE_NORM_STDS


def _fake_windows(n=4, seq=24, feats=8, seed=0):
    """Build a `build_forecast_windows` return value in normalised space."""
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(0.0, 1.0, (n, seq, feats)), dtype=torch.float32)
    dt = torch.tensor(rng.random((n, seq, 1)), dtype=torch.float32)
    precip = torch.zeros((n, 1), dtype=torch.float32)
    meta = []
    for i in range(n):
        meta.append({
            "station_id": "03pqkGAj",
            "target_timestamp": "2026-09-11T%02d:00:00+00:00" % (14 + i),
            "target_temperature": 27.0 + i, "target_humidity": 80.0 - i,
            "target_pressure": 1005.0 + i, "target_wind_speed": 2.0 + i,
            "origin_temperature": 26.0, "origin_humidity": 82.0,
            "origin_pressure": 1006.0, "origin_wind_speed": 1.8,
            "origin_wind_u": 1.0, "origin_wind_v": 1.4,
            "last_observed_precip": 0.0,
        })
    return (x, dt, None, precip, None, None, meta)


def _run_candidate_main(monkeypatch, tmp_path, res):
    """Drive benchmark_vs_nwp.main() down the --candidate-dir branch.

    Every heavy dependency is replaced; the branch under test -- which tensor is
    handed to _score_candidate -- is the real code. Returns (calls, captured stdout).
    """
    calls = []

    def recorder(candidate_dir, h, telemetry, dt, meta, pipe):
        calls.append({"dir": candidate_dir, "h": h, "telemetry": telemetry,
                      "dt": dt, "meta": meta, "pipe": pipe})
        n = len(meta)
        return ({v: np.full(n, 27.0) for v in bnv.VAR_MAP}, np.zeros(n))

    monkeypatch.setattr(bnv, "TelemetryDataPipeline", lambda weather_csv=None: _FakePipeline())
    monkeypatch.setattr(bnv, "load_nwp", lambda: {})
    monkeypatch.setattr(bnv, "build_forecast_windows", lambda **kw: res)
    monkeypatch.setattr(bnv, "_score_candidate", recorder)
    # main() rebinds the module global HORIZONS from --horizons; monkeypatch undoes it.
    monkeypatch.setattr(bnv, "HORIZONS", list(bnv.HORIZONS))
    monkeypatch.setattr(sys, "argv", [
        "benchmark_vs_nwp.py",
        "--candidate-dir", CANDIDATE_DIR,
        "--horizons", "6",
        "--out", os.path.join(str(tmp_path), "unused_results.json"),
    ])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bnv.main()
    return calls, buf.getvalue()


class TestCandidateScoringFeatureSpace:
    """A candidate checkpoint is trained on z-scores and must be scored on z-scores."""

    def test_candidate_scoring_receives_the_normalised_tensor(self, monkeypatch, tmp_path):
        """The primary guard, and the one that is exact.

        `main()` builds both `X` (normalised) and `X_raw` (de-normalised + clipped)
        and passes exactly one of them to `_score_candidate`. The correct choice is
        the normalised tensor: passing `X_raw` shifted every feature out of
        distribution and turned the +3h wind MAE into 0.742 instead of 0.498.
        """
        res = _fake_windows()
        calls, out = _run_candidate_main(monkeypatch, tmp_path, res)

        assert len(calls) == 1, "the candidate scoring path was not exercised:\n" + out
        seen = calls[0]["telemetry"]
        assert torch.is_tensor(seen), (
            "candidate scoring was handed %s; a candidate checkpoint consumes a "
            "torch tensor" % type(seen).__name__
        )
        assert seen is res[0], (
            "the candidate was scored on something other than the normalised "
            "res[0] tensor"
        )
        assert seen.shape == res[0].shape

    def test_candidate_scoring_is_not_given_the_denormalised_array(self, monkeypatch, tmp_path):
        """Value-based restatement: de-normalising what was passed yields X_raw.

        Identity alone is brittle; this pins the semantics. If the tensor that
        reached the scorer de-normalises and clips to exactly the production
        bundle's expected input, it was the wrong tensor.
        """
        res = _fake_windows()
        calls, out = _run_candidate_main(monkeypatch, tmp_path, res)
        assert len(calls) == 1, out

        seen = calls[0]["telemetry"].detach().cpu().numpy().astype(np.float64)
        x = res[0].detach().cpu().numpy().astype(np.float64)
        x_raw = np.clip(x * FAKE_NORM_STDS + FAKE_NORM_MEANS, RAW_CLIP_LO, RAW_CLIP_HI)

        assert not np.allclose(seen, x_raw), (
            "the candidate was scored on the DE-NORMALISED, clipped tensor -- "
            "that is the production bundle predictor's input space, not the "
            "candidate's; every feature is out of distribution"
        )
        assert np.allclose(
            np.clip(seen * FAKE_NORM_STDS + FAKE_NORM_MEANS, RAW_CLIP_LO, RAW_CLIP_HI),
            x_raw,
        ), "the scored tensor is neither the normalised tensor nor its inverse"

    def test_the_two_feature_spaces_are_distinguishable(self):
        """Non-vacuity: the previous test can actually fail."""
        res = _fake_windows()
        x = res[0].detach().cpu().numpy().astype(np.float64)
        x_raw = np.clip(x * FAKE_NORM_STDS + FAKE_NORM_MEANS, RAW_CLIP_LO, RAW_CLIP_HI)
        assert not np.allclose(x, x_raw)
        # Physical units are ~28 degC / ~1006 hPa; z-scores are O(1).
        assert 20.0 < x_raw[..., 0].mean() < 40.0
        assert abs(x[..., 0].mean()) < 5.0

    def test_candidate_call_site_does_not_pass_x_raw(self):
        """Source-level guard so a rename or a refactor cannot quietly undo this.

        Cheap, name-stable, and it fails long before a wrong number is published.
        """
        with open(os.path.join(HERE, "benchmark_vs_nwp.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_score_candidate"]
        assert calls, "_score_candidate is no longer called from this module"
        for call in calls:
            for arg in call.args:
                assert not (isinstance(arg, ast.Name) and arg.id == "X_raw"), (
                    "_score_candidate is being passed X_raw at line %d; candidates "
                    "are trained on the normalised tensor" % call.lineno
                )


_SCORE_CACHE = {}


def _score(candidate_dir, horizon, space, pipeline, res, meta, dt):
    """Score the real checkpoint once per (dir, horizon, feature space)."""
    key = (candidate_dir, horizon, space)
    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]
    if space == "normalised":
        tensor = res[0]
    else:
        x = res[0].detach().cpu().numpy().astype(np.float64)
        tensor = np.clip(x * pipeline.norm_stds + pipeline.norm_means,
                         RAW_CLIP_LO, RAW_CLIP_HI)
    # _score_candidate returns (predictions_by_variable, rain_probability).
    _SCORE_CACHE[key] = bnv._score_candidate(
        candidate_dir, horizon, tensor, dt, meta, pipeline)[0]
    return _SCORE_CACHE[key]


def _have_checkpoint(horizon):
    path = os.path.join(CANDIDATE_DIR, "candidate_h%dh.pt" % horizon)
    if not os.path.isdir(CANDIDATE_DIR):
        return None, "candidate_artifacts/ does not exist; no candidate to score"
    if not os.path.exists(path):
        return None, "no candidate checkpoint at %s" % path
    return path, ""


class TestCandidateFeatureSpaceCostsAccuracy:
    """What the structural guard above is actually protecting, in degrees.

    These call `_score_candidate` directly with an explicit feature space, so they
    are properties of the CHECKPOINT rather than of the call site: they hold
    whether or not `main()` passes `res[0]` or `X_raw`. Their job is to quantify
    the damage and to keep the benchmark's printed number tied to the number
    training audited in candidate_h*h_manifest.json, on the corpus the checkpoint
    was actually fitted on (weather_telemetry_sha256 = 86ce906453aa... ==
    data/weather_telemetry.csv).
    """

    def test_denormalised_input_degrades_the_plus12h_temperature_mae(self, default_pipeline):
        """+12h is where the corruption is loudest and most consistent.

        Measured on data/weather_telemetry.csv, candidate_artifacts:
            normalised     temperature MAE 1.835
            de-normalised  temperature MAE 2.490   (+35.7%)
        The same direction holds on the refetched corpus (+28.8%) and the same
        effect shows at +6h pressure (+12.3%) and +1h temperature (+9.8%). The
        threshold is set at +15% to sit well inside the observed margin rather
        than on top of it.
        """
        norm_mae, raw_mae = self._temp_mae(12, default_pipeline)
        assert raw_mae > norm_mae * 1.15, (
            "feeding the candidate its own training corpus in de-normalised units "
            "barely changed the error (normalised %.3f vs de-normalised %.3f). The "
            "checkpoint is supposed to be near-persistence, which is exactly why "
            "the +3h wind MAE printed 0.742 instead of 0.498 without looking wrong"
            % (norm_mae, raw_mae)
        )

    def test_normalised_scoring_reproduces_the_audited_manifest_mae(self, default_pipeline):
        """The number the benchmark prints must match the number training audited.

        candidate_h6h_manifest.json records candidate_temp_mae = 1.493 for the
        +6h candidate on data/weather_telemetry.csv. Scoring the checkpoint by
        hand through _score_candidate on the normalised test windows reproduces
        1.493 to three decimals. Anything the benchmark prints that cannot be
        reproduced against the manifest is a measurement bug wearing a
        three-decimal costume.
        """
        expected = self._manifest_temp_mae(6)
        got, _raw = self._temp_mae(6, default_pipeline)
        assert got == pytest.approx(expected, rel=0.10), (
            "re-scoring candidate_h6h.pt by hand gives temperature MAE %.3f but the "
            "audited manifest says %.3f; the benchmark is not scoring what training "
            "audited" % (got, expected)
        )

    def test_denormalised_scoring_cannot_reproduce_the_audited_manifest_mae(
            self, default_pipeline):
        """The negative half of the anchor, and it is the one with teeth.

        +12h: the manifest audited candidate_temp_mae = 1.745. Re-scoring on the
        normalised windows gives 1.835 (+5.2%, inside tolerance). Re-scoring the
        SAME checkpoint on de-normalised, clipped input gives 2.490 -- 42.7% away
        from the audited value. If the de-normalised run could still reproduce the
        audited number, this tolerance would no longer discriminate the two
        feature spaces and the +12h anchor would be decoration.
        """
        expected = self._manifest_temp_mae(12)
        _norm, raw = self._temp_mae(12, default_pipeline)
        assert abs(raw - expected) / expected > 0.10, (
            "de-normalised input scored %.3f against an audited %.3f and still "
            "looked acceptable; the +12h tolerance no longer discriminates the "
            "two feature spaces and this test needs a wider horizon" % (raw, expected)
        )

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _manifest_temp_mae(horizon):
        ck, why = _have_checkpoint(horizon)
        if ck is None:
            pytest.skip(why)
        path = os.path.join(CANDIDATE_DIR, "candidate_h%dh_manifest.json" % horizon)
        if not os.path.exists(path):
            pytest.skip("candidate_h%dh_manifest.json is not present" % horizon)
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["metrics"]["candidate_temp_mae"])

    @staticmethod
    def _temp_mae(horizon, pipe):
        """(normalised MAE, de-normalised MAE) for temperature at this horizon."""
        res = build_forecast_windows(pipeline=pipe, split="test", horizon=horizon,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        meta, dt = res[6], res[1].detach().cpu().numpy().astype(np.float64)
        truth = np.array([m[bnv.VAR_MAP["temperature"][1]] for m in meta], float)
        if len(truth) == 0:
            pytest.skip("test split is empty at +%dh" % horizon)
        out = []
        for space in ("normalised", "denormalised"):
            pred = _score(CANDIDATE_DIR, horizon, space, pipe, res, meta, dt)
            out.append(float(np.abs(pred["temperature"] - truth).mean()))
        return out[0], out[1]


# --------------------------------------------------------------------------- #
# BUG C
# --------------------------------------------------------------------------- #

AUDIT_LOOP_TARGETS = ["_label", "_target", "_metric", "_block"]
AUDITED_TARGETS = {
    "temperature", "humidity", "pressure", "wind_speed",
    "rain_occurrence", "precipitation_amount", "heat_index",
}


def _promotion_audit_tree():
    """Parse the training source once and hand back the tree plus the audit loop."""
    if not os.path.exists(TRAIN_SRC):
        pytest.skip("train_predictive_quality.py is not present")
    with open(TRAIN_SRC, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=TRAIN_SRC)
    return tree


def _promotion_audit_loop(tree=None):
    """Locate the `_target_routes` loop by structural fingerprint.

    Fingerprint: a `for` whose body calls `_route` and whose iterable is a
    literal list of constant tuples. Unique in this file today and robust to
    renames and to line movement, which a line-number lookup would not be.
    """
    if tree is None:
        tree = _promotion_audit_tree()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        calls_route = any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "_route"
            for c in ast.walk(node)
        )
        literal = isinstance(node.iter, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Tuple) for e in node.iter.elts
        )
        if calls_route and literal:
            found.append(node)
    assert len(found) == 1, (
        "expected exactly one literal promotion-audit loop calling _route, found %d"
        % len(found)
    )
    return found[0]


class TestPromotionAuditLoopShape:
    """Arity of the audit table vs arity of the loop that consumes it.

    The audit runs LAST, after every checkpoint is written. A ValueError here
    exits rc=1 with a complete artifact set on disk and no promotion verdict,
    which is indistinguishable from a clean run by looking at the filesystem.
    """

    def test_loop_targets_are_four_names(self):
        node = _promotion_audit_loop()
        assert isinstance(node.target, ast.Tuple), (
            "the audit loop no longer unpacks a tuple; if the target is a single "
            "name the row shape changed and this test needs rethinking"
        )
        names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
        assert names == AUDIT_LOOP_TARGETS, (
            "loop targets are %s; the audit routes (label, target, metric, block) "
            "and the scorecard lookup depends on all four" % (names,)
        )

    def test_every_row_has_exactly_as_many_fields_as_the_loop_has_targets(self):
        node = _promotion_audit_loop()
        n_targets = len(node.target.elts)
        rows = ast.literal_eval(node.iter)
        assert rows, "the promotion audit table is empty"
        for row in rows:
            assert len(row) == n_targets, (
                "%r has %d fields but the loop unpacks %d; this raises "
                "'not enough values to unpack (expected %d, got %d)' AFTER every "
                "checkpoint has already been written"
                % (row, len(row), n_targets, n_targets, len(row))
            )

    def test_rows_are_plain_constants_not_expressions(self):
        """`literal_eval` above already implies this; assert it explicitly."""
        node = _promotion_audit_loop()
        for el in node.iter.elts:
            for field in el.elts:
                assert isinstance(field, ast.Constant), (
                    "audit rows must be constant tuples; found %s"
                    % ast.dump(field)
                )

    def test_the_audit_covers_every_operational_target(self):
        node = _promotion_audit_loop()
        labels = {row[0] for row in ast.literal_eval(node.iter)}
        assert AUDITED_TARGETS.issubset(labels), (
            "the audit no longer routes %s; a dropped target is silently never "
            "promoted or demoted" % sorted(AUDITED_TARGETS - labels)
        )

    def test_loop_body_only_uses_names_it_binds_or_injects(self):
        """The loop is executed standalone below, so it must be self-contained."""
        node = _promotion_audit_loop()
        bound = {e.id for e in node.target.elts}
        assigned, used = set(), set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, (ast.Store, ast.Del)):
                    assigned.add(n.id)
                else:
                    used.add(n.id)
        allowed = {"_route", "_target_routes"} | set(dir(builtins))
        unbound = used - bound - assigned - allowed
        assert not unbound, (
            "the audit loop references free names %s; executing it standalone "
            "(as the next test does) would NameError before it could verdict"
            % sorted(unbound)
        )

    def test_the_audit_loop_actually_runs(self):
        """Execute the real source fragment and demand a verdict for every target.

        This is the test that reproduces the original failure verbatim. The
        extracted `for` node is compiled and run with a stub `_route`, so a
        three-field row raises the same ValueError the training run raised --
        here, in milliseconds, instead of after a full training run.
        """
        node = _promotion_audit_loop()
        seen = {}

        def stub_route(metric_name, target, metric, block="candidate_featured_model"):
            seen.setdefault(block, set()).add((metric_name, target, metric))
            return "INSUFFICIENT_EVIDENCE", {1: 0.5, 3: 0.25, 6: 0.0}

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {"_target_routes": {}, "_route": stub_route}
        try:
            exec(compile(module, TRAIN_SRC, "exec"), ns)  # noqa: S102
        except ValueError as exc:
            pytest.fail(
                "the promotion audit loop raised %r: the promotion verdict is "
                "computed LAST, so this aborted the run after all checkpoints "
                "were written and left a complete artifact set on disk"
                % (exc,)
            )

        routes = ns["_target_routes"]
        assert set(routes) == AUDITED_TARGETS, (
            "the audit produced %s" % sorted(routes)
        )
        for label, verdict in routes.items():
            assert verdict["status"] in (
                "PROMOTED", "RETAIN_BASELINE", "INSUFFICIENT_EVIDENCE"
            ), "%s: unknown verdict %r" % (label, verdict["status"])
            assert verdict["relative_improvement_by_horizon"], (
                "%s: audit returned no per-horizon skill" % label
            )
        assert seen.get("candidate_featured_model"), (
            "_route was never called with the featured-candidate block; the audit "
            "is reading the wrong scorecard section"
        )

    def test_audit_loop_lives_in_train_and_evaluate_all_horizons(self):
        """Guard against the verdict being computed somewhere nobody runs."""
        tree = _promotion_audit_tree()
        node = _promotion_audit_loop(tree)
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(x is node for x in ast.walk(fn)):
                    assert fn.name == "train_and_evaluate_all_horizons", (
                        "the promotion audit has moved into %s(); the pipeline entry "
                        "point that ships the verdict is train_and_evaluate_all_horizons"
                        % fn.name
                    )
                    return
        pytest.fail("the promotion audit loop is not inside any function")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
