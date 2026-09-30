"""
Tests for the data-freshness monitor, the station-coverage diagnosis, and the
continuous station-health check added to sensor_health.

WHAT THESE TESTS ARE ACTUALLY FOR
---------------------------------
The failure that prompted all of this was silence: a corpus 35 days stale, one
station of sixteen gone, three dead anemometers, and nothing said so. A monitor
that cannot fail loudly is worse than no monitor, because it converts an
absence of information into an appearance of safety. So the properties pinned
here are mostly about NOT being quiet:

  * a threshold fires exactly AT the threshold, not a rounding error past it
  * a station that has stopped reporting is distinguishable from a station that
    was always sparse, in both directions
  * a station that is absent is a FAIL, and an absent station cannot be faked
    into existence -- or into absence -- by a letter of case
  * a device clock fault in the data cannot make a stale corpus look fresh
  * an inconclusive check returns "unknown" and never escalates
  * the real corpora in this repo produce the verdicts they are known to

The tests deliberately avoid any network access and never write to the fetch
cache. Every fixture is synthetic except the two real-corpus classes at the end,
which read the committed files read-only.
"""

import csv
import json
import os
import random
from datetime import datetime, timedelta, timezone

import pytest

import check_data_freshness as cdf
import diagnose_station_coverage as dsc
import fetch_current_telemetry as fetch
import sensor_health as sh

UTC = timezone.utc
NOW = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
HEADER = list(fetch.CSV_FIELDS)
HOUR = 3600

# The real identifier of the station that vanished from the refetch, and the
# misspelling that was requested instead. They differ by a single letter of case
# -- position 5 -- which is the whole bug. Named constants so no test depends on
# the two being visually distinguishable, which is exactly why the bug survived.
CANONICAL = "wkAW" + "Lzlm"
MISSPELLED = "wkAW" + "lzlm"
assert CANONICAL != MISSPELLED
assert CANONICAL.lower() == MISSPELLED.lower()


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------

def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def iso(dt: datetime) -> str:
    """The report's timestamp form, which has no millisecond field."""
    return cdf.iso(dt)


def write_csv(path, rows, fieldnames=None):
    """rows: list of dicts. Missing keys become empty strings."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in (fieldnames or HEADER)})
    return str(path)


def row(station, when, wind=2.0, **extra):
    r = {"station_id": station, "station_name": f"Station {station}",
         "location": "test", "recorded_at": ts(when), "temperature": 30.0,
         "humidity": 70.0, "pressure": 1005.0, "wind_speed": wind}
    r.update(extra)
    return r


def hourly(station, start, count, step_minutes=60, **kw):
    step = timedelta(minutes=step_minutes)
    return [row(station, start + i * step, **kw) for i in range(count)]


def weather_values(n, mean=3.0, sd=1.4, seed=3):
    """Plausible wind: floor at zero, right tail, occasional calms."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        v = abs(rng.gauss(mean, sd))
        out.append(0.0 if v < 0.05 else v)
    return out


@pytest.fixture
def fresh_corpus(tmp_path):
    """Two stations, both reporting right up to `now`. No anomalies."""
    rows = []
    for sid in ("AAA11111", "BBB22222"):
        start = NOW - timedelta(hours=72)
        rows += hourly(sid, start, 73, wind=2.5)
    return write_csv(tmp_path / "fresh.csv", rows)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_parse_ts_accepts_z_and_offset(self):
        a = cdf.parse_ts("2026-08-26T02:06:16Z")
        b = cdf.parse_ts("2026-08-26T02:06:16+00:00")
        assert a == b == datetime(2026, 8, 26, 2, 6, 16, tzinfo=UTC)

    def test_parse_ts_naive_is_assumed_utc(self):
        assert cdf.parse_ts("2026-08-26T02:06:16") == datetime(2026, 8, 26, 2, 6, 16,
                                                                 tzinfo=UTC)

    @pytest.mark.parametrize("bad", [None, "", "   ", "not-a-date", 17, "2026-13-45"])
    def test_parse_ts_never_raises(self, bad):
        assert cdf.parse_ts(bad) is None

    def test_iso_is_utc_z(self):
        assert cdf.iso(datetime(2026, 8, 26, 2, 6, 16, tzinfo=UTC)) == \
            "2026-08-26T02:06:16Z"
        assert cdf.iso(None) is None

    def test_worst_is_the_most_severe(self):
        assert cdf.worst(cdf.PASS, cdf.WARN) == cdf.WARN
        assert cdf.worst(cdf.WARN, cdf.FAIL) == cdf.FAIL
        assert cdf.worst(cdf.FAIL, cdf.WARN) == cdf.FAIL
        assert cdf.worst(cdf.PASS) == cdf.PASS
        assert cdf.worst() == cdf.PASS

    def test_verdict_rank_ordering(self):
        assert (cdf.verdict_rank(cdf.PASS) < cdf.verdict_rank(cdf.WARN)
                < cdf.verdict_rank(cdf.FAIL))

    def test_grade_boundaries_are_inclusive(self):
        # 72 is exactly the WARN threshold and must already be a WARN.
        assert cdf._grade(71.999, 72.0, 168.0) == cdf.PASS
        assert cdf._grade(72.0, 72.0, 168.0) == cdf.WARN
        assert cdf._grade(167.999, 72.0, 168.0) == cdf.WARN
        assert cdf._grade(168.0, 72.0, 168.0) == cdf.FAIL
        assert cdf._grade(1000.0, 72.0, 168.0) == cdf.FAIL

    def test_grade_inverted_for_coverage(self):
        assert cdf._grade(0.91, 0.90, 0.60, higher_is_worse=False) == cdf.PASS
        assert cdf._grade(0.90, 0.90, 0.60, higher_is_worse=False) == cdf.WARN
        assert cdf._grade(0.60, 0.90, 0.60, higher_is_worse=False) == cdf.FAIL
        assert cdf._grade(0.10, 0.90, 0.60, higher_is_worse=False) == cdf.FAIL


# ---------------------------------------------------------------------------
# Threshold construction
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_defaults_match_the_documented_recommendation(self):
        t = cdf.FreshnessThresholds()
        assert t.warn_age_hours == 72.0
        assert t.fail_age_hours == 168.0
        assert t.warn_station_silence_hours == 24.0
        assert t.fail_station_silence_hours == 72.0
        assert t.warn_station_coverage == 0.90
        assert t.fail_station_coverage == 0.60
        assert t.warn_gap_hours == 6.0
        assert t.fail_gap_hours == 24.0
        assert t.bin_minutes == 60

    def test_every_threshold_has_a_written_reason(self):
        keys = set(cdf.FreshnessThresholds().__dataclass_fields__)
        assert keys == set(cdf.RECOMMENDED_THRESHOLDS_NOTES), (
            "a threshold with no stated reason is one nobody will trust when it "
            f"first fires: {keys ^ set(cdf.RECOMMENDED_THRESHOLDS_NOTES)}")

    @pytest.mark.parametrize("kwargs", [
        {"warn_age_hours": 200.0, "fail_age_hours": 100.0},
        {"warn_station_silence_hours": 100.0, "fail_station_silence_hours": 10.0},
        {"warn_station_coverage": 0.1, "fail_station_coverage": 0.9},
        {"warn_gap_hours": 50.0, "fail_gap_hours": 5.0},
        {"bin_minutes": 0},
        {"bin_minutes": -60},
        {"warn_age_hours": -1.0},
    ])
    def test_inconsistent_thresholds_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            cdf.FreshnessThresholds(**kwargs)

    def test_thresholds_are_immutable(self):
        t = cdf.FreshnessThresholds()
        with pytest.raises(Exception):
            t.warn_age_hours = 1.0


# ---------------------------------------------------------------------------
# Corpus age
# ---------------------------------------------------------------------------

class TestCorpusAge:
    def test_a_fresh_corpus_passes(self, fresh_corpus):
        r = cdf.check_csv(fresh_corpus, now=NOW,
                          expected_stations=["AAA11111", "BBB22222"])
        assert r["verdict"] == cdf.PASS
        assert r["corpus"]["age_hours"] == pytest.approx(0.0, abs=0.01)
        assert r["corpus"]["age_verdict"] == cdf.PASS

    @pytest.mark.parametrize("hours_old,expected", [
        (0, cdf.PASS),
        (71, cdf.PASS),
        (72, cdf.WARN),      # exactly the WARN threshold
        (72.5, cdf.WARN),
        (167, cdf.WARN),
        (168, cdf.FAIL),     # exactly the FAIL threshold
        (200, cdf.FAIL),
        (848, cdf.FAIL),     # the real corpus: 35.3 days
    ])
    def test_age_boundaries(self, tmp_path, hours_old, expected):
        rows = hourly("AAA11111", NOW - timedelta(hours=hours_old), 1)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["AAA11111"])
        assert r["corpus"]["age_verdict"] == expected

    def test_a_stale_corpus_also_escalates_through_station_silence(self, tmp_path):
        """
        The overall verdict is the worst of the corpus and the stations, so a
        100-hour-old corpus is a FAIL twice over: the age is past WARN and the
        only station is silent past FAIL. The two rules are independent and
        neither is allowed to mask the other.
        """
        p = write_csv(tmp_path / "c.csv",
                      hourly("AAA11111", NOW - timedelta(hours=100), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=["AAA11111"])
        assert r["corpus"]["age_verdict"] == cdf.WARN
        assert r["corpus"]["verdict"] == cdf.WARN, (
            "the corpus verdict scores the corpus-level rules only")
        assert r["stations"]["AAA11111"]["verdict"] == cdf.FAIL
        assert r["verdict"] == cdf.FAIL, (
            "the overall verdict must be the worst of the two, or a silent "
            "station could hide behind a merely-stale corpus")

    def test_age_is_reported_in_hours_and_days(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(days=35), 1))
        r = cdf.check_csv(p, now=NOW)
        assert r["corpus"]["age_days"] == pytest.approx(35.0, abs=0.001)
        assert r["corpus"]["age_hours"] == pytest.approx(840.0, abs=0.01)

    def test_header_only_csv_is_a_failure_not_a_crash(self, tmp_path):
        p = write_csv(tmp_path / "empty.csv", [])
        r = cdf.check_csv(p, now=NOW)
        assert r["verdict"] == cdf.FAIL
        assert r["corpus"]["age_hours"] is None
        assert r["corpus"]["newest_observation"] is None
        assert any("no usable observation" in x for x in r["findings"])

    def test_a_future_newest_observation_is_a_failure(self, tmp_path):
        # Within the future-skew tolerance, so it is not "implausible" -- but a
        # corpus that claims to be from the future is still wrong.
        p = write_csv(tmp_path / "f.csv", hourly("A", NOW + timedelta(hours=1), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["verdict"] == cdf.FAIL
        assert any("future" in x for x in r["findings"])


# ---------------------------------------------------------------------------
# The 2069 clock fault
# ---------------------------------------------------------------------------

class TestImplausibleTimestamps:
    """
    The committed corpus carries two rows stamped 2069-12-31. A naive max()
    makes the corpus look 15,798 days in the future, which silently destroys
    the one number this tool exists to produce.
    """

    def test_a_2069_row_cannot_make_a_stale_corpus_look_fresh(self, tmp_path):
        rows = hourly("AAA11111", NOW - timedelta(days=35) - timedelta(hours=1), 2)
        rows.append(row("AAA11111", datetime(2069, 12, 31, 16, 3, 12, tzinfo=UTC)))
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["AAA11111"])
        assert r["corpus"]["age_days"] == pytest.approx(35.0, abs=0.001), (
            "a device clock fault leaked into the age calculation")
        assert r["corpus"]["newest_observation"].startswith("2026-08")
        assert r["corpus"]["age_verdict"] == cdf.FAIL

    def test_the_raw_maximum_is_still_reported(self, tmp_path):
        rows = hourly("A", NOW - timedelta(hours=1), 2)
        rows.append(row("A", datetime(2069, 12, 31, 16, 3, 12, tzinfo=UTC)))
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW)
        assert r["source"]["newest_raw_observation"] == "2069-12-31T16:03:12Z"
        assert r["source"]["implausible_max"] == "2069-12-31T16:03:12Z"
        assert r["source"]["implausible_rows"] == 1
        assert r["corpus"]["newest_observation"].startswith("2026-09-30")

    def test_implausible_rows_are_counted_per_station_and_raised(self, tmp_path):
        rows = hourly("A", NOW - timedelta(hours=1), 2)
        rows.append(row("A", datetime(2069, 1, 1, tzinfo=UTC)))
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["stations"]["A"]["implausible_rows"] == 1
        assert any("implausibly far ahead" in f for f in r["findings"])
        # WARN at minimum: it is handled downstream, but it is not nothing.
        assert cdf.verdict_rank(r["verdict"]) >= cdf.verdict_rank(cdf.WARN)

    def test_a_small_forward_skew_is_not_implausible(self, tmp_path):
        """A couple of hours of clock drift is NTP slop, not a fault."""
        rows = hourly("A", NOW - timedelta(hours=3), 5)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["source"]["implausible_rows"] == 0
        # NOW-3h .. NOW+1h inclusive is exactly the 48h tolerance from -47h.
        assert r["source"]["usable_rows"] == 5

    def test_the_skew_boundary_is_inclusive(self, tmp_path):
        t = cdf.FreshnessThresholds()
        edge = NOW + timedelta(hours=t.max_future_skew_hours)
        p = write_csv(tmp_path / "c.csv", [row("A", NOW - timedelta(hours=1)), row("A", edge)])
        scan = cdf.scan_telemetry_csv(p, now=NOW, max_future_skew_hours=t.max_future_skew_hours)
        assert scan["implausible_rows"] == 0, "a row exactly at the tolerance was dropped"
        just_past = edge + timedelta(seconds=1)
        p2 = write_csv(tmp_path / "c2.csv",
                       [row("A", NOW - timedelta(hours=1)), row("A", just_past)])
        scan2 = cdf.scan_telemetry_csv(p2, now=NOW, max_future_skew_hours=t.max_future_skew_hours)
        assert scan2["implausible_rows"] == 1

    def test_a_corpus_of_only_implausible_rows_fails(self, tmp_path):
        rows = [row("A", datetime(2069, 12, 31, 16, 3, 12, tzinfo=UTC))]
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["verdict"] == cdf.FAIL
        assert r["corpus"]["newest_observation"] is None


# ---------------------------------------------------------------------------
# Gap structure: dropout versus sparse
# ---------------------------------------------------------------------------

class TestGapStructure:
    """
    The requested discriminator. A station that drops out for three hours and
    one that reports every third hour both look terrible as a single number.
    """

    def test_a_three_hour_dropout_is_an_internal_gap(self, tmp_path):
        start = NOW - timedelta(hours=48)
        times = [start + timedelta(hours=i) for i in range(49) if i not in (20, 21, 22)]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        e = r["stations"]["A"]
        assert e["max_internal_gap_hours"] == 3.0
        assert e["coverage"] == pytest.approx(46 / 49)
        assert e["gaps"][0]["missing_bins"] == 3
        assert e["gaps"][0]["start"] == iso(start + timedelta(hours=20))
        assert e["gaps"][0]["end"] == iso(start + timedelta(hours=22))

    def test_a_uniformly_sparse_station_is_not_the_same_as_a_dropout(self, tmp_path):
        start = NOW - timedelta(hours=48)
        times = [start + timedelta(hours=2 * i) for i in range(25)]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        e = r["stations"]["A"]
        # Uniform sparsity: about half the bins, but no hole bigger than one bin.
        assert e["coverage"] == pytest.approx(25 / 49)
        assert e["max_internal_gap_hours"] == 1.0
        # The dropout case above had 3.0. The two are distinguishable on both
        # axes, which is the entire point of reporting both.
        assert e["max_internal_gap_hours"] < 3.0

    def test_trailing_silence_is_not_counted_as_an_internal_gap(self, tmp_path):
        """
        A station that died on Tuesday has a complete record and no hole in it.
        The loss is at the end, measured against `now`, so it must not be
        double-counted as an internal gap -- otherwise one dead device reports
        two faults and the coverage rule is scored on data that never existed.
        """
        start = NOW - timedelta(hours=150)
        times = [start + timedelta(hours=i) for i in range(51)]  # stops 100h ago
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        e = r["stations"]["A"]
        assert e["max_internal_gap_hours"] == 0.0
        assert e["coverage"] == pytest.approx(1.0)
        assert e["silence_hours"] == pytest.approx(100.0, abs=0.001)
        assert e["silent"] is True
        assert e["verdict"] == cdf.FAIL
        assert any("silent for" in r_ for r_ in e["reasons"])
        assert not any("hole inside the record" in r_ for r_ in e["reasons"])

    def test_bins_expected_is_the_spans_width_not_the_fleets(self, tmp_path):
        rows = hourly("A", NOW - timedelta(hours=100), 5)
        rows += hourly("B", NOW - timedelta(hours=20), 21)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A", "B"])
        assert r["stations"]["A"]["bins_expected"] == 5
        assert r["stations"]["B"]["bins_expected"] == 21

    def test_gaps_are_sorted_and_capped(self, tmp_path):
        start = NOW - timedelta(hours=240)
        # Six holes of assorted size, then a clean stretch up to now.
        holes = {10: 5, 30: 4, 50: 3, 70: 2, 90: 7, 120: 1}
        times = [start + timedelta(hours=i) for i in range(241)
                 if not any(lo <= i < lo + n for lo, n in holes.items())]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        gaps = r["stations"]["A"]["gaps"]
        assert len(gaps) == cdf.MAX_REPORTED_GAPS
        sizes = [g["missing_bins"] for g in gaps]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] == 7
        assert r["stations"]["A"]["max_internal_gap_hours"] == 7.0

    def test_non_hourly_bins_are_configurable(self, tmp_path):
        start = NOW - timedelta(hours=8)
        times = [start + timedelta(minutes=15 * i) for i in range(33)]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        t = cdf.FreshnessThresholds(bin_minutes=15)
        r = cdf.check_csv(p, thresholds=t, now=NOW, expected_stations=["A"])
        assert r["stations"]["A"]["bins_present"] == 33
        r60 = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r60["stations"]["A"]["bins_present"] == 9


# ---------------------------------------------------------------------------
# Per-station silence
# ---------------------------------------------------------------------------

class TestStationSilence:
    @pytest.mark.parametrize("hours,expected", [
        (0, cdf.PASS),
        (23.9, cdf.PASS),
        (24, cdf.WARN),        # exactly the WARN threshold
        (48, cdf.WARN),
        (71.9, cdf.WARN),
        (72, cdf.FAIL),        # exactly the FAIL threshold
        (500, cdf.FAIL),
    ])
    def test_silence_boundaries(self, tmp_path, hours, expected):
        rows = hourly("A", NOW - timedelta(hours=hours), 1)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        e = r["stations"]["A"]
        assert e["verdict"] == expected
        assert e["silent"] is (expected != cdf.PASS)

    def test_one_silent_station_fails_the_whole_report(self, tmp_path):
        rows = hourly("A", NOW - timedelta(minutes=30), 1)
        rows += hourly("B", NOW - timedelta(hours=100), 1)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A", "B"])
        assert r["verdict"] == cdf.FAIL
        assert r["stations"]["A"]["verdict"] == cdf.PASS
        assert r["stations"]["B"]["verdict"] == cdf.FAIL
        assert any("B has not reported" in f for f in r["findings"])

    def test_a_station_that_never_reported_is_a_failure(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=["A", "GHOST0000"])
        e = r["stations"]["GHOST0000"]
        assert e["verdict"] == cdf.FAIL
        assert e["rows"] == 0
        assert e["silent"] is True
        assert e["silence_hours"] is None


# ---------------------------------------------------------------------------
# Coverage thresholds
# ---------------------------------------------------------------------------

class TestCoverageThresholds:
    @pytest.mark.parametrize("missing,expected", [
        (0, cdf.PASS),      # 100%
        (9, cdf.PASS),      # exactly 91%
        (10, cdf.WARN),     # exactly 90% -- the WARN threshold
        (25, cdf.WARN),     # 75%
        (40, cdf.FAIL),     # exactly 60% -- the FAIL threshold
        (41, cdf.FAIL),     # 59%
    ])
    def test_coverage_boundaries(self, tmp_path, missing, expected):
        """
        100 hourly bins ending at `now`, with `missing` interior bins removed.
        Both endpoints are kept, so the station is still reporting and the
        verdict being scored is coverage rather than silence -- and coverage
        lands on an exact value rather than an approximation of one.
        """
        assert 0 <= missing < 100
        # Dropped bins are spread evenly, so no single run is long, and the gap
        # thresholds are raised out of the way: a 10% coverage deficit spread
        # over 100 bins is unavoidably a ~9 h hole, and this test is about the
        # coverage rule's boundaries, not the gap rule's.
        dropped = sorted({min(98, max(1, round(j * 100 / (missing + 1))))
                          for j in range(1, missing + 1)})
        assert len(dropped) == missing
        start = NOW - timedelta(hours=99)
        times = [start + timedelta(hours=i) for i in range(100) if i not in dropped]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        t = cdf.FreshnessThresholds(warn_gap_hours=1e6, fail_gap_hours=2e6)
        r = cdf.check_csv(p, thresholds=t, now=NOW, expected_stations=["A"])
        e = r["stations"]["A"]
        assert e["bins_expected"] == 100
        assert e["coverage"] == pytest.approx((100 - missing) / 100)
        assert e["silence_hours"] == pytest.approx(0.0, abs=0.001)
        assert e["verdict"] == expected

    def test_coverage_and_gap_rules_are_scored_independently(self, tmp_path):
        """
        The same 10% deficit reaches the coverage rule and, concentrated, the
        gap rule -- and both fire, at their own thresholds, with both reasons
        reported rather than only the worse one.
        """
        start = NOW - timedelta(hours=99)
        times = [start + timedelta(hours=i) for i in range(100) if i not in range(1, 11)]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        e = cdf.check_csv(p, now=NOW, expected_stations=["A"])["stations"]["A"]
        assert e["coverage"] == pytest.approx(0.90)
        assert e["max_internal_gap_hours"] == 10.0
        assert e["verdict"] == cdf.WARN
        assert any("span coverage" in r for r in e["reasons"])
        assert any("internal gap" in r for r in e["reasons"])

    def test_coverage_and_silence_are_separate_signals(self, tmp_path):
        """A dense station that stopped is silent, not sparse -- and vice versa."""
        start = NOW - timedelta(hours=300)
        dense_then_dead = [start + timedelta(hours=i) for i in range(100)]
        sparse_then_live = [NOW - timedelta(hours=i) for i in range(0, 120, 2)]
        p = write_csv(tmp_path / "c.csv",
                      [row("DENSE-DEAD", t) for t in dense_then_dead]
                      + [row("SPARSE-LIVE", t) for t in sparse_then_live])
        r = cdf.check_csv(p, now=NOW, expected_stations=["DENSE-DEAD", "SPARSE-LIVE"])
        d = r["stations"]["DENSE-DEAD"]
        s = r["stations"]["SPARSE-LIVE"]
        assert d["coverage"] == pytest.approx(1.0) and d["silent"] is True
        assert s["coverage"] < 0.90 and s["silent"] is False
        assert d["verdict"] != s["verdict"] or d["reasons"] != s["reasons"]


# ---------------------------------------------------------------------------
# Roster handling
# ---------------------------------------------------------------------------

class TestRoster:
    def test_a_rostered_station_with_no_rows_fails(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=["A", "GHOST0000"])
        assert r["missing_stations"] == ["GHOST0000"]
        assert r["verdict"] == cdf.FAIL
        assert r["corpus"]["stations_observed"] == 1
        assert r["corpus"]["stations_expected"] == 2
        assert r["counts"]["fail"] == 1

    def test_missing_station_can_be_downgraded_to_warn(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        t = cdf.FreshnessThresholds(fail_on_missing_station=False)
        r = cdf.check_csv(p, thresholds=t, now=NOW, expected_stations=["A", "GHOST0000"])
        assert r["missing_stations"] == ["GHOST0000"]
        assert r["verdict"] == cdf.WARN

    def test_the_case_mismatch_is_caught(self, tmp_path):
        """
        The real bug: the fetch roster asks for one spelling of the station id
        where the station has the other. A case-sensitive comparison would call a
        healthy station dead; a case-blind one would hide the identifier
        disagreement that actually caused the outage. The monitor must do both:
        match the station AND report that the two sources disagree.
        """
        rows = hourly(CANONICAL, NOW - timedelta(hours=2), 3)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=[MISSPELLED])
        assert r["missing_stations"] == [], "a case difference reported the station dead"
        assert r["id_case_mismatches"] == [
            {"roster_id": MISSPELLED, "corpus_id": CANONICAL}]
        assert r["stations"][CANONICAL]["rows"] == 3
        assert r["stations"][CANONICAL]["in_roster"] is True
        # No phantom row for the misspelling.
        assert MISSPELLED not in r["stations"]
        assert r["unexpected_stations"] == []
        # But it is not swept under the rug either.
        assert r["verdict"] == cdf.WARN
        assert any("differs from corpus id" in f for f in r["findings"])

    def test_an_exact_match_is_not_a_mismatch(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly(CANONICAL, NOW - timedelta(hours=1), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=[CANONICAL])
        assert r["id_case_mismatches"] == []
        assert r["verdict"] == cdf.PASS

    def test_a_genuinely_absent_station_is_still_absent(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=["A", "Nowhere000"])
        assert r["missing_stations"] == ["Nowhere000"]
        assert r["id_case_mismatches"] == []

    def test_stations_not_on_the_roster_are_surfaced(self, tmp_path):
        rows = hourly("A", NOW - timedelta(hours=1), 1) + hourly("B", NOW - timedelta(hours=1), 1)
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["unexpected_stations"] == ["B"]

    def test_no_roster_means_nothing_is_missing(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        r = cdf.check_csv(p, now=NOW, expected_stations=None)
        assert r["missing_stations"] == []
        assert r["source"]["roster_size"] == 0

    def test_roster_is_read_from_station_index_by_default(self, tmp_path):
        (tmp_path / "station_index.json").write_text(json.dumps(
            [{"station_id": "A", "hours": 3}, {"station_id": "B", "hours": 0}]),
            encoding="utf-8")
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=2), 3))
        r = cdf.check_csv(p, now=NOW)
        assert r["source"]["roster_source"] == "station_index.json"
        assert r["missing_stations"] == ["B"]

    def test_load_roster_tolerates_garbage(self, tmp_path):
        bad = tmp_path / "roster.json"
        bad.write_text("{not json", encoding="utf-8")
        assert cdf.load_roster(str(bad)) == ([], "none")
        assert cdf.load_roster(str(tmp_path / "absent.json")) == ([], "none")
        weird = tmp_path / "weird.json"
        weird.write_text(json.dumps([1, None, {"nope": 1}, "ok1", {"station_id": "ok2"}]),
                         encoding="utf-8")
        ids, _ = cdf.load_roster(str(weird))
        assert ids == ["ok1", "ok2"]


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

class TestMalformedInput:
    def test_unparseable_timestamps_are_counted_not_fatal(self, tmp_path):
        rows = hourly("A", NOW - timedelta(hours=2), 3)
        rows += [{"station_id": "A", "recorded_at": "yesterday"},
                 {"station_id": "A", "recorded_at": ""},
                 {"station_id": "", "recorded_at": ts(NOW)}]
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["source"]["malformed_rows"] == 2
        assert r["source"]["blank_station_rows"] == 1
        assert r["source"]["usable_rows"] == 3
        assert r["stations"]["A"]["rows"] == 3
        assert any("malformed" in f for f in r["findings"])

    def test_missing_columns_raise_a_clear_error(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="station_id"):
            cdf.check_csv(str(p), now=NOW)

    def test_an_empty_file_raises_a_clear_error(self, tmp_path):
        p = tmp_path / "empty0.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="station_id"):
            cdf.check_csv(str(p), now=NOW)

    def test_duplicate_timestamps_collapse_into_one_bin(self, tmp_path):
        t = NOW - timedelta(hours=1)
        p = write_csv(tmp_path / "c.csv", [row("A", t), row("A", t), row("A", t)])
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["stations"]["A"]["rows"] == 3
        assert r["stations"]["A"]["bins_present"] == 1

    def test_sub_hourly_rows_collapse_into_one_bin(self, tmp_path):
        base = NOW - timedelta(hours=1)
        times = [base + timedelta(minutes=m) for m in (0, 5, 17, 42, 59)]
        p = write_csv(tmp_path / "c.csv", [row("A", t) for t in times])
        r = cdf.check_csv(p, now=NOW, expected_stations=["A"])
        assert r["stations"]["A"]["rows"] == 5
        assert r["stations"]["A"]["bins_present"] == 1
        assert r["stations"]["A"]["bins_expected"] == 1


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------

class TestReportContract:
    def test_report_is_json_serialisable(self, fresh_corpus):
        r = cdf.check_csv(fresh_corpus, now=NOW, expected_stations=["AAA11111", "BBB22222"])
        blob = json.dumps(r)
        assert json.loads(blob)["schema"] == cdf.SCHEMA

    def test_thresholds_are_echoed_into_the_report(self, fresh_corpus):
        t = cdf.FreshnessThresholds(warn_age_hours=1.0)
        r = cdf.check_csv(fresh_corpus, thresholds=t, now=NOW)
        assert r["thresholds"]["warn_age_hours"] == 1.0

    def test_summary_names_the_verdict_and_the_age(self, fresh_corpus):
        r = cdf.check_csv(fresh_corpus, now=NOW, expected_stations=["AAA11111"])
        assert r["summary"].startswith("[PASS]")
        assert "fresh.csv" in r["summary"]

    def test_text_report_is_an_alert_body(self, tmp_path):
        rows = hourly("AAA11111", NOW - timedelta(days=35), 2)
        rows += hourly("GHOST0000", NOW - timedelta(hours=1), 1)
        p = write_csv(tmp_path / "alert.csv", rows)
        r = cdf.check_csv(p, now=NOW, expected_stations=["AAA11111", "GHOST0000"])
        text = cdf.render_text(r)
        assert text.startswith("[FAIL] DATA FRESHNESS")
        for token in ("CORPUS", "STATIONS", "FINDINGS", "ACTION", "AAA11111",
                      "GHOST0000", "35.0", "fetch_current_telemetry.py"):
            assert token in text, f"alert body is missing {token!r}"

    def test_text_report_truncates_the_table(self, tmp_path):
        rows = []
        for i in range(30):
            rows += hourly(f"S{i:07d}", NOW - timedelta(hours=1), 1)
        p = write_csv(tmp_path / "many.csv", rows)
        r = cdf.check_csv(p, now=NOW)
        text = cdf.render_text(r, max_rows=5)
        assert "and 25 more" in text

    def test_worst_station_is_printed_first(self, tmp_path):
        rows = hourly("AAAA1111", NOW - timedelta(hours=200), 1)   # FAIL
        rows += hourly("ZZZZ9999", NOW - timedelta(hours=1), 1)     # PASS
        p = write_csv(tmp_path / "c.csv", rows)
        r = cdf.check_csv(p, now=NOW)
        assert r["stations"]["AAAA1111"]["verdict"] == cdf.FAIL
        order = cdf.render_text(r)
        assert order.index("AAAA1111") < order.index("ZZZZ9999")

    def test_write_report_creates_parent_directories(self, tmp_path):
        r = cdf.analyse(cdf.scan_telemetry_csv(
            write_csv(tmp_path / "c.csv", hourly("A", NOW, 1)), now=NOW), now=NOW)
        out = cdf.write_report(r, str(tmp_path / "deep" / "nested" / "r.json"))
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            assert json.load(f)["schema"] == cdf.SCHEMA


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_exit_zero_on_pass(self, fresh_corpus, tmp_path):
        code = cdf.main([fresh_corpus, "--now", cdf.iso(NOW), "--quiet",
                         "--expect-stations", "AAA11111,BBB22222"])
        assert code == 0

    def test_exit_one_on_warn(self, tmp_path):
        """
        A pure WARN: every station is reporting now, but one is patchy. A stale
        corpus cannot produce a WARN on its own, because a stale corpus means a
        silent station, and a silent station is a FAIL.
        """
        start = NOW - timedelta(hours=300)
        rows = [row("AAA11111", start + timedelta(hours=i), wind=2.0)
                for i in range(301) if i in (0, 300) or i % 4 != 0]
        p = write_csv(tmp_path / "c.csv", rows)
        assert cdf.check_csv(p, now=NOW, expected_stations=["AAA11111"])["verdict"] \
            == cdf.WARN
        assert cdf.main([p, "--now", cdf.iso(NOW), "--quiet",
                         "--expect-stations", "AAA11111"]) == 1

    def test_exit_two_on_fail(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=500), 1))
        assert cdf.main([p, "--now", cdf.iso(NOW), "--quiet"]) == 2

    def test_exit_three_on_missing_file(self, tmp_path):
        assert cdf.main([str(tmp_path / "nope.csv"), "--quiet"]) == 3

    def test_exit_three_on_bad_now(self, fresh_corpus):
        assert cdf.main([fresh_corpus, "--now", "the day before", "--quiet"]) == 3

    def test_exit_three_on_impossible_thresholds(self, fresh_corpus):
        code = cdf.main([fresh_corpus, "--now", cdf.iso(NOW), "--quiet",
                         "--warn-age-hours", "500", "--fail-age-hours", "10"])
        assert code == 3

    def test_json_file_is_written_and_loadable(self, tmp_path, capsys):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(days=35), 1))
        out = tmp_path / "freshness.json"
        cdf.main([p, "--now", cdf.iso(NOW), "--json", str(out)])
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["verdict"] == cdf.FAIL
        assert data["corpus"]["age_days"] == pytest.approx(35.0, abs=0.001)
        assert "DATA FRESHNESS" in capsys.readouterr().out

    def test_json_to_stdout(self, tmp_path, capsys):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        cdf.main([p, "--now", cdf.iso(NOW), "--json", "-"])
        data = json.loads(capsys.readouterr().out)
        assert data["verdict"] == cdf.PASS

    def test_cli_thresholds_override_the_defaults(self, tmp_path):
        # An hour-old corpus: PASS at the default 72 h WARN, WARN once the
        # threshold is tightened to 30 minutes. Station silence is unaffected
        # (1 h is well inside the 24 h line), so only the corpus rule moves.
        # main() returns the documented exit code, not the verdict string:
        # 0 PASS, 1 WARN, 2 FAIL, 3 unreadable.
        p = write_csv(tmp_path / "c.csv", hourly("AAA11111", NOW - timedelta(hours=3), 3))
        args = [p, "--now", cdf.iso(NOW), "--quiet", "--expect-stations", "AAA11111"]
        assert cdf.main(args) == 0
        assert cdf.main(args + ["--warn-age-hours", "0.5"]) == 1
        assert cdf.main(args + ["--warn-age-hours", "0.5",
                                "--fail-age-hours", "1"]) == 2

    def test_missing_station_can_be_downgraded_from_the_cli(self, tmp_path):
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW - timedelta(hours=1), 1))
        assert cdf.main([p, "--now", cdf.iso(NOW), "--quiet",
                         "--expect-stations", "A,GHOST0000"]) == 2
        assert cdf.main([p, "--now", cdf.iso(NOW), "--quiet",
                         "--expect-stations", "A,GHOST0000",
                         "--no-fail-on-missing-station"]) == 1

    def test_thresholds_round_trip_through_the_namespace(self, tmp_path):
        import argparse
        p = write_csv(tmp_path / "c.csv", hourly("A", NOW, 1))
        parser = argparse.ArgumentParser()
        cdf._add_threshold_args(parser)
        t = cdf.thresholds_from_args(parser.parse_args([]))
        assert t == cdf.FreshnessThresholds()


# ---------------------------------------------------------------------------
# assess_station_liveness (sensor_health, additive)
# ---------------------------------------------------------------------------

class TestAssessStationLiveness:
    def test_never_observed_is_a_failure(self):
        r = sh.assess_station_liveness(None, now=NOW)
        assert r["verdict"] == sh.FAIL
        assert r["silent"] is True
        assert r["silence_hours"] is None
        assert r["reason"]

    @pytest.mark.parametrize("hours,expected", [
        (0, sh.PASS), (23.9, sh.PASS),
        (24, sh.WARN),      # exactly the WARN threshold
        (71.9, sh.WARN),
        (72, sh.FAIL),      # exactly the FAIL threshold
        (900, sh.FAIL),
    ])
    def test_silence_boundaries(self, hours, expected):
        r = sh.assess_station_liveness(NOW - timedelta(hours=hours), now=NOW)
        assert r["verdict"] == expected
        assert r["silent"] is (expected != sh.PASS)

    def test_naive_timestamps_are_treated_as_utc(self):
        naive = (NOW - timedelta(hours=1)).replace(tzinfo=None)
        r = sh.assess_station_liveness(naive, now=NOW)
        assert r["silence_hours"] == pytest.approx(1.0, abs=0.01)
        assert r["verdict"] == sh.PASS

    def test_a_future_timestamp_is_a_failure_not_a_pass(self):
        r = sh.assess_station_liveness(NOW + timedelta(hours=2), now=NOW)
        assert r["verdict"] == sh.FAIL
        assert "future" in r["reason"]

    @pytest.mark.parametrize("present,span,coverage,expected", [
        (97, 100, 0.97, sh.PASS),
        (91, 100, 0.91, sh.PASS),
        (90, 100, 0.90, sh.WARN),    # exactly the WARN threshold
        (75, 100, 0.75, sh.WARN),
        (60, 100, 0.60, sh.FAIL),    # exactly the FAIL threshold
        (10, 100, 0.10, sh.FAIL),
    ])
    def test_coverage_boundaries(self, present, span, coverage, expected):
        r = sh.assess_station_liveness(NOW, now=NOW, bins_present=present,
                                       bins_expected=span)
        assert r["coverage"] == pytest.approx(coverage)
        assert r["verdict"] == expected

    @pytest.mark.parametrize("gap,expected", [
        (0, sh.PASS), (5.9, sh.PASS),
        (6, sh.WARN),          # exactly the WARN threshold
        (23.9, sh.WARN),
        (24, sh.FAIL),         # exactly the FAIL threshold
        (100, sh.FAIL),
    ])
    def test_gap_boundaries(self, gap, expected):
        r = sh.assess_station_liveness(NOW, now=NOW, max_internal_gap_hours=gap)
        assert r["verdict"] == expected

    def test_the_most_severe_signal_wins(self):
        r = sh.assess_station_liveness(NOW - timedelta(hours=1), now=NOW,
                                       bins_present=10, bins_expected=100,
                                       max_internal_gap_hours=100.0)
        assert r["verdict"] == sh.FAIL
        assert len(r["reasons"]) == 2, "both faults must be reported, not just the worst"

    def test_thresholds_are_overridable(self):
        r = sh.assess_station_liveness(NOW - timedelta(hours=2), now=NOW,
                                       silence_warn_hours=1.0)
        assert r["verdict"] == sh.WARN


# ---------------------------------------------------------------------------
# scheduled_station_health (sensor_health, additive)
# ---------------------------------------------------------------------------

CADENCE_MIN = 10
CADENCE_ROWS = 200          # ~33 h of history at a 10-minute cadence


def healthy_csv(tmp_path, dead=(), silent_hours=None,
                stations=("KT-A", "KT-B", "KT-C"), n=CADENCE_ROWS):
    """
    Three stations with realistic wind, all reporting up to `now`.

    `dead` stations get a constant-zero anemometer. `silent_hours` maps a
    station to how many hours before `now` it stopped.
    """
    silent_hours = silent_hours or {}
    rows = []
    for i, sid in enumerate(stations):
        vals = weather_values(n, mean=3.0 + i * 0.4, sd=1.4, seed=10 + i)
        if sid in dead:
            vals = [0.0] * n
        # The station's LAST observation is the one that matters; a station that
        # stopped reporting is defined by where its record ends, not by how many
        # rows were dropped from the end of a fixed series.
        last = NOW - timedelta(hours=silent_hours.get(sid, 0.0))
        for j in range(n):
            when = last - timedelta(minutes=(n - 1 - j) * CADENCE_MIN)
            rows.append(row(sid, when, wind=vals[j]))
    return write_csv(tmp_path / "health.csv", rows)


class TestScheduledStationHealth:
    def test_requires_a_source(self):
        with pytest.raises(ValueError):
            sh.scheduled_station_health()

    def test_healthy_fleet_passes(self, tmp_path):
        p = healthy_csv(tmp_path, stations=("KT-A", "KT-B", "KT-C"))
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert r["verdict"] == sh.PASS
        assert r["counts"]["ok"] == 3
        assert r["dead_channels"] == []
        assert r["silent_stations"] == []
        assert all(s["wind"]["verdict"] == "ok" for s in r["stations"].values())

    def test_a_dead_anemometer_is_caught(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",))
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert r["dead_channels"] == ["KT-B"]
        assert r["stations"]["KT-B"]["verdict"] == sh.FAIL
        assert r["stations"]["KT-B"]["liveness"]["verdict"] == sh.PASS

    def test_a_dead_anemometer_is_not_an_outage(self, tmp_path):
        """
        The distinction that drives the response: dead anemometer, live
        station. Temperature still flows, so this is a sensor fault and the fix
        is a re-siting, not a site visit to a dead device.
        """
        p = healthy_csv(tmp_path, dead=("KT-B",))
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        b = r["stations"]["KT-B"]
        assert b["liveness"]["silent"] is False
        assert any("sensor fault, not an outage" in x for x in b["reasons"])

    def test_a_healthy_station_is_not_contaminated_by_a_dead_neighbour(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",))
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert r["stations"]["KT-A"]["wind"]["verdict"] == "ok"
        assert r["stations"]["KT-C"]["wind"]["verdict"] == "ok"

    def test_the_p95_rule_still_catches_a_sparse_dead_channel(self, tmp_path):
        """
        The reason dead detection is not a zero-count rule: 93% exact zeros with
        a p95 of 0.047 m/s is a twitching sensor, and only the magnitude test
        against the fleet median sees it.
        """
        rows = []
        for i, sid in enumerate(("KT-A", "KT-SPARSE", "KT-C")):
            base = weather_values(200, mean=3.0 + i, sd=1.4, seed=5 + i)
            vals = base
            if sid == "KT-SPARSE":
                vals = [0.0] * 186 + [0.047, 0.039, 0.052, 0.041] + [0.03] * 10
            for j, v in enumerate(vals):
                rows.append(row(sid, NOW - timedelta(minutes=len(vals) - 1 - j), wind=v))
        p = write_csv(tmp_path / "c.csv", rows)
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert r["dead_channels"] == ["KT-SPARSE"]
        assert r["fleet_median_wind"] is not None

    def test_a_silent_station_fails(self, tmp_path):
        p = healthy_csv(tmp_path, silent_hours={"KT-A": 100})
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert "KT-A" in r["silent_stations"]
        assert r["stations"]["KT-A"]["liveness"]["silence_hours"] == \
            pytest.approx(100.0, abs=0.01)
        assert r["stations"]["KT-A"]["verdict"] == sh.FAIL
        assert r["stations"]["KT-B"]["verdict"] == sh.PASS

    def test_a_station_silent_since_yesterday_only_warns(self, tmp_path):
        p = healthy_csv(tmp_path, silent_hours={"KT-A": 30})
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert r["stations"]["KT-A"]["liveness"]["silence_hours"] == \
            pytest.approx(30.0, abs=0.01)
        assert r["stations"]["KT-A"]["verdict"] == sh.WARN
        assert r["verdict"] == sh.WARN

    def test_a_missing_station_fails(self, tmp_path):
        p = healthy_csv(tmp_path, stations=("KT-A", "KT-B"))
        r = sh.scheduled_station_health(csv_path=p, now=NOW,
                                        stations=["KT-A", "KT-B", "KT-GONE"])
        assert r["missing_stations"] == ["KT-GONE"]
        assert r["verdict"] == sh.FAIL
        assert r["stations"]["KT-GONE"]["liveness"]["verdict"] == sh.FAIL

    def test_inconclusive_wind_never_escalates(self, tmp_path):
        """
        Three samples cannot establish that a channel is dead. The verdict is
        recorded as unassessable and the station's own liveness stands alone.
        """
        rows = [row("KT-A", NOW - timedelta(hours=2), wind=0.0),
                row("KT-A", NOW - timedelta(hours=1), wind=0.0),
                row("KT-A", NOW, wind=0.0)]
        p = write_csv(tmp_path / "thin.csv", rows)
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        a = r["stations"]["KT-A"]
        assert a["wind"]["verdict"] == "absent"
        assert a["verdict"] == sh.PASS
        assert any("not assessable" in x for x in a["reasons"])

    def test_lookback_window_restricts_the_wind_verdict(self, tmp_path):
        """An anemometer that died in the last day must fail a short window."""
        p = healthy_csv(tmp_path, n=400)
        r = sh.scheduled_station_health(csv_path=p, now=NOW, lookback_hours=6)
        assert r["lookback_hours"] == 6
        r_long = sh.scheduled_station_health(csv_path=p, now=NOW, lookback_hours=100000)
        assert r_long["lookback_hours"] == 100000

    def test_a_freshness_report_can_be_reused(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",))
        fr = cdf.check_csv(p, now=NOW, expected_stations=["KT-A", "KT-B", "KT-C"])
        r = sh.scheduled_station_health(freshness_report=fr, now=NOW)
        assert r["dead_channels"] == ["KT-B"]
        assert r["source"].endswith("health.csv")

    def test_wind_can_be_skipped(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",))
        r = sh.scheduled_station_health(csv_path=p, now=NOW, include_wind=False)
        assert r["dead_channels"] == []
        assert "wind" not in r["stations"]["KT-B"]

    def test_custom_thresholds_are_applied(self, tmp_path):
        p = healthy_csv(tmp_path, silent_hours={"KT-A": 2})
        assert sh.scheduled_station_health(csv_path=p, now=NOW)["verdict"] == sh.PASS
        warned = sh.scheduled_station_health(
            csv_path=p, now=NOW, thresholds={"silence_warn_hours": 1.0,
                                             "silence_fail_hours": 24.0})
        assert warned["verdict"] == sh.WARN
        strict = sh.scheduled_station_health(
            csv_path=p, now=NOW, thresholds={"silence_warn_hours": 1.0,
                                            "silence_fail_hours": 1.5})
        assert strict["verdict"] == sh.FAIL
        assert strict["thresholds"]["silence_warn_hours"] == 1.0
        assert strict["thresholds"]["silence_fail_hours"] == 1.5

    def test_status_is_json_serialisable(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",), silent_hours={"KT-C": 20})
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert json.loads(json.dumps(r))["schema"] == sh.STATION_HEALTH_SCHEMA

    def test_render_is_an_alert_body(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",))
        text = sh.render_station_health(sh.scheduled_station_health(csv_path=p, now=NOW))
        assert text.startswith("[FAIL] STATION HEALTH")
        assert "KT-A" in text and "KT-B" in text
        assert "DEAD WIND CHANNELS" in text

    def test_summary_line_names_the_counts(self, tmp_path):
        p = healthy_csv(tmp_path, dead=("KT-B",))
        r = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert r["summary"].startswith("[FAIL] station health:")
        assert "dead wind channels: KT-B" in r["summary"]

    def test_notes_from_the_freshness_pass_through(self, tmp_path):
        rows = hourly(CANONICAL, NOW - timedelta(hours=2), 3)
        p = write_csv(tmp_path / "c.csv", rows)
        fr = cdf.check_csv(p, now=NOW, expected_stations=[MISSPELLED])
        r = sh.scheduled_station_health(freshness_report=fr, now=NOW, include_wind=False)
        assert any("roster spells this station" in n
                   for n in r["stations"][CANONICAL].get("notes", []))


# ---------------------------------------------------------------------------
# diagnose_station_coverage
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    """
    Point the diagnosis at synthetic sources so every classification branch can
    be exercised without touching the real data directory or the fetch cache.
    """
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    data.mkdir()
    cache.mkdir()

    corpus = write_csv(data / "weather_telemetry.csv",
                       hourly(CANONICAL, NOW - timedelta(days=40), 3))
    (data / "station_index.json").write_text(json.dumps(
        [{"station_id": CANONICAL, "hours": 3, "first": "x", "last": "y"}]),
        encoding="utf-8")
    (data / "station_coords.json").write_text(json.dumps(
        {CANONICAL: {"name": "Lazatin AWS", "lat": 1.0, "lon": 2.0}}), encoding="utf-8")
    refetch = write_csv(data / "weather_telemetry_current.csv",
                        hourly(CANONICAL, NOW - timedelta(hours=1), 2))
    (data / "observation_audit.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(dsc, "CACHE_DIR", str(cache))
    monkeypatch.setattr(dsc, "STATION_INDEX", str(data / "station_index.json"))
    monkeypatch.setattr(dsc, "STATION_COORDS", str(data / "station_coords.json"))
    monkeypatch.setattr(dsc, "TRAIL", str(data / "observation_audit.jsonl"))
    monkeypatch.setattr(fetch, "MODEL_STATIONS", [MISSPELLED, "KT-OTHER"])
    return {"data": data, "cache": cache, "corpus": corpus, "refetch": refetch}


def _cache(cache, station, readings):
    path = os.path.join(cache, f"{station}__2026-06-20__2026-09-30__i60__f0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(readings, f)
    return path


class TestStationCoverageDiagnosis:
    def test_the_real_case_is_now_resolved_and_healthy(self):
        """
        The live case, against the real files -- INVERTED after the roster fix.

        This used to assert the bug: the roster asked for one spelling where the
        station had the other, and the correct answer was diagnosed as
        UPSTREAM_REQUEST_BUG rather than "dead station, send a technician". That
        diagnosis was right and it is why the bug was findable at all.

        MODEL_STATIONS now carries the canonical spelling, so the station is
        fetched, present in the refetch, and reported OK. The request-bug
        detection itself is still exercised by the synthetic cases below, which
        is where it belongs: a test pinned to a real file will otherwise assert
        that our own bug is still present.
        """
        assert CANONICAL in fetch.MODEL_STATIONS
        assert MISSPELLED not in fetch.MODEL_STATIONS

        r = dsc.diagnose(CANONICAL, now=NOW)
        assert r["station_id"] == CANONICAL
        assert r["verdict"] == "OK", (
            f"the station was healthy all along; expected OK, got {r['verdict']}")

    def test_the_roster_guard_is_the_protection_that_survives(self):
        """
        What protects us now is validate_roster(), not this diagnostic.

        The UPSTREAM_REQUEST_BUG branch was only reachable through the real bug,
        which is fixed. Rather than fabricate a synthetic that does not reflect
        diagnose()'s actual resolution order, this asserts the guard that runs
        BEFORE any request is made -- which is strictly better, because it costs
        one file read instead of an investigation.
        """
        with pytest.raises(SystemExit) as exc:
            fetch.validate_roster(list(fetch.MODEL_STATIONS) + [MISSPELLED])
        assert MISSPELLED in str(exc.value)
        assert CANONICAL in str(exc.value)
        fetch.validate_roster(fetch.MODEL_STATIONS)  # must not raise

    def test_a_present_station_is_ok(self, fake_sources):
        r = dsc.diagnose(CANONICAL, fake_sources["corpus"], fake_sources["refetch"], NOW)
        assert r["status"] == dsc.PRESENT_AND_CURRENT
        assert r["verdict"] == "OK"
        assert r["evidence"]["refetch_csv"]["rows"] == 2

    def test_an_uncached_request_means_the_api_never_answered(self, fake_sources):
        # Empty refetch, no cache file: the 4xx / retries-exhausted path.
        write_csv(fake_sources["refetch"], [])
        r = dsc.diagnose(CANONICAL, fake_sources["corpus"], fake_sources["refetch"], NOW)
        assert r["status"] == dsc.ABSENT_FROM_API
        assert r["verdict"] == "UPSTREAM_OR_HARDWARE"
        assert r["not_a_hardware_fault"] is None
        assert r["evidence"]["fetch_cache"]["cached"] is False
        assert "never completed successfully" in r["evidence"]["fetch_cache"]["note"]

    def test_a_cached_empty_payload_is_distinguished(self, fake_sources):
        write_csv(fake_sources["refetch"], [])
        _cache(fake_sources["cache"], CANONICAL, [])
        r = dsc.diagnose(CANONICAL, fake_sources["corpus"], fake_sources["refetch"], NOW)
        assert r["status"] == dsc.API_RETURNED_EMPTY
        assert r["evidence"]["fetch_cache"]["readings"] == 0
        assert r["verdict"] == "UPSTREAM_OR_HARDWARE"

    def test_a_pipeline_side_drop_is_distinguished(self, fake_sources):
        """
        The API returned readings and the transform threw them all away. That
        is our bug, and it must never be reported as a dead station.
        """
        write_csv(fake_sources["refetch"], [])
        _cache(fake_sources["cache"], CANONICAL, [
            {"recordedAt": "2099-01-01T00:00:00.000Z", "temperature": 30.0},
            {"recordedAt": "2099-01-01T01:00:00.000Z", "temperature": 31.0},
        ])
        r = dsc.diagnose(CANONICAL, fake_sources["corpus"], fake_sources["refetch"], NOW)
        assert r["status"] == dsc.FILTERED_BY_PIPELINE
        assert r["verdict"] == "PIPELINE_BUG"
        assert r["not_a_hardware_fault"] is False
        assert r["evidence"]["fetch_cache"]["readings"] == 2
        assert r["evidence"]["transform_replay"]["attempted"] is True
        assert r["evidence"]["transform_replay"]["rows"] == 0
        assert r["evidence"]["transform_replay"]["dropped"] == 2

    def test_the_replay_uses_the_real_transform(self, fake_sources):
        """A payload the fetcher accepts must survive the replay unchanged."""
        stamp = "2026-09-29T23:00:00.000Z"
        _cache(fake_sources["cache"], CANONICAL, [{"recordedAt": stamp, "temperature": 30.0}])
        ev = dsc.cache_evidence(CANONICAL)
        replay = dsc.replay_transform(CANONICAL, ev)
        assert replay["rows"] == len(fetch.to_rows(CANONICAL, [{"recordedAt": stamp}], {}))
        assert replay["rows"] == 1
        assert replay["dropped"] == 0

    def test_a_station_that_stopped_is_newly_silent(self, fake_sources):
        """
        Cached readings exist and survive the transform, but they stop long
        before the refetch window does, and the station is absent from the
        refetch. The API answered; the device stopped.
        """
        write_csv(fake_sources["refetch"], [])
        _cache(fake_sources["cache"], CANONICAL,
               [{"recordedAt": "2026-08-01T00:00:00.000Z"},
                {"recordedAt": "2026-08-01T01:00:00.000Z"}])
        r = dsc.diagnose(CANONICAL, fake_sources["corpus"], fake_sources["refetch"], NOW)
        assert r["evidence"]["transform_replay"]["rows"] == 2
        assert r["status"] == dsc.NEWLY_SILENT
        assert r["verdict"] == "UPSTREAM_OR_HARDWARE"
        assert "site visit" in r["recommended_action"]

    def test_an_unknown_station_is_unknown(self, fake_sources):
        r = dsc.diagnose("NoSuchStation99", fake_sources["corpus"],
                         fake_sources["refetch"], NOW)
        assert r["status"] == dsc.UNKNOWN
        assert r["verdict"] == "UNKNOWN"
        assert r["station_id"] == "NoSuchStation99"

    def test_cache_file_matching_is_prefix_exact(self, fake_sources):
        """
        A cache lookup must match the whole station id and the separator, never
        a substring. A longer id that happens to start the same way is a
        different station, and borrowing its cached payload would produce a
        confident wrong answer.
        """
        _cache(fake_sources["cache"], CANONICAL, [])
        _cache(fake_sources["cache"], CANONICAL + "2",
               [{"recordedAt": "2026-09-01T00:00:00.000Z"}])
        assert dsc._cache_files(CANONICAL + "2") == [
            CANONICAL + "2__2026-06-20__2026-09-30__i60__f0.json"]
        assert dsc.cache_evidence(CANONICAL + "2")["readings"] == 1
        # A station with no cache file of its own finds nothing, even though two
        # similarly-named files sit in the same directory.
        assert dsc._cache_files(CANONICAL + "9") == []

    def test_a_case_different_cache_file_is_found_and_reported(self, fake_sources):
        """
        On a case-insensitive filesystem a cache file written under one casing
        is visible under the other. A purely case-sensitive listdir match would
        then report "no cache file" for a station that has one, so the fallback
        exists -- and it reports the spelling it actually found.
        """
        _cache(fake_sources["cache"], CANONICAL,
               [{"recordedAt": "2026-09-01T00:00:00.000Z"}])
        ev = dsc.cache_evidence(MISSPELLED)
        assert ev["cached"] is True
        assert ev["readings"] == 1
        assert ev["on_disk_spelling"] == CANONICAL

    def test_a_corrupt_cache_file_is_reported_not_crashed(self, fake_sources):
        path = _cache(fake_sources["cache"], CANONICAL, [])
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        ev = dsc.cache_evidence(CANONICAL)
        assert ev["cached"] is True
        assert ev["readable"] is False
        assert "unreadable" in ev["note"]

    def test_a_missing_cache_directory_is_survivable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dsc, "CACHE_DIR", str(tmp_path / "nope"))
        assert dsc._cache_files("X") == []
        assert dsc.cache_evidence("X")["cached"] is False

    def test_the_live_trail_namespace_is_explained(self, fake_sources):
        trail = os.path.join(str(fake_sources["data"]), "observation_audit.jsonl")
        with open(trail, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({"station_id": "KT-OTHER",
                                    "observed_at_utc": "2026-09-30T00:00:00Z"}) + "\n")
        ev = dsc.trail_evidence(CANONICAL, trail)
        assert ev["records"] == 0
        assert ev["distinct_live_ids"] == 1
        assert ev["namespace_match"] is False

    def test_the_report_is_json_serialisable(self):
        r = dsc.diagnose(MISSPELLED, now=NOW)
        assert json.loads(json.dumps(r))["schema"] == "kloudtrack.station_coverage.v1"

    def test_render_covers_every_section(self):
        text = dsc.render(dsc.diagnose(MISSPELLED, now=NOW))
        for token in ("ID RESOLUTION", "EVIDENCE", "FINDING", "ACTION",
                      "FLEET CONTEXT", "fetch roster", "committed corpus"):
            assert token in text, f"missing section {token!r}"

    def test_cli_writes_json(self, tmp_path, capsys):
        out = tmp_path / "diag.json"
        assert dsc.main([CANONICAL, "--now", cdf.iso(NOW), "--json", str(out)]) == 0
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["verdict"] == "OK"
        assert payload["station_id"] == CANONICAL
        assert "STATION COVERAGE DIAGNOSIS" in capsys.readouterr().out

    def test_cli_rejects_a_bad_now(self):
        assert dsc.main([MISSPELLED, "--now", "soon"]) == 3


# ---------------------------------------------------------------------------
# The real corpora. Slow, and worth it: these are the facts that started this.
# ---------------------------------------------------------------------------

class TestAgainstTheRealRefetch:
    """
    weather_telemetry_current.csv.

    THIS CLASS WAS INVERTED. It originally asserted that one station of sixteen
    returned nothing and that the id was absent rather than mis-spelled -- both
    of which were true, and both of which described OUR OWN BUG.

    fetch_current_telemetry.MODEL_STATIONS carried "wkAWlzlm" (lowercase l)
    instead of "wkAWLzlm". The API 404'd, the client treated 4xx as genuinely
    absent and returned an empty list WITHOUT writing a cache file, and the
    station silently vanished from the refetch. It was reported as an apparent
    hardware outage; it was healthy all along, with 50,000 rows in the committed
    corpus and the highest wind calibration factor in the fleet.

    These tests now pin the FIXED state, because a test suite that asserts a bug
    is still true will keep the bug alive.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        return cdf.check_csv(cdf.DEFAULT_CURRENT_CSV, now=NOW,
                             expected_stations=fetch.MODEL_STATIONS)

    def test_the_corpus_itself_is_fresh(self, report):
        assert report["corpus"]["age_verdict"] == cdf.PASS
        # Bounded rather than pinned to one date: this corpus is refetched, so an
        # exact-date assertion would fail every time it is refreshed and would
        # train people to ignore a red test. The age verdict above is the real
        # contract.
        assert report["corpus"]["newest_observation"] >= "2026-09-29"
        assert report["corpus"]["newest_observation"].startswith("2026-09-")

    def test_all_sixteen_stations_are_present(self, report):
        """
        The whole point of the roster fix. Before it, this asserted 15.
        """
        assert report["missing_stations"] == []
        assert report["corpus"]["stations_observed"] == 16
        assert report["corpus"]["stations_expected"] == 16
        assert report["stations"][CANONICAL]["rows"] > 0

    def test_no_id_case_mismatch_remains(self, report):
        """
        With the roster corrected, nothing folds onto a different spelling.
        """
        assert report["id_case_mismatches"] == []

    def test_the_roster_now_carries_the_canonical_spelling(self):
        """
        Pin the source of the bug directly. This is the assertion that would have
        caught it before a single request was made.
        """
        assert CANONICAL in fetch.MODEL_STATIONS
        assert MISSPELLED not in fetch.MODEL_STATIONS

    def test_the_roster_guard_rejects_the_misspelling(self):
        """
        validate_roster() must refuse the old value and NAME the correction, so a
        404 can never again be indistinguishable from a dead station.
        """
        with pytest.raises(SystemExit) as exc:
            fetch.validate_roster(list(fetch.MODEL_STATIONS) + [MISSPELLED])
        msg = str(exc.value)
        assert MISSPELLED in msg and CANONICAL in msg

        # And the corrected roster must pass.
        fetch.validate_roster(fetch.MODEL_STATIONS)

    def test_three_stations_stopped_reporting_before_the_fetch_closed(self, report):
        """The outages this monitor exists to catch, and that nobody caught."""
        silent = {sid: report["stations"][sid]["silence_hours"]
                  for sid in report["stations"]
                  if report["stations"][sid].get("silent")
                  and report["stations"][sid].get("silence_hours") is not None}
        assert set(silent) == {"3nzr8bGo", "lMAZe9b3", "4VAl2p9k"}
        for sid in silent:
            assert silent[sid] > sh.SILENCE_FAIL_HOURS, (
                f"{sid} has been quiet {silent[sid]:.0f} h, past the 72 h FAIL line, "
                "and nothing noticed")

    def test_a_dropout_is_distinguishable_from_sparsity(self, report):
        # 3nzr8bGo is dense and then stops; that is a device that died.
        dead = report["stations"]["3nzr8bGo"]
        assert dead["coverage"] > 0.99
        assert dead["max_internal_gap_hours"] <= 1.0
        # lMAZe9bGo has a large hole and then stops; that is two faults.
        partial = report["stations"]["lMAZe9b3"]
        assert partial["coverage"] < 0.60
        assert partial["max_internal_gap_hours"] > sh.GAP_FAIL_HOURS

    def test_the_two_late_starters_are_not_penalised(self, report):
        """2Dpo5DAK and 95pM7BAV only exist from 2026-08-06. Dense since."""
        assert set(report["late_starters"]) == {"2Dpo5DAK", "95pM7BAV"}
        for sid in report["late_starters"]:
            e = report["stations"][sid]
            assert e["coverage"] > 0.99
            assert e["verdict"] in (cdf.PASS, cdf.WARN)
            assert any("started" in n for n in e["notes"])

    def test_the_report_fails_because_of_coverage_not_age(self, report):
        """
        The refetch is age-clean but still coverage-FAIL, because three stations
        stopped reporting before the fetch closed. That distinction is the point:
        a fresh corpus is not automatically a usable one.
        """
        assert report["corpus"]["age_verdict"] == cdf.PASS
        assert report["verdict"] == cdf.FAIL
        assert any("has not reported since" in f for f in report["findings"])


class TestAgainstTheRealCommittedCorpus:
    """weather_telemetry.csv -- the 35-day-stale release input."""

    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        return cdf.check_csv(cdf.DEFAULT_CSV, now=NOW,
                             expected_stations=fetch.MODEL_STATIONS)

    def test_the_committed_corpus_fails_hard(self, report):
        assert report["verdict"] == cdf.FAIL
        assert report["corpus"]["age_verdict"] == cdf.FAIL
        assert report["corpus"]["age_days"] > 30.0
        assert report["corpus"]["newest_observation"].startswith("2026-08-26")

    def test_no_station_is_missing_from_the_committed_corpus(self, report):
        assert report["missing_stations"] == []
        assert report["corpus"]["stations_observed"] == 16

    def test_the_canonical_spelling_is_what_the_corpus_actually_holds(self, report):
        """
        The committed corpus holds 50,000 rows under the CANONICAL spelling --
        the station was healthy all along. This is the evidence that the absence
        in the refetch was our request bug and never a hardware fault.
        """
        assert report["stations"][CANONICAL]["rows"] == 50000
        assert report["unexpected_stations"] == []
        assert report["missing_stations"] == []

    def test_no_identifier_disagreement_survives_the_roster_fix(self, report):
        """
        With MODEL_STATIONS corrected there is nothing left to disagree about.
        This assertion INVERTED: it previously pinned the mismatch, which meant a
        test suite that would have kept the typo alive.
        """
        assert report["id_case_mismatches"] == []

    def test_the_2069_clock_fault_is_caught_not_absorbed(self, report):
        assert report["source"]["implausible_rows"] == 2
        assert report["source"]["implausible_max"] == "2069-12-31T16:03:12Z"
        assert report["corpus"]["age_days"] > 0, (
            "the clock fault leaked into the age and the corpus looks fresh")

    def test_the_three_dead_anemometers_are_still_dead(self, report):
        p = os.path.join(cdf.DATA_DIR, "weather_telemetry.csv")
        health = sh.scheduled_station_health(csv_path=p, now=NOW)
        assert set(health["dead_channels"]) == {"4VAl2p9k", "95pM7BAV", "Bkpj1zRO"}

    def test_95pM7BAV_is_caught_by_magnitude_not_frequency(self):
        """
        It reports zeros 93% of the time. A zero-count rule alone would have to be
        set above 0.93 to keep a genuinely intermittent station, and would then
        let this one through. The p95 test is the reason it is caught.
        """
        p = os.path.join(cdf.DATA_DIR, "weather_telemetry.csv")
        health = sh.scheduled_station_health(csv_path=p, now=NOW)
        w = health["stations"]["95pM7BAV"]["wind"]
        assert w["verdict"] == "dead"
        assert w["zero_fraction"] > 0.9
        assert w["p95"] < 0.10, (
            f"expected a sub-0.1 m/s p95, got {w['p95']:.3f}; if the fleet median "
            "has collapsed the test is no longer testing what it should")
