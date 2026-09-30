"""
Tests for the data quality reporting layer.

What is worth testing here is not the JSON shape -- it is JSON, and a shape
test would only prove a dict literal is still a dict literal. The properties
that matter are the ones that would let a false reassurance reach an operator:

  1. The report must agree with dataset.py about which rows are quarantine
     and why. If the two drift, the report describes a pipeline that does not
     exist, and it will do so confidently. The bounds are compared by parsing
     dataset.py's source rather than importing it, because importing pulls in
     torch and the whole split/normalisation machinery for a dict literal.
  2. A ZERO is not a missing value. The historical bug in this pipeline was
     exactly that: a calm wind_speed of 0.0 was indistinguishable from an
     absent reading. If it regresses, rows are fabricated.
  3. Completeness is checked BEFORE bounds, so a row that is both incomplete
     and out of bounds is reported as incomplete. The report has to say so
     explicitly, otherwise the physical-bounds counters under-report and an
     operator chases a sensor outage that is really a corrupt transport.
  4. The clustering verdict must separate an outage from scattered noise. The
     remedy is a site visit in one case and a sensor swap in the other, and the
     two are indistinguishable in a row count.
  5. The bias measurement must exclude the field whose failure CAUSED the
     quarantine, or a row rejected for reporting 845 hPa always looks wildly
     biased on pressure -- true, and completely uninformative.
  6. A cohort too small to characterise must be reported as under-determined,
     not dressed up with an effect size computed from ten rows.
"""

import ast
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_data_quality import (  # noqa: E402
    CLUSTER_GAP_HOURS,
    MIN_BIAS_COHORT,
    OUTAGE_RUN_MIN_HOURS,
    PHYSICAL_BOUNDS,
    REASON_CLASS,
    REQUIRED_FIELDS,
    bias_verdict,
    build_report,
    classify_clustering,
    classify_row,
    companion_corruption_rate,
    continuity_metrics,
    contiguous_runs,
    describe,
    diagnose_pressure_cohort,
    implied_altitude_m,
    in_bounds,
    main,
    parse_field_value,
    parse_utc_timestamp,
    percentile,
    pressure_population_census,
    station_standardised_bias,
    time_matched_bias,
    trigger_fields_for_reason,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PY = os.path.join(SRC_DIR, "dataset.py")

CSV_HEADER = [
    "station_id", "station_name", "location", "recorded_at",
    "temperature", "heat_index", "humidity", "pressure",
    "wind_speed", "wind_direction", "precipitation",
]

T0 = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)


def ts(offset_hours: float) -> str:
    return (T0 + timedelta(hours=offset_hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def good_row(station="AAA111", hour=0.0, **overrides):
    """A fully in-bounds row. Override any field; pass None to blank it."""
    row = {
        "station_id": station,
        "station_name": "Test AWS",
        "location": "testville",
        "recorded_at": ts(hour),
        "temperature": 28.0,
        "heat_index": 31.0,
        "humidity": 70.0,
        "pressure": 1008.0,
        "wind_speed": 6.0,
        "wind_direction": 180.0,
        "precipitation": 0.0,
    }
    row.update(overrides)
    return row


def varied_row(station="AAA111", hour=0.0, temperature=28.0, humidity=70.0,
                pressure=1008.0, **overrides):
    """good_row with a diurnal cycle, so no channel is accidentally flat.

    A fixture whose temperature never varies has a zero IQR, and every effect
    size against it divides by zero. That is a property of the fixture, not of
    the code, and it turns a bias test into a null test by accident.
    """
    return good_row(station, hour,
                    temperature=temperature + 2.5 * math.sin(hour / 5.0),
                    humidity=humidity + 9.0 * math.sin(hour / 5.0 + 1.0),
                    pressure=pressure + 1.5 * math.sin(hour / 7.0),
                    **overrides)


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, ""))
                        for k in CSV_HEADER})
    return str(path)


# ---------------------------------------------------------------------------
# 1. Agreement with the pipeline this report describes
# ---------------------------------------------------------------------------

class TestContractWithDatasetPy:
    def test_physical_bounds_match_dataset_py(self):
        """Read PHYSICAL_BOUNDS out of dataset.py without importing it.

        Importing dataset.py pulls in torch and runs module-level work. The
        contract under test is a dict literal, so parse the literal.
        """
        with open(DATASET_PY, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (isinstance(tgt, ast.Name)
                            and tgt.id == "PHYSICAL_BOUNDS"):
                        found = ast.literal_eval(node.value)
        assert found is not None, "PHYSICAL_BOUNDS not found in dataset.py"
        for field, bounds in PHYSICAL_BOUNDS.items():
            assert tuple(found[field]) == tuple(bounds), (
                "report says %s bounds are %s, dataset.py says %s -- the "
                "report is describing a pipeline that does not exist"
                % (field, bounds, found[field]))

    def test_required_fields_match_dataset_parsing_loop(self):
        with open(DATASET_PY, "r", encoding="utf-8") as f:
            src = f.read()
        # The pipeline iterates its required fields as a literal tuple.
        assert '"temperature", "heat_index", "humidity", "wind_speed",' in src
        assert 'wind_direction", "pressure", "precipitation"' in src
        assert REQUIRED_FIELDS == (
            "temperature", "heat_index", "humidity", "wind_speed",
            "wind_direction", "pressure", "precipitation")

    def test_pressure_floor_is_nine_hundred(self):
        """A floor near 850 hPa would silently pass mountain-station data.

        Pinned explicitly because the task brief said "around 850" and the
        code says 900. If the floor is ever moved, a station-pressure reading
        at 890 hPa would start being ingested as if it were sea level.
        """
        assert PHYSICAL_BOUNDS["pressure"] == (900.0, 1050.0)

    def test_year_bounds_match(self):
        with open(DATASET_PY, "r", encoding="utf-8") as f:
            src = f.read()
        assert "MIN_VALID_YEAR = 2025" in src
        assert "MAX_VALID_YEAR = 2027" in src


# ---------------------------------------------------------------------------
# 2. A zero is not a missing value
# ---------------------------------------------------------------------------

class TestZeroIsNotMissing:
    @pytest.mark.parametrize("raw", ["0", "0.0", "0.000000", " -0.0 "])
    def test_zero_parses_as_zero(self, raw):
        assert parse_field_value(raw) == 0.0

    @pytest.mark.parametrize("raw", [
        "", "   ", "nan", "NaN", "NA", "n/a", "NULL", "none", "None", "-",
        "abc", "1,008", "1e400", "inf", "-inf", "Infinity", None,
    ])
    def test_absent_tokens_parse_as_none(self, raw):
        assert parse_field_value(raw) is None

    def test_a_calm_zero_wind_row_is_ingested(self):
        r = classify_row(good_row(wind_speed=0.0, precipitation=0.0))
        assert r["reason"] is None, (
            "a calm reading of 0.0 km/h and a true 0.0 mm were treated as "
            "missing -- this is the exact regression the old `float(x or "
            "default)` code caused")

    def test_a_zero_wind_row_still_reaches_the_parsed_values(self):
        r = classify_row(good_row(wind_speed=0.0))
        assert r["parsed"]["wind_speed"] == 0.0
        assert "wind_speed" not in r["missing_fields"]


# ---------------------------------------------------------------------------
# 3. Classification and reason precedence
# ---------------------------------------------------------------------------

class TestClassification:
    def test_clean_row_is_ingested(self):
        r = classify_row(good_row())
        assert r["reason"] is None
        assert r["missing_fields"] == []
        assert r["unreported_bounds_violations"] == []

    def test_one_blank_field_quarantines_the_whole_row(self):
        r = classify_row(good_row(humidity=""))
        assert r["reason"] == "weather_missing_field"
        assert r["missing_fields"] == ["humidity"]

    def test_several_blank_fields_are_all_listed(self):
        r = classify_row(good_row(humidity="", wind_speed=None, pressure="nan"))
        assert r["reason"] == "weather_missing_field"
        assert set(r["missing_fields"]) == {"humidity", "wind_speed", "pressure"}

    def test_out_of_bounds_field_names_the_field(self):
        r = classify_row(good_row(pressure=845.0))
        assert r["reason"] == "weather_bounds_pressure"

    def test_bounds_use_the_physical_range_not_a_loose_one(self):
        assert classify_row(good_row(temperature=9.9))["reason"] == \
            "weather_bounds_temperature"
        assert classify_row(good_row(temperature=50.1))["reason"] == \
            "weather_bounds_temperature"
        assert classify_row(good_row(temperature=10.0))["reason"] is None
        assert classify_row(good_row(temperature=50.0))["reason"] is None

    def test_completeness_beats_bounds_and_the_row_says_so(self):
        """A row both incomplete and out of bounds is filed as incomplete.

        dataset.py returns on the completeness branch, so its bounds counters
        never see the violation. The report must surface that rather than
        inherit the silence.
        """
        r = classify_row(good_row(humidity="", temperature=104.0, pressure=423.0))
        assert r["reason"] == "weather_missing_field"
        assert set(r["unreported_bounds_violations"]) == {"temperature", "pressure"}

    def test_a_bounds_trigger_is_not_reported_as_an_unreported_violation(self):
        """Regression: the offending field is the trigger, not a hidden one.

        Counting it twice inflated the figure from 347 rows to 1212 and made
        the corpus look twice as corrupt as it is.
        """
        r = classify_row(good_row(pressure=845.0))
        assert r["reason"] == "weather_bounds_pressure"
        assert r["unreported_bounds_violations"] == [], (
            "the trigger field was double-counted as a hidden violation")

    def test_bounds_checked_in_required_field_order(self):
        r = classify_row(good_row(temperature=104.0, pressure=423.0))
        assert r["reason"] == "weather_bounds_temperature"

    def test_malformed_timestamp(self):
        r = classify_row(good_row(recorded_at="not-a-time"))
        assert r["reason"] == "weather_malformed_timestamp"

    @pytest.mark.parametrize("year", ["2024", "2028", "2069"])
    def test_year_out_of_bounds(self, year):
        r = classify_row(good_row(
            recorded_at="%s-01-01T00:00:00.000Z" % year))
        assert r["reason"] == "weather_year_out_of_bounds"

    def test_missing_station_id(self):
        r = classify_row(good_row(station_id=""))
        assert r["reason"] == "weather_missing_station_id"

    def test_z_suffix_and_offset_forms_agree(self):
        a = parse_utc_timestamp("2026-08-01T00:00:00.000Z")
        b = parse_utc_timestamp("2026-08-01T00:00:00+00:00")
        c = parse_utc_timestamp("2026-08-01 00:00:00")
        assert a == b == c

    def test_every_reason_the_pipeline_can_emit_is_classified(self):
        """A reason with no entry falls through as 'other' and loses its
        remedy classification, silently."""
        emitted = {"weather_missing_field", "weather_year_out_of_bounds",
                   "weather_malformed_timestamp", "weather_missing_station_id"}
        emitted |= {"weather_bounds_" + f for f in REQUIRED_FIELDS}
        assert emitted <= set(REASON_CLASS)
        assert set(REASON_CLASS.values()) == {
            "completeness", "physical_bounds", "structural"}


# ---------------------------------------------------------------------------
# 4. Numbers
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_percentile_is_nearest_rank_and_never_interpolates(self):
        v = [1.0, 2.0, 3.0, 4.0]
        assert percentile(v, 0) == 1.0
        assert percentile(v, 50) == 3.0
        assert percentile(v, 100) == 4.0
        assert percentile([], 50) is None
        assert percentile([7.0], 99) == 7.0

    def test_describe_reports_the_requested_statistics(self):
        d = describe(list(range(101)))  # 0..100
        assert d["n"] == 101
        assert d["mean"] == pytest.approx(50.0)
        assert d["sd"] == pytest.approx(math.sqrt(850.0), abs=1e-3)
        assert d["min"] == 0.0 and d["max"] == 100.0
        assert d["p1"] == 1.0 and d["p99"] == 99.0 and d["p50"] == 50.0

    def test_describe_of_nothing_is_all_nulls_not_a_crash(self):
        d = describe([])
        assert d["n"] == 0
        assert all(d[k] is None for k in ("mean", "sd", "p1", "p50", "p99",
                                          "min", "max"))

    def test_in_bounds_is_inclusive_at_both_ends(self):
        assert in_bounds("pressure", 900.0)
        assert in_bounds("pressure", 1050.0)
        assert not in_bounds("pressure", 899.999)
        assert not in_bounds("pressure", 1050.001)


class TestAltitude:
    def test_sea_level_pressure_is_zero_altitude(self):
        assert implied_altitude_m(1013.25) == pytest.approx(0.0, abs=1e-6)

    def test_altitude_increases_as_pressure_falls(self):
        a = [implied_altitude_m(p) for p in (1013.25, 1000.0, 900.0, 845.0)]
        assert all(x is not None for x in a)
        assert a == sorted(a)

    def test_the_observed_845_hpa_deficit_is_about_1_4_km(self):
        # -159 hPa from a 1005 hPa normal: a 1.4 km site, i.e. a datum error
        # rather than a plausible station elevation for this fleet.
        alt = implied_altitude_m(845.37) - implied_altitude_m(1004.83)
        assert 1350.0 < alt < 1500.0

    def test_nonsense_pressure_has_no_altitude(self):
        assert implied_altitude_m(-188.9) is None
        assert implied_altitude_m(0) is None
        assert implied_altitude_m(None) is None


# ---------------------------------------------------------------------------
# 5. Clustering: outage or noise
# ---------------------------------------------------------------------------

def at(h):
    return T0 + timedelta(hours=h)


class TestClustering:
    def test_a_contiguous_block_is_an_outage(self):
        c = classify_clustering([at(h) for h in range(0, 30)])
        assert c["verdict"] == "clustered_outage"
        assert c["longest_run_hours"] == 30
        assert c["n_runs"] == 1
        assert c["implied_outage_hours"] == 30

    def test_scattered_rows_are_not_an_outage(self):
        c = classify_clustering([at(h) for h in range(0, 200, 7)])
        assert c["verdict"] == "spread"
        assert c["n_runs"] > 10
        assert c["implied_outage_hours"] == 0

    def test_a_run_just_under_the_threshold_is_not_an_outage(self):
        under = int(OUTAGE_RUN_MIN_HOURS) - 1
        over = int(OUTAGE_RUN_MIN_HOURS) + 1
        assert classify_clustering([at(h) for h in range(under)])["verdict"] == \
            "spread"
        assert classify_clustering([at(h) for h in range(over)])["verdict"] == \
            "clustered_outage"

    def test_a_gap_larger_than_the_tolerance_splits_the_run(self):
        # 2 h apart is beyond CLUSTER_GAP_HOURS, so this is two episodes even
        # though the wall-clock span is short.
        stamps = [at(h) for h in range(0, 20)] + [at(h) for h in range(22, 42)]
        c = classify_clustering(stamps)
        assert c["n_runs"] == 2
        assert c["longest_run_hours"] == 20
        assert len(contiguous_runs(stamps)) == 2

    def test_fill_fraction_of_a_dense_block_is_one(self):
        c = classify_clustering([at(h) for h in range(0, 11)])
        assert c["fill_fraction"] == pytest.approx(1.0)

    def test_span_and_day_count(self):
        c = classify_clustering([at(0), at(50), at(51)])
        assert c["span_hours"] == 51.0
        assert c["distinct_days"] >= 1

    def test_empty_input_does_not_crash(self):
        c = classify_clustering([])
        assert c["verdict"] == "spread"
        assert c["rows"] == 0 and c["n_runs"] == 0

    def test_unordered_input_is_handled(self):
        a = classify_clustering([at(h) for h in range(0, 30)])
        b = classify_clustering([at(h) for h in reversed(range(0, 30))])
        assert a["longest_run_hours"] == b["longest_run_hours"] == 30

    def test_gap_tolerance_default_is_between_1_and_2_hours(self):
        assert 1.0 < CLUSTER_GAP_HOURS < 2.0


class TestContinuity:
    def test_longest_gap_and_distinct_gap_count(self):
        m = continuity_metrics([at(0), at(1), at(2), at(20), at(21), at(22)])
        assert m["longest_gap_hours"] == 18.0
        assert m["distinct_gaps"] == 1
        assert m["observed_bins"] == 6

    def test_a_contiguous_series_has_no_gaps(self):
        m = continuity_metrics([at(h) for h in range(10)])
        assert m["longest_gap_hours"] == 0.0
        assert m["distinct_gaps"] == 0

    def test_a_multi_day_outage_is_measured_in_hours(self):
        stamps = [at(h) for h in range(5)] + [at(h) for h in range(1000, 1005)]
        m = continuity_metrics(stamps)
        assert m["longest_gap_hours"] == 996.0
        assert m["missing_bins"] == 995
        assert m["gaps_over_6h"] == 1

    def test_duplicate_timestamps_collapse(self):
        m = continuity_metrics([at(0), at(0), at(1)])
        assert m["observed_bins"] == 2

    def test_empty_input(self):
        m = continuity_metrics([])
        assert m["observed_bins"] == 0
        assert m["longest_gap_hours"] == 0.0


# ---------------------------------------------------------------------------
# 6. Pressure cohort diagnosis
# ---------------------------------------------------------------------------

def normal_pressure(n=400, median=1005.0):
    """Plausible sea-level pressure: wide synoptic swings, small hourly steps."""
    out, v = [], median
    for i in range(n):
        v += 0.7 * math.sin(i / 3.0) + 0.3 * math.sin(i / 11.0)
        out.append(v)
    return out


class TestPressureDiagnosis:
    def test_a_frozen_channel_is_detected(self):
        vals = [666.875] * 200
        d = diagnose_pressure_cohort(vals, describe(normal_pressure()))
        assert d["verdict"] == "frozen_sensor"
        assert d["modal_value"] == 666.875
        assert d["modal_value_share"] == 1.0
        assert d["recoverable_by_calibration"] is False

    def test_a_datum_offset_is_detected_and_marked_recoverable(self):
        """A tight band ~160 hPa low, with its structure intact, is a datum.

        This is the distinction the whole investigation turns on: the remedy is
        a calibration offset, not a drop.
        """
        normal = normal_pressure()
        base = describe(normal)
        shifted = [p - 159.3 + 0.9 * math.sin(i / 3.0)
                   for i, p in enumerate(normal[:200])]
        d = diagnose_pressure_cohort(shifted, base)
        assert d["verdict"] == "datum_offset", d["explanation"]
        assert d["recoverable_by_calibration"] is True
        assert d["median_offset_from_retained_hpa"] == pytest.approx(-159.3, abs=2.0)
        assert d["implied_altitude_delta_m"] == pytest.approx(1430.0, abs=60.0)
        assert d["abs_step_sd"] > 0.0, "structure was lost; that is not a datum"

    def test_a_datum_band_that_merely_wobbles_is_not_called_a_datum(self):
        """Tightness alone is not enough; the offset must be large in the
        station's own units or every mildly-low cluster becomes a 'datum'."""
        base = describe(normal_pressure())
        near_normal = [p - 1.0 for p in normal_pressure(100)]
        d = diagnose_pressure_cohort(near_normal, base)
        assert d["verdict"] != "datum_offset"

    def test_scattered_garbage_is_corrupt(self):
        vals = [-188.9, 320.6, 1021719.0, 407.9, 1992.1, 666.9, 812.3, 55.0]
        d = diagnose_pressure_cohort(vals, describe(normal_pressure()))
        assert d["verdict"] == "corrupt"
        assert d["recoverable_by_calibration"] is False

    def test_a_small_cohort_is_never_diagnosed(self):
        """Four bad rows cannot establish a pattern, whatever they look like."""
        d = diagnose_pressure_cohort([845.1, 845.2, 845.0, 845.3],
                                     describe(normal_pressure()))
        assert d["verdict"] != "datum_offset"

    def test_diagnosis_without_retained_reference_does_not_claim_an_offset(self):
        d = diagnose_pressure_cohort([845.0 + 0.01 * i for i in range(200)], None)
        assert d["verdict"] in ("corrupt", "frozen_sensor")

    def test_census_aggregates_by_verdict(self):
        per_station = {
            "AAA": {"verdict": "datum_offset", "rows": 100,
                    "median_offset_from_retained_hpa": -159.0, "modal_value": None,
                    "modal_value_share": 0.0},
            "BBB": {"verdict": "frozen_sensor", "rows": 40,
                    "median_offset_from_retained_hpa": -342.0,
                    "modal_value": 666.875, "modal_value_share": 1.0},
            "CCC": {"verdict": "corrupt", "rows": 10,
                    "median_offset_from_retained_hpa": 5.0, "modal_value": None,
                    "modal_value_share": 0.0},
        }
        c = pressure_population_census(per_station)
        assert c["total_rejected_rows"] == 150
        assert c["populations"]["datum_offset"]["rows"] == 100
        assert c["populations"]["datum_offset"]["pct_of_rejected"] == pytest.approx(66.67, abs=0.01)
        assert set(c["populations"]) == {"datum_offset", "frozen_sensor", "corrupt"}


# ---------------------------------------------------------------------------
# 7. Bias measurement
# ---------------------------------------------------------------------------

def retained_series(station, n=600, temperature=28.0, humidity=80.0):
    rows = []
    for i in range(n):
        rows.append(classify_row(good_row(
            station, hour=i,
            temperature=temperature + 1.2 * math.sin(i / 4.0),
            humidity=humidity + 6.0 * math.sin(i / 4.0))))
    return rows


def quarantined_series(station, n, temperature, humidity, start_hour=100.0):
    return [classify_row(good_row(
        station, hour=start_hour + i,
        temperature=temperature, humidity=humidity,
        pressure=845.0)) for i in range(n)]


class TestBiasMeasurement:
    def test_the_trigger_field_is_excluded(self):
        """Circular otherwise: a row rejected on pressure always looks biased
        on pressure, and that says nothing about the rows we kept."""
        ret = retained_series("AAA")
        quar = quarantined_series("AAA", 60, 20.0, 55.0)
        bias = time_matched_bias(quar, {"AAA": ret},
                                fields=("temperature", "humidity", "pressure"),
                                exclude_fields=("pressure",))
        assert bias["pressure"]["excluded"]
        assert "effect_size" not in bias["pressure"]
        assert "effect_size" in bias["temperature"]

    def test_a_planted_shift_is_detected(self):
        ret = retained_series("AAA", temperature=28.0, humidity=85.0)
        quar = quarantined_series("AAA", 80, 22.0, 60.0)
        bias = time_matched_bias(quar, {"AAA": ret},
                                fields=("temperature", "humidity"),
                                exclude_fields=("pressure",))
        assert bias["temperature"]["effect_size"] < -1.0
        assert bias["humidity"]["effect_size"] < -1.0
        assert bias["temperature"]["n_pairs"] > 30

    def test_no_shift_means_no_bias(self):
        ret = retained_series("AAA", temperature=28.0, humidity=85.0)
        quar = quarantined_series("AAA", 80, 28.0, 85.0)
        bias = time_matched_bias(quar, {"AAA": ret},
                                fields=("temperature", "humidity"),
                                exclude_fields=("pressure",))
        assert abs(bias["temperature"]["effect_size"]) <= 0.5
        assert abs(bias["humidity"]["effect_size"]) <= 0.5

    def test_the_station_itself_does_not_cause_a_spurious_shift(self):
        """Two stations with genuinely different climates must not read as
        biased against each other; the match is within a station."""
        ret = retained_series("AAA", temperature=28.0, humidity=85.0)
        quar = quarantined_series("BBB", 60, 31.0, 92.0)
        bias = time_matched_bias(quar, {"BBB": []},
                                fields=("temperature", "humidity"),
                                exclude_fields=("pressure",))
        assert bias["temperature"].get("n_pairs", 0) == 0

    def test_a_thin_control_is_refused_rather_than_divided_by(self):
        """Three neighbours give a meaningless IQR; dividing by it produces an
        enormous fake effect size."""
        ret = [classify_row(good_row("AAA", hour=0))]
        quar = [classify_row(good_row("AAA", hour=h, temperature=10.0))
                for h in range(1, 40)]
        bias = time_matched_bias(quar, {"AAA": ret}, fields=("temperature",),
                                exclude_fields=("pressure",))
        assert bias["temperature"].get("effect_size") is None
        assert "too few" in bias["temperature"].get("note", "")

    def test_control_scale_is_robust_to_one_absurd_companion(self):
        """SD would be inflated by a single corrupt control reading and would
        shrink every effect size toward zero."""
        ret = retained_series("AAA", temperature=28.0, humidity=85.0)
        for r in ret[:20]:
            r["parsed"]["humidity"] = 100.0
        quar = quarantined_series("AAA", 80, 28.0, 60.0)
        bias = time_matched_bias(quar, {"AAA": ret}, fields=("humidity",),
                                exclude_fields=("pressure",))
        assert abs(bias["humidity"]["effect_size"]) >= 1.0

    def test_station_standardised_z_is_reported(self):
        ret = retained_series("AAA", temperature=28.0)
        quar = quarantined_series("AAA", 60, 22.0, 80.0)
        ss = station_standardised_bias(
            quar, {"AAA": ret},
            fields=("temperature", "humidity", "pressure"),
            exclude_fields=("pressure",))
        assert ss["temperature"]["mean_z_clamped"] < -1.0
        assert ss["pressure"]["excluded"]

    def test_a_flat_control_confirms_zero_but_cannot_size_a_shift(self):
        """A dead-flat control channel has no baseline to divide by. A zero
        delta is still confirmable without one; a non-zero delta is not, and
        inventing a scale for it would manufacture an effect size."""
        ret = [classify_row(good_row("AAA", hour=h)) for h in range(200)]
        flat_q = [classify_row(good_row("AAA", hour=100 + h, temperature=28.0,
                                        pressure=845.0)) for h in range(60)]
        same = time_matched_bias(flat_q, {"AAA": ret},
                                 fields=("temperature",),
                                 exclude_fields=("pressure",))
        assert same["temperature"]["control_is_flat"] is True
        assert same["temperature"]["effect_size"] == 0.0

        moved_q = [classify_row(good_row("AAA", hour=100 + h, temperature=20.0,
                                         pressure=845.0)) for h in range(60)]
        moved = time_matched_bias(moved_q, {"AAA": ret},
                                  fields=("temperature",),
                                  exclude_fields=("pressure",))
        assert moved["temperature"]["effect_size"] is None

    def test_no_control_in_range_is_reported_as_undetermined(self):
        """A quarantined episode entirely outside the retained history has no
        local baseline. Saying so beats dividing by nothing."""
        ret = [classify_row(good_row("AAA", hour=h)) for h in range(50)]
        far = [classify_row(good_row("AAA", hour=5000 + h, temperature=20.0,
                                     pressure=845.0)) for h in range(60)]
        b = time_matched_bias(far, {"AAA": ret}, fields=("temperature",),
                              exclude_fields=("pressure",))
        assert b["temperature"]["n_pairs"] == 0


class TestCompanionCorruption:
    def test_a_clean_cohort_has_no_companion_corruption(self):
        quar = quarantined_series("AAA", 40, 24.0, 78.0)
        c = companion_corruption_rate(quar, ("pressure",))
        assert c["rate"] == 0.0
        assert c["clean_subset_rows"] == 40

    def test_a_corrupt_cohort_is_identified(self):
        quar = quarantined_series("AAA", 40, 24.0, 78.0)
        for r in quar[:30]:
            r["parsed"]["precipitation"] = 432137.0
        c = companion_corruption_rate(quar, ("pressure",))
        assert c["rows_with_corrupt_companion"] == 30
        assert c["rate"] == 0.75
        assert "precipitation" in c["by_companion_field"]

    def test_the_trigger_field_is_never_a_companion(self):
        quar = quarantined_series("AAA", 10, 24.0, 78.0)
        c = companion_corruption_rate(quar, ("pressure",))
        assert "pressure" not in c["companion_fields_tested"]

    def test_trigger_field_extraction(self):
        assert trigger_fields_for_reason("weather_bounds_pressure") == ("pressure",)
        assert trigger_fields_for_reason("weather_missing_field") == ()
        assert trigger_fields_for_reason(None) == ()


class TestBiasVerdict:
    def test_thresholds_are_honoured(self):
        assert bias_verdict(0.0) == "no_material_selection_bias"
        assert bias_verdict(0.5) == "no_material_selection_bias"
        assert bias_verdict(0.7) == "minor_selection_bias"
        assert bias_verdict(2.0) == "material_selection_bias"
        assert bias_verdict(1.0) == "material_selection_bias"
        assert bias_verdict(None) == "undetermined"

    def test_a_small_cohort_is_under_determined_not_measured(self):
        assert bias_verdict(50.0, clean_subset_rows=9) == \
            "undetermined_cohort_too_small"
        assert bias_verdict(50.0, clean_subset_rows=MIN_BIAS_COHORT) != \
            "undetermined_cohort_too_small"

    def test_a_huge_effect_on_ten_rows_never_claims_a_bias(self):
        """Ten garbage rows out of 30,000 must not outrank a 4,000-row outage."""
        v = bias_verdict(9999.0, clean_subset_rows=3)
        assert v == "undetermined_cohort_too_small"
        assert bias_verdict(9999.0, clean_subset_rows=MIN_BIAS_COHORT + 1) == \
            "material_selection_bias"# ---------------------------------------------------------------------------
# 8. End to end
# ---------------------------------------------------------------------------

def build(path):
    return build_report(csv_path=path, git_commit="test", generated_at="FIXED")


class TestReportEndToEnd:
    def test_a_clean_corpus_is_reported_good(self, tmp_path):
        rows = [good_row("AAA%03d" % s, hour=h)
                for s in range(3) for h in range(200)]
        rep = build(write_csv(tmp_path / "clean.csv", rows))
        a = rep["row_accounting"]
        assert a["rows_in"] == 600
        assert a["rows_quarantined"] == 0
        assert a["ingested_pct"] == 100.0
        assert rep["data_health_verdict"]["verdict"] == "GOOD"

    def test_row_accounting_adds_up(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(200)]
        rows += [good_row("BBB", hour=h, humidity="") for h in range(20)]
        rows += [good_row("CCC", hour=h, pressure=845.0) for h in range(10)]
        rep = build(write_csv(tmp_path / "mixed.csv", rows))
        a = rep["row_accounting"]
        assert a["rows_in"] == 230
        assert sum(v["rows"] for v in a["by_reason"].values()) == \
            a["rows_quarantined"]
        assert a["rows_ingested"] + a["rows_quarantined"] == a["rows_in"]
        assert a["by_reason"]["weather_missing_field"]["rows"] == 20
        assert a["by_reason"]["weather_bounds_pressure"]["rows"] == 10

    def test_every_reason_carries_a_percentage_not_just_a_count(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(100)]
        rows += [good_row("BBB", hour=h, wind_speed="") for h in range(25)]
        rep = build(write_csv(tmp_path / "pct.csv", rows))
        for reason, d in rep["row_accounting"]["by_reason"].items():
            assert d["pct_of_rows_in"] > 0, reason
            assert d["pct_of_quarantined"] > 0, reason
            assert d["reason_class"], reason

    def test_field_level_missingness_is_not_addable(self, tmp_path):
        """dataset.py tallies one counter per absent field, so a row missing
        three fields lands in three buckets. The report must say so, or a
        reader will add them up and get a number larger than the row count."""
        rows = [good_row("AAA", hour=0, humidity="", pressure="", wind_speed="")]
        rep = build(write_csv(tmp_path / "multi.csv", rows))
        f = rep["row_accounting"]["field_level_missingness"]
        assert f["rows_with_at_least_one_missing_field"] == 1
        assert sum(v["field_occurrences"] for v in f["by_field"].values()) == 3
        assert "DO NOT sum" in f["note"]

    def test_a_contiguous_outage_is_labelled_clustered_not_spread(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(400)]
        rows += [good_row("AAA", hour=100 + h, humidity="") for h in range(120)]
        rep = build(write_csv(tmp_path / "outage.csv", rows))
        d = rep["quarantine_detail"]["weather_missing_field"]
        assert d["dominant_pattern"] == "clustered_outage"
        per = d["per_station"]["AAA"]
        assert per["longest_run_hours"] == 120
        assert per["implied_outage_hours"] == 120

    def test_scattered_loss_is_labelled_spread(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(400)]
        rows += [good_row("AAA", hour=h, humidity="") for h in range(0, 400, 37)]
        rep = build(write_csv(tmp_path / "spread.csv", rows))
        d = rep["quarantine_detail"]["weather_missing_field"]
        assert d["dominant_pattern"] == "spread"

    def test_the_report_finds_a_planted_datum_offset(self, tmp_path):
        """End-to-end version of the investigation: a station that drops
        160 hPa for three weeks must come back as a recoverable datum offset
        with a recommendation to calibrate rather than drop."""
        rows = []
        for s in ("AAA", "BBB"):
            for h in range(600):
                normal = 1005.0 + 0.8 * math.sin(h / 5.0)
                p = normal
                if s == "AAA" and 100 <= h < 400:
                    p = normal - 159.3
                rows.append(good_row(s, hour=h, pressure=p))
        rep = build(write_csv(tmp_path / "datum.csv", rows))
        p = rep["pressure_bounds_investigation"]
        assert p["rows"] == 300
        d = p["per_station"]["AAA"]
        assert d["verdict"] == "datum_offset", d["explanation"]
        assert d["recoverable_by_calibration"] is True
        assert d["clustering"]["verdict"] == "clustered_outage"
        assert d["clustering"]["longest_run_hours"] == 300
        assert p["population_census"]["populations"]["datum_offset"]["rows"] == 300
        assert "CALIBRATE" in _action_for(rep, "weather_bounds_pressure")

    def test_the_report_finds_a_planted_selection_bias(self, tmp_path):
        """The headline question: does dropping these rows move the retained
        set? Mirrors the real corpus: a cool, dry episode during which the
        station also loses its pressure datum, so the whole episode is
        quarantined and the retained set ends up warmer and more humid.
        """
        rows = []
        for h in range(800):
            cool = 200 <= h < 500
            rows.append(good_row("AAA", hour=h,
                                 temperature=(22.0 if cool else 28.0)
                                 + 1.2 * math.sin(h / 5.0),
                                 humidity=(58.0 if cool else 88.0)
                                 + 4.0 * math.sin(h / 5.0),
                                 pressure=(845.0 if cool else 1008.0)))
        rep = build(write_csv(tmp_path / "bias.csv", rows))
        e = rep["training_impact"]["by_reason"]["weather_bounds_pressure"]
        assert e["rows"] == 300
        assert e["bias_verdict"] == "material_selection_bias"
        assert e["bias_direction"] == "cooler and drier"
        assert set(e["biased_fields"]) == {"temperature", "humidity"}
        assert any("IS BIASED" in r for r in rep["data_health_verdict"]["reasons"])
        assert "cooler" in rep["data_health_verdict"]["reasons"][1]

    def test_the_retained_shape_phrase_inverts_the_measured_direction(self,
                                                                     tmp_path):
        """A sign error here would be believed, because 'warmer and drier'
        sounds entirely plausible for a tropical afternoon."""
        rows = []
        for h in range(800):
            cool = 200 <= h < 500
            rows.append(good_row("AAA", hour=h,
                                 temperature=(22.0 if cool else 28.0)
                                 + 1.2 * math.sin(h / 5.0),
                                 humidity=(58.0 if cool else 88.0)
                                 + 4.0 * math.sin(h / 5.0),
                                 pressure=(845.0 if cool else 1008.0)))
        rep = build(write_csv(tmp_path / "shape.csv", rows))
        summary = rep["data_health_verdict"]["summary"]
        assert "warmer" in summary and "more humid" in summary
        assert "cooler" not in summary and "drier" not in summary

    def test_a_completeness_outage_is_not_reported_as_a_bias(self, tmp_path):
        """A fifth of rows missing is a volume problem. Calling it a selection
        bias would send an operator looking for a weather-regime effect that
        is not there."""
        rows = [varied_row("AAA", h) for h in range(600)]
        rows += [varied_row("AAA", 100 + h, wind_speed="") for h in range(200)]
        rep = build(write_csv(tmp_path / "vol.csv", rows))
        e = rep["training_impact"]["by_reason"]["weather_missing_field"]
        assert e["bias_verdict"] == "no_material_selection_bias"
        assert abs(e["max_abs_effect_size"]) < 0.5
        assert e["companion_corruption"]["rate"] == 0.0

    def test_completeness_is_reported_per_field_per_station(self, tmp_path):
        rows = []
        for h in range(300):
            rows.append(good_row("AAA", hour=h))
            rows.append(good_row("BBB", hour=h, humidity="" if h < 150 else 70.0))
        rep = build(write_csv(tmp_path / "comp.csv", rows))
        a = rep["completeness_by_station"]["AAA"]["fields"]
        b = rep["completeness_by_station"]["BBB"]["fields"]
        assert set(a) == set(REQUIRED_FIELDS)
        assert a["humidity"]["present_pct"] == 100.0
        assert b["humidity"]["usable_pct"] == pytest.approx(50.0, abs=1.0)
        assert b["humidity"]["present_but_unusable_bins"] == 0

    def test_distributions_report_the_requested_statistics_and_bounds_hits(
            self, tmp_path):
        rows = [good_row("AAA", hour=h, temperature=20.0 + h * 0.05)
                for h in range(200)]
        rows += [good_row("AAA", hour=500 + h, temperature=104.0)
                 for h in range(3)]
        rep = build(write_csv(tmp_path / "dist.csv", rows))
        t = rep["distributions_by_station"]["AAA"]["fields"]["temperature"]
        for key in ("mean", "sd", "p1", "p50", "p99", "min", "max",
                    "out_of_bounds_count"):
            assert key in t, key
        assert t["out_of_bounds_count"] == 3
        assert t["out_of_bounds_pct"] > 0

    def test_continuity_is_measured_on_the_ingested_grid(self, tmp_path):
        """Quarantined hours are holes in the grid the model trains on, so
        continuity has to be measured after quarantine, not before."""
        rows = [good_row("AAA", hour=h) for h in range(100) if not 40 <= h < 60]
        rows += [good_row("AAA", hour=h, humidity="") for h in range(40, 60)]
        rep = build(write_csv(tmp_path / "cont.csv", rows))
        c = rep["continuity_by_station"]["AAA"]
        assert c["raw_bins_observed"] == 100
        assert c["usable_bins"] == 80
        assert c["longest_gap_hours"] == 21.0

    def test_impact_ranks_by_consequence_not_by_count(self, tmp_path):
        """A 13% unbiased outage must outrank a 66-row corrupt drop."""
        rows = [good_row("AAA", hour=h) for h in range(900)]
        rows += [good_row("AAA", hour=100 + h, humidity="")
                 for h in range(200)]          # 200 rows, no bias
        rows += [good_row("AAA", hour=400 + h, temperature=104.0)
                 for h in range(66)]           # 66 rows, all corrupt companions
        rep = build(write_csv(tmp_path / "rank.csv", rows))
        ranked = {f["problem"]: f for f in rep["impact_ranking"]}
        assert ranked["weather_missing_field"]["impact_score"] > \
            ranked["weather_bounds_temperature"]["impact_score"]
        assert [f["rank"] for f in rep["impact_ranking"]] == \
            list(range(1, len(rep["impact_ranking"]) + 1))
        assert rep["impact_method"]["weights"]["volume"] > 0

    def test_the_verdict_names_its_reasons(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(300)]
        rows += [good_row("AAA", hour=100 + h, humidity="") for h in range(150)]
        rep = build(write_csv(tmp_path / "verdict.csv", rows))
        v = rep["data_health_verdict"]
        assert v["verdict"] in ("GOOD", "FAIR", "POOR")
        assert v["reasons"], "a verdict with no reasons is an opinion"
        assert v["disclosure"]
        assert set(v["scale"]) == {"GOOD", "FAIR", "POOR"}

    def test_a_station_with_a_48_day_hole_is_surfaced(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(100)]
        rows += [good_row("AAA", hour=2000 + h) for h in range(100)]
        rep = build(write_csv(tmp_path / "hole.csv", rows))
        worst = rep["data_health_verdict"]["worst_continuity"]
        assert worst["station_id"] == "AAA"
        assert worst["longest_gap_hours"] == 1901.0
        assert any("hole in its ingested hourly grid" in r
                   for r in rep["data_health_verdict"]["reasons"])

    def test_an_under_determined_effect_is_flagged_as_not_supporting_a_claim(
            self, tmp_path):
        """A clamped effect size on twelve rows is still a real measurement,
        and it must not be readable as a supported bias claim."""
        rows = []
        for h in range(400):
            hot = 100 <= h < 112
            rows.append(good_row("AAA", hour=h,
                                 temperature=36.0 if hot else 26.0,
                                 heat_index=99.0 if hot else 30.0))
        rep = build(write_csv(tmp_path / "small.csv", rows))
        e = rep["training_impact"]["by_reason"]["weather_bounds_heat_index"]
        assert e["bias_verdict"] == "undetermined_cohort_too_small"
        assert e["max_abs_effect_size_supports_claim"] is False
        ranked = _ranking_for(rep, "weather_bounds_heat_index")
        assert ranked["impact_components"]["bias"] == 0.0

    def test_defects_are_reported_not_silently_fixed(self, tmp_path):
        rows = [good_row("AAA", hour=h, humidity="", temperature=104.0)
                for h in range(50)]
        rep = build(write_csv(tmp_path / "defect.csv", rows))
        defects = rep["defects_reported_not_fixed"]
        assert defects
        assert all("file" in x and "fix_not_applied" in x for x in defects)
        assert any("Completeness is tested before physical bounds" in x["what"]
                   for x in defects)

    def test_report_is_json_serialisable_and_free_of_absolute_paths(self, tmp_path):
        rows = [good_row("AAA%03d" % s, hour=h) for s in range(4)
                for h in range(120)]
        rep = build(write_csv(tmp_path / "json.csv", rows))
        blob = json.dumps(rep, indent=2, default=str)
        for bad in ("C:\\", "C:/", "/home/", "/Users/"):
            assert bad not in blob, "path hygiene gate would fail on %r" % bad
        assert rep["provenance"]["source_file"] == "json.csv"

    def test_all_sections_present(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(60)]
        rep = build(write_csv(tmp_path / "sections.csv", rows))
        for key in ("schema", "generated_at", "provenance", "row_accounting",
                    "quarantine_detail", "pressure_bounds_investigation",
                    "completeness_by_station", "distributions_by_station",
                    "continuity_by_station", "training_impact",
                    "impact_method", "impact_ranking", "data_health_verdict",
                    "defects_reported_not_fixed"):
            assert key in rep, key

    def test_build_is_deterministic(self, tmp_path):
        rows = [good_row("AAA%03d" % s, hour=h) for s in range(3)
                for h in range(80)]
        path = write_csv(tmp_path / "det.csv", rows)
        assert json.dumps(build(path), sort_keys=True) == \
            json.dumps(build(path), sort_keys=True)

    def test_build_writes_nothing(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(40)]
        path = write_csv(tmp_path / "quiet.csv", rows)
        before = sorted(os.listdir(tmp_path))
        build(path)
        assert sorted(os.listdir(tmp_path)) == before

    def test_an_empty_corpus_does_not_crash(self, tmp_path):
        path = write_csv(tmp_path / "empty.csv", [])
        rep = build(path)
        assert rep["row_accounting"]["rows_in"] == 0
        assert rep["data_health_verdict"]["verdict"] == "GOOD"

    def test_a_single_station_with_no_rejections_has_no_pressure_section_rows(
            self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(50)]
        rep = build(write_csv(tmp_path / "nop.csv", rows))
        assert rep["pressure_bounds_investigation"]["rows"] == 0


class TestCli:
    def test_no_write_leaves_the_output_untouched(self, tmp_path, capsys):
        rows = [good_row("AAA", hour=h) for h in range(60)]
        src = write_csv(tmp_path / "in.csv", rows)
        out = tmp_path / "out.json"
        rc = main(["--csv", src, "--out", str(out), "--no-write", "--no-git"])
        assert rc == 0
        assert not out.exists()
        assert "DATA QUALITY REPORT" in capsys.readouterr().out

    def test_write_produces_parseable_json(self, tmp_path):
        rows = [good_row("AAA", hour=h) for h in range(60)]
        src = write_csv(tmp_path / "in.csv", rows)
        out = tmp_path / "out.json"
        assert main(["--csv", src, "--out", str(out), "--no-git"]) == 0
        rep = json.loads(out.read_text(encoding="utf-8"))
        assert rep["row_accounting"]["rows_in"] == 60
        assert rep["code_commit"] == "not_recorded"

    def test_missing_csv_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--csv", str(tmp_path / "nope.csv"), "--no-write", "--no-git"])


def _action_for(rep, reason):
    return _ranking_for(rep, reason)["recommended_action"]


def _ranking_for(rep, reason):
    for f in rep["impact_ranking"]:
        if f["problem"] == reason:
            return f
    raise AssertionError("no ranking entry for %r" % reason)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
