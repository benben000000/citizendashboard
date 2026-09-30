"""
Guard on station metadata.

`station_coords.json` shipped with latitude and longitude transposed -- every
entry carried a longitude under the key "lat". Nothing caught it: haversine
distance is symmetric under a swap, so the neighbour analysis still produced
sensible-looking results, and the values were still "near the Philippines". The
bug only surfaced when a live GFS grid sample came back as -9.68 degC, because
the provider was handed latitude 121.

These tests exist so that class of silent corruption cannot recur.
"""

import json
import os

import pytest

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
COORDS = os.path.join(DATA_DIR, "station_coords.json")

pytestmark = pytest.mark.skipif(not os.path.exists(COORDS),
                                reason="station_coords.json not present")


@pytest.fixture(scope="module")
def coords():
    with open(COORDS, "r", encoding="utf-8") as f:
        return json.load(f)


def test_latitudes_are_physically_valid(coords):
    bad = {k: v for k, v in coords.items()
           if not (-90.0 <= float(v["lat"]) <= 90.0)}
    assert not bad, (
        f"{len(bad)} station(s) have an impossible latitude, which means lat and "
        f"lon are transposed: "
        + ", ".join(f"{k}(lat={v['lat']})" for k, v in list(bad.items())[:5]))


def test_longitudes_are_physically_valid(coords):
    bad = {k: v for k, v in coords.items()
           if not (-180.0 <= float(v["lon"]) <= 360.0)}
    assert not bad, f"impossible longitude: {list(bad)[:5]}"


def test_stations_are_in_the_operational_area(coords):
    """The fleet is in Luzon, Philippines. Catches a wholesale axis swap."""
    for k, v in coords.items():
        assert 4.0 <= float(v["lat"]) <= 22.0, f"{k} latitude {v['lat']} outside Luzon"
        assert 118.0 <= float(v["lon"]) <= 123.0, f"{k} longitude {v['lon']} outside Luzon"


def test_every_entry_has_a_name(coords):
    missing = [k for k, v in coords.items() if not v.get("name")]
    assert not missing, f"stations without a name: {missing[:5]}"


def test_coordinates_are_not_all_identical(coords):
    """A degenerate file would pass the range checks but be useless."""
    lats = {round(float(v["lat"]), 4) for v in coords.values()}
    lons = {round(float(v["lon"]), 4) for v in coords.values()}
    assert len(lats) > 1, "every station shares one latitude"
    assert len(lons) > 1, "every station shares one longitude"
