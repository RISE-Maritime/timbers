"""Scoring under a shaft-power ceiling: speed inversion and forward integration.

Properties under test, in increasing order of complexity:
- the inversion matches the closed form where one exists (calm water);
- the inverted speed respects the ceiling and is the largest that does;
- with no ceiling the forward integrator reproduces the fixed-schedule scorer;
- a lower ceiling never gives an earlier arrival;
- ``max_hours`` stops the voyage and reports where the ship got to.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_np  # noqa: E402

from timbers import optimizer as op  # noqa: E402
from timbers.scoring import (
    evaluate_route_full,
    evaluate_route_saturated,  # noqa: E402
    v_max_for_power,
)

COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)
K, L, NSP = 6, 40, 4
DEP = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _grids(swh=1.5):
    lat = np.arange(35.0, 50.001, 0.5)
    lon = np.arange(-75.0, 5.001, 0.5)
    nt = 200
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(nt) * np.timedelta64(1, "h")
    shape = (nt, lat.size, lon.size)
    tt = np.arange(nt)[:, None, None]
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {
        **base,
        "u10": (5.0 + 3.0 * np.sin(0.1 * tt)) * np.ones(shape, np.float32),
        "v10": np.full(shape, -4.0, np.float32),
    }
    wave = {
        **base,
        "swh": np.full(shape, swh, np.float32),
        "mwd": np.full(shape, 200.0, np.float32),
    }
    return wind, wave


def _route():
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    return (
        np.asarray(lat, float),
        np.asarray(op.working_to_signed(wlon), float),
        np.asarray(seg, float),
    )


# --- speed inversion ------------------------------------------------------------
@pytest.mark.parametrize("p_max", [50.0, 500.0, 1500.0])
def test_inversion_matches_closed_form_in_calm_water(p_max):
    """Toy model in calm water is 5 v^3, so v = (p / 5)^(1/3)."""
    v = float(v_max_for_power(toy_power_np, 0.0, 0.0, 0.0, 0.0, False, p_max))
    assert v == pytest.approx((p_max / 5.0) ** (1 / 3), rel=1e-6)


def test_inversion_respects_the_ceiling_and_is_largest():
    rng = np.random.default_rng(0)
    n = 1000
    tws, twa = rng.uniform(0, 25, n), rng.uniform(0, 360, n)
    hs, mwa = rng.uniform(0, 8, n), rng.uniform(0, 360, n)
    v = v_max_for_power(toy_power_np, tws, twa, hs, mwa, True, 800.0)
    assert np.all(toy_power_np(tws, twa, hs, mwa, v, True) <= 800.0 + 1e-6)
    room = v < 20.0 - 1e-3
    assert np.all(
        toy_power_np(tws[room], twa[room], hs[room], mwa[room], v[room] + 1e-3, True)
        >= 800.0 - 1e-3
    )


# --- forward integration --------------------------------------------------------
def test_no_ceiling_reproduces_the_fixed_schedule():
    wind, wave = _grids()
    lat, lon, seg = _route()
    r = evaluate_route_saturated(wind, wave, DEP, lat, lon, seg, toy_power_np)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    wps = [
        (DEP.replace(tzinfo=None) + timedelta(hours=float(h)), a, b) for h, a, b in zip(t, lat, lon)
    ]
    ref = evaluate_route_full(wind, wave, wps, toy_power_np)
    assert r["arrived"] and abs(r["delay_h"]) < 1e-3  # float32 durations
    assert r["saturated_frac"] == 0.0
    assert r["energy_mwh"] == pytest.approx(ref["energy_mwh"], rel=1e-3)


def test_lower_ceiling_never_arrives_earlier():
    wind, wave = _grids(swh=3.0)
    lat, lon, seg = _route()
    arrivals = [
        evaluate_route_saturated(wind, wave, DEP, lat, lon, seg, toy_power_np, p_max=p)[
            "actual_hours"
        ]
        for p in (np.inf, 20000.0, 8000.0, 4000.0)
    ]
    assert all(b >= a - 1e-9 for a, b in zip(arrivals, arrivals[1:]))
    assert arrivals[-1] > arrivals[0] + 1.0  # the lowest ceiling binds


def test_max_hours_stops_and_reports_position():
    wind, wave = _grids()
    lat, lon, seg = _route()
    r = evaluate_route_saturated(wind, wave, DEP, lat, lon, seg, toy_power_np, max_hours=12.0)
    assert r["actual_hours"] == pytest.approx(12.0)
    assert not r["arrived"] and 0.0 < r["fraction_done"] < 1.0
    assert r["track"]["t_h"][-1] == pytest.approx(12.0)
    assert (r["end_lat"], r["end_lon"]) == (r["track"]["lat"][-1], r["track"]["lon"][-1])


def test_pieces_report_arrival_on_the_voyage_clock():
    """Sailed in two pieces, the second piece's arrival matches one piece."""
    wind, wave = _grids()
    lat, lon, seg = _route()
    whole = evaluate_route_saturated(wind, wave, DEP, lat, lon, seg, toy_power_np)
    first = evaluate_route_saturated(wind, wave, DEP, lat, lon, seg, toy_power_np, max_hours=12.0)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    j = int(np.searchsorted(t, 12.0))
    rest_lat = np.r_[first["end_lat"], lat[j:]]
    rest_lon = np.r_[first["end_lon"], lon[j:]]
    rest_seg = np.r_[t[j] - 12.0, seg[j:]]
    second = evaluate_route_saturated(
        wind, wave, DEP, rest_lat, rest_lon, rest_seg, toy_power_np, t_offset_h=12.0
    )
    assert second["arrival"] == DEP + timedelta(hours=12.0 + second["actual_hours"])
    assert abs((second["arrival"] - whole["arrival"]).total_seconds()) < 60.0


def test_antimeridian_crossing_samples_the_right_weather():
    """A Pacific route across 180 on a 0-360 grid sees the calm band it sails
    through, as the fixed-schedule scorer does, and its track stays near 180."""
    lat_ax = np.arange(30.0, 40.001, 0.5)
    lon_ax = np.arange(140.0, 230.001, 0.5)
    nt = 120
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(nt) * np.timedelta64(1, "h")
    shape = (nt, lat_ax.size, lon_ax.size)
    swh = np.where((lon_ax >= 150.0) & (lon_ax <= 210.0), 1.0, 8.0) * np.ones(shape)
    base = dict(lat=lat_ax, lon=lon_ax, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 5.0, np.float32), "v10": np.full(shape, -4.0, np.float32)}
    wave = {**base, "swh": swh.astype(np.float32), "mwd": np.full(shape, 200.0, np.float32)}
    cor = op.Corridor("pacific", 35.0, 170.0, 35.0, 200.0, 96.0)
    lat, wlon, seg = op.decode_route(op.gc_init_theta(cor, K, NSP), cor, K, L, NSP)
    lat, lon, seg = np.asarray(lat, float), op.working_to_signed(np.asarray(wlon, float)), seg
    r = evaluate_route_saturated(wind, wave, DEP, lat, lon, seg, toy_power_np)
    tt = np.concatenate([[0.0], np.cumsum(seg)])
    ref = evaluate_route_full(
        wind,
        wave,
        [
            (DEP.replace(tzinfo=None) + timedelta(hours=float(h)), a, b)
            for h, a, b in zip(tt, lat, lon)
        ],
        toy_power_np,
    )
    assert ref["max_hs_m"] == pytest.approx(1.0)
    assert r["max_hs_m"] == pytest.approx(1.0)
    assert r["energy_mwh"] == pytest.approx(ref["energy_mwh"], rel=1e-3)
    trk = np.asarray(r["track"]["lon"])
    assert np.all((trk >= 170.0 - 1e-6) | (trk <= -160.0 + 1e-6))
    assert r["end_lon"] == pytest.approx(-160.0, abs=1e-3)
