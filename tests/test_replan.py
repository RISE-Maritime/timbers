"""Re-planning loop and track fitting on synthetic weather.

Properties under test:
- fitting a decoded route recovers its geometry and schedule;
- with no power ceiling and a perfect forecast the voyage arrives on time, one
  plan per cycle, each asked for the remaining horizon at the cycle time;
- one grid spanning the voyage plans as a forecast cut to each cycle does;
- under a binding ceiling the ship falls behind and each plan is given the time
  left to the original arrival (floored at the calm-water passage time);
- the promise counts members that breach a limit over the sailed window, which
  ends where the plan puts the ship at the end of the cycle;
- a leg that stalls short of its cycle leaves the next plan on the latest cycle
  issued by the ship's clock;
- a voyage across 180 on a 0-360 grid takes the short way;
- a ceiling too low to move the ship is rejected.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax, toy_power_np  # noqa: E402

from timbers import optimizer as op  # noqa: E402
from timbers import replan  # noqa: E402
from timbers.geo import haversine_m  # noqa: E402
from timbers.model import Grids  # noqa: E402
from timbers.scoring import v_max_for_power  # noqa: E402

ORIGIN, DEST, HOURS = (43.6, -4.0), (42.0, -14.0), 48.0
DEP = datetime(2024, 1, 1)
LIMITS = dict(hs_lim=7.0, tws_lim=20.0)
SMALL = dict(K=4, L=20, n_speed=4, cycle_h=12.0, seeds=1, iters=20, pop=16)


def _truth(swh=1.5, swell=0.0, nt=120):
    """Uniform weather; ``swell`` adds a daily cycle to the wave height."""
    lat = np.arange(35.0, 50.001, 0.5)
    lon = np.arange(-20.0, 2.001, 0.5)
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(nt) * np.timedelta64(1, "h")
    shape = (nt, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 5.0, np.float32), "v10": np.full(shape, -4.0, np.float32)}
    wave = {
        **base,
        "swh": (swh + swell * np.sin(2 * np.pi * np.arange(nt) / 24.0))[:, None, None]
        * np.ones(shape, np.float32),
        "mwd": np.full(shape, 260.0, np.float32),
    }
    return wind, wave


def _land():
    llat, lwlon = np.arange(35.0, 50.001, 1.0), np.arange(-20.0, 2.001, 1.0)
    return op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )


def _nm(lat0, lon0, lat1, lon1):
    return float(haversine_m(lat0, lon0, lat1, lon1)) / 1852.0


def _sail(swh=1.5, p_max=np.inf, whole=False, swell=0.0, **kw):
    wind, wave = _truth(swh, swell)
    calls = []

    def forecast(t, hours):
        calls.append((t, hours))
        return Grids.from_era5(wind, wave) if whole else Grids.from_era5(wind, wave, t, hours)

    out = replan.sail_with_replanning(
        ORIGIN,
        DEST,
        HOURS,
        DEP,
        forecast,
        wind,
        wave,
        _land(),
        objective="deterministic",
        power_fn=toy_power_jax,
        power_fn_host=toy_power_np,
        p_max=p_max,
        **LIMITS,
        **SMALL,
        **kw,
    )
    return out, calls


def _along_nm(lat, lon):
    return np.r_[
        0.0,
        np.cumsum([_nm(lat[i], lon[i], lat[i + 1], lon[i + 1]) for i in range(len(lat) - 1)]),
    ]


def test_fit_recovers_a_decoded_route():
    """Same curve within a nautical mile; passage times along it within 5 %."""
    cor = op.Corridor("fit", *ORIGIN, *DEST, HOURS)
    rng = np.random.default_rng(0)
    K, L, NSP = 6, 200, 5
    theta = (
        op.gc_init_theta(cor, K, NSP)
        + np.r_[rng.normal(0, 0.05, 2 * (K - 2)), rng.normal(0, 0.3, NSP)]
    )
    lat, wlon, seg = op.decode_route(theta, cor, K, L, NSP)
    t = np.r_[0.0, np.cumsum(seg)]
    fit, diag = op.fit_theta_to_track(
        cor, t, lat, op.working_to_signed(wlon), K=K, L=L, n_speed=NSP
    )
    assert diag["resid_nm_max"] < 1.0
    f_lat, f_wlon, f_seg = op.decode_route(fit, cor, K, L, NSP)
    s, f_s = _along_nm(lat, wlon), _along_nm(f_lat, f_wlon)
    t_true = np.interp(f_s, s, t)
    assert np.max(np.abs(np.r_[0.0, np.cumsum(f_seg)] - t_true)) < 0.05 * HOURS


def test_perfect_forecast_without_ceiling_arrives_on_time():
    out, calls = _sail(record_promise=True)
    assert out["arrived"]
    assert out["actual_hours"] == pytest.approx(HOURS, abs=1e-3)
    assert out["n_legs"] == int(HOURS / SMALL["cycle_h"])
    assert [c[0] for c in calls] == [
        DEP + timedelta(hours=k * SMALL["cycle_h"]) for k in range(out["n_legs"])
    ]
    assert [c[1] for c in calls] == pytest.approx(
        [HOURS - k * SMALL["cycle_h"] for k in range(out["n_legs"])], abs=1e-3
    )
    assert out["energy_mwh"] == pytest.approx(sum(g["energy_mwh"] for g in out["legs"]))
    assert np.all(np.diff(out["track"]["t_h"]) > 0)
    assert _nm(out["track"]["lat"][-1], out["track"]["lon"][-1], *DEST) < 1.0
    assert all(g["p_hat"] == 0.0 for g in out["legs"])


def test_arrival_time_does_not_move_when_behind():
    p_max = 300.0
    out, _ = _sail(p_max=p_max)
    assert out["delay_h"] > 1.0
    assert out["max_power_kw"] <= p_max + 1e-6
    v_calm = float(v_max_for_power(toy_power_np, 0.0, 0.0, 0.0, 0.0, False, p_max))
    for prev, g in zip(out["legs"], out["legs"][1:]):
        floor_h = _nm(prev["end_lat"], prev["end_lon"], *DEST) * 1852.0 / (v_calm * 3600.0)
        assert g["planned_hours"] == pytest.approx(
            max(HOURS - prev["elapsed_h"], floor_h), rel=1e-9
        )


def test_promise_counts_breaching_members():
    out, _ = _sail(swh=9.0, record_promise=True)
    assert out["max_hs_m"] > LIMITS["hs_lim"]
    assert all(g["p_hat"] == 1.0 for g in out["legs"])


def test_one_grid_for_the_whole_voyage_plans_like_cut_forecasts():
    cut, _ = _sail(p_max=300.0, swell=1.5)
    whole, _ = _sail(p_max=300.0, swell=1.5, whole=True)
    assert whole["n_legs"] == cut["n_legs"]
    assert whole["energy_mwh"] == pytest.approx(cut["energy_mwh"], rel=1e-4)
    assert whole["actual_hours"] == pytest.approx(cut["actual_hours"], rel=1e-4)


def test_sailed_window_ends_where_the_plan_puts_the_ship():
    lat, wlon, seg = np.zeros(3), np.array([0.0, 1.0, 2.0]), np.array([10.0, 10.0])
    w_lat, w_wlon, sub = replan._sailed_window(lat, wlon, seg, 15.0)
    np.testing.assert_allclose(w_wlon, [0.0, 1.0, 1.5])
    np.testing.assert_allclose(sub, [10.0, 5.0])
    np.testing.assert_allclose(w_lat, 0.0)


def test_stalled_leg_keeps_plans_on_the_ships_clock():
    """Forecast calm, truth rough, low ceiling: late legs stall short of their
    cycle, and every plan uses the cycle issued last before it starts."""
    fwind, fwave = _truth(1.5, nt=400)
    twind, twave = _truth(5.0, nt=400)
    calls = []

    def forecast(t, hours):
        calls.append((t, hours))
        return Grids.from_era5(fwind, fwave, t, hours)

    out = replan.sail_with_replanning(
        ORIGIN,
        DEST,
        HOURS,
        DEP,
        forecast,
        twind,
        twave,
        _land(),
        objective="deterministic",
        power_fn=toy_power_jax,
        power_fn_host=toy_power_np,
        p_max=300.0,
        **LIMITS,
        **SMALL,
    )
    assert out["arrived"]
    cyc = SMALL["cycle_h"]
    starts = [0.0] + [g["elapsed_h"] for g in out["legs"][:-1]]
    stalled = [
        g for g, t0 in zip(out["legs"][:-1], starts) if g["sailed_hours"] < cyc - (t0 % cyc) - 1e-6
    ]
    assert stalled
    for (cycle_t, hours), g, t0 in zip(calls, out["legs"], starts):
        into = t0 - (cycle_t - DEP).total_seconds() / 3600.0
        assert -1e-6 <= into < cyc
        assert hours == pytest.approx(into + g["planned_hours"])
        assert g["sailed_hours"] <= cyc - into + 1e-6


def test_voyage_across_180_takes_the_short_way():
    lat = np.arange(30.0, 50.001, 0.5)
    lon = np.arange(140.0, 230.001, 0.5)
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(120) * np.timedelta64(1, "h")
    shape = (t.size, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 5.0, np.float32), "v10": np.full(shape, -4.0, np.float32)}
    wave = {
        **base,
        "swh": np.full(shape, 1.5, np.float32),
        "mwd": np.full(shape, 260.0, np.float32),
    }
    llat, lwlon = np.arange(30.0, 50.001, 1.0), np.arange(160.0, 205.001, 1.0)
    land = op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )
    origin, dest = (40.0, 170.0), (42.0, -165.0)
    out = replan.sail_with_replanning(
        origin,
        dest,
        48.0,
        DEP,
        lambda c, h: Grids.from_era5(wind, wave, c, h),
        wind,
        wave,
        land,
        objective="deterministic",
        power_fn=toy_power_jax,
        power_fn_host=toy_power_np,
        **LIMITS,
        **SMALL,
    )
    assert out["arrived"]
    assert out["actual_hours"] == pytest.approx(48.0, abs=1e-3)
    trk_lat, trk_lon = out["track"]["lat"], out["track"]["lon"]
    sailed = sum(
        _nm(trk_lat[i], trk_lon[i], trk_lat[i + 1], trk_lon[i + 1]) for i in range(len(trk_lat) - 1)
    )
    assert sailed < 1.05 * _nm(*origin, *dest)
    assert all(x >= 169.0 or x <= -164.0 for x in trk_lon)


def test_ceiling_too_low_to_move_is_rejected():
    with pytest.raises(ValueError, match="cannot drive"):
        _sail(p_max=0.0)
