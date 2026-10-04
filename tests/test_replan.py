"""Re-planning loop and track fitting on synthetic weather.

Properties under test:
- fitting a decoded route recovers its geometry and schedule;
- with no power ceiling and a perfect forecast the voyage arrives on time, one
  plan per cycle, each asked for the remaining horizon at the cycle time;
- under a binding ceiling the ship falls behind and each plan is given the time
  left to the original arrival (floored at the calm-water passage time);
- the promise counts members that breach a limit over the sailed window.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax, toy_power_np  # noqa: E402

from timbers import ensemble as te  # noqa: E402
from timbers import optimizer as op  # noqa: E402
from timbers import replan  # noqa: E402
from timbers.geo import gc_distance_nm  # noqa: E402
from timbers.scoring import v_max_for_power  # noqa: E402

ORIGIN, DEST, HOURS = (43.6, -4.0), (42.0, -14.0), 48.0
DEP = datetime(2024, 1, 1)
LIMITS = dict(hs_lim=7.0, tws_lim=20.0)
SMALL = dict(K=4, L=20, n_speed=4, cycle_h=12.0, seeds=1, iters=20, pop=16)


def _truth(swh=1.5, nt=120):
    lat = np.arange(35.0, 50.001, 0.5)
    lon = np.arange(-20.0, 2.001, 0.5)
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(nt) * np.timedelta64(1, "h")
    shape = (nt, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 5.0, np.float32), "v10": np.full(shape, -4.0, np.float32)}
    wave = {**base, "swh": np.full(shape, swh, np.float32), "mwd": np.full(shape, 260.0, np.float32)}
    return wind, wave


def _land():
    llat, lwlon = np.arange(35.0, 50.001, 1.0), np.arange(-20.0, 2.001, 1.0)
    return op.DeviceLand({"lat": llat, "wlon": lwlon,
                          "mask": np.zeros((llat.size, lwlon.size), np.float32)})


def _sail(swh=1.5, p_max=np.inf, **kw):
    wind, wave = _truth(swh)
    calls = []

    def forecast(t, hours):
        calls.append((t, hours))
        return te.as_ensemble(wind, wave, t, hours)

    out = replan.sail_with_replanning(ORIGIN, DEST, HOURS, DEP, forecast, wind, wave, _land(),
                                      objective="deterministic", power_fn=toy_power_jax,
                                      power_fn_host=toy_power_np, p_max=p_max, **LIMITS,
                                      **SMALL, **kw)
    return out, calls


def _along_nm(lat, lon):
    return np.r_[0.0, np.cumsum([gc_distance_nm(lat[i], lon[i], lat[i + 1], lon[i + 1])
                                 for i in range(len(lat) - 1)])]


def test_fit_recovers_a_decoded_route():
    """Same curve within a nautical mile; passage times along it within 5 %."""
    cor = op.Corridor("fit", *ORIGIN, *DEST, HOURS)
    rng = np.random.default_rng(0)
    K, L, NSP = 6, 200, 5
    theta = op.gc_init_theta(cor, K, NSP) + np.r_[rng.normal(0, 0.05, 2 * (K - 2)),
                                                  rng.normal(0, 0.3, NSP)]
    lat, wlon, seg = op.decode_route(theta, cor, K, L, NSP)
    t = np.r_[0.0, np.cumsum(seg)]
    fit, diag = op.fit_theta_to_track(cor, t, lat, op.working_to_signed(wlon), K=K, L=L, n_speed=NSP)
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
    assert [c[0] for c in calls] == [DEP + timedelta(hours=k * SMALL["cycle_h"])
                                     for k in range(out["n_legs"])]
    assert [c[1] for c in calls] == pytest.approx([HOURS - k * SMALL["cycle_h"]
                                                   for k in range(out["n_legs"])], abs=1e-3)
    assert out["energy_mwh"] == pytest.approx(sum(g["energy_mwh"] for g in out["legs"]))
    assert np.all(np.diff(out["track"]["t_h"]) > 0)
    assert gc_distance_nm(out["track"]["lat"][-1], out["track"]["lon"][-1], *DEST) < 1.0
    assert all(g["p_hat"] == 0.0 for g in out["legs"])


def test_arrival_time_does_not_move_when_behind():
    p_max = 300.0
    out, _ = _sail(p_max=p_max)
    assert out["delay_h"] > 1.0
    assert out["max_power_kw"] <= p_max + 1e-6
    v_calm = float(v_max_for_power(toy_power_np, 0.0, 0.0, 0.0, 0.0, False, p_max))
    for prev, g in zip(out["legs"], out["legs"][1:]):
        floor_h = gc_distance_nm(prev["end_lat"], prev["end_lon"], *DEST) * 1852.0 / (v_calm * 3600.0)
        assert g["planned_hours"] == pytest.approx(max(HOURS - prev["elapsed_h"], floor_h), rel=1e-9)


def test_promise_counts_breaching_members():
    out, _ = _sail(swh=9.0, record_promise=True)
    assert out["max_hs_m"] > LIMITS["hs_lim"]
    assert all(g["p_hat"] == 1.0 for g in out["legs"])
