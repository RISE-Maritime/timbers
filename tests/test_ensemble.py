"""Ensemble cost and scoring on a synthetic ensemble.

Properties under test:
- time interpolation follows a non-uniform step vector;
- non-finite fields are rejected at construction;
- a single gridded field wraps as a one-member ensemble;
- the objectives reduce members as documented (member 0 against the mean;
  identical members make all four agree);
- the breach probability of a route counts the members that breach;
- the per-segment series integrates to the route energy;
- the cost drives the optimizer.
"""

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax  # noqa: E402

from timbers import cmaes as jc  # noqa: E402
from timbers import ensemble as te  # noqa: E402
from timbers import model as jm  # noqa: E402
from timbers import optimizer as op  # noqa: E402
from timbers.model import Grids  # noqa: E402

COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)
K, L, NSP, ALIGN = 6, 40, 4, 0.25
LAT = np.arange(35.0, 50.001, 0.5)
LON = np.arange(-75.0, 5.001, 0.5)
STEPS = np.array([0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60], np.float32)
LIMITS = dict(hs_lim=7.0, tws_lim=20.0)


def _members(swh):
    """Calm, uniform members; ``swh`` is one value per member."""
    shape = (len(swh), STEPS.size, LAT.size, LON.size)
    wind = dict(
        lat=LAT, lon=LON, u10=np.full(shape, 4.0, np.float32), v10=np.full(shape, -2.0, np.float32)
    )
    wave = dict(
        lat=LAT,
        lon=LON,
        mwd=np.full(shape, 200.0, np.float32),
        swh=np.asarray(swh, np.float32)[:, None, None, None] * np.ones(shape, np.float32),
    )
    return wind, wave


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    return op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )


def _fit(grids, objective, **kw):
    fit, shared = te.make_ensemble_cost(
        grids,
        _land(),
        COR,
        objective=objective,
        L=L,
        K=K,
        n_speed=NSP,
        align=ALIGN,
        wps=False,
        power_fn=toy_power_jax,
        **LIMITS,
        **kw,
    )
    theta = jnp.asarray(op.gc_init_theta(COR, K, NSP))[None, :]
    return float(fit(theta, (jnp.float32(0.0), *shared))[0])


def _gc_members(grids):
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    return te.score_members(
        grids, COR, lat, wlon, seg, wps=False, power_fn=toy_power_jax, align=ALIGN, **LIMITS
    )


def test_time_index_on_non_uniform_steps():
    ti, tf = jm.time_index(jnp.asarray([9.0, 15.0, 0.0]), jnp.asarray(STEPS))
    assert list(np.asarray(ti)) == [3, 4, 0]
    np.testing.assert_allclose(np.asarray(tf), [0.0, 0.5, 0.0], atol=1e-6)


def test_non_finite_fields_are_rejected():
    wind, wave = _members([1.0, 1.0])
    wave["swh"][0, 0, 3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Grids(wind, wave, STEPS)


def test_single_field_wraps_as_one_member():
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(100) * np.timedelta64(1, "h")
    shape = (100, LAT.size, LON.size)
    base = dict(lat=LAT, lon=LON, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.ones(shape, np.float32), "v10": np.ones(shape, np.float32)}
    wave = {**base, "swh": np.ones(shape, np.float32), "mwd": np.zeros(shape, np.float32)}
    g = Grids.from_era5(wind, wave, datetime(2024, 1, 1, 6), 48.0)
    assert g.n_members == 1
    np.testing.assert_allclose(np.asarray(g.steps)[:3], [0.0, 1.0, 2.0])
    assert g.nt == 51


def test_objectives_reduce_members_as_documented():
    """Calm weather, no penalties active: deterministic is member 0's energy and
    expected-value the member mean; identical members make all four agree."""
    g = Grids(*_members([0.5, 2.0, 3.0, 4.0]), STEPS)
    e = _gc_members(g)["energy_mwh"]
    assert _fit(g, "deterministic") == pytest.approx(e[0], rel=1e-5)
    assert _fit(g, "expected_value") == pytest.approx(e.mean(), rel=1e-5)
    same = Grids(*_members([2.0] * 4), STEPS)
    vals = [_fit(same, o) for o in te.OBJECTIVES]
    assert max(vals) == pytest.approx(min(vals), rel=1e-6)


def test_breach_probability_counts_members():
    g = Grids(*_members([1.0, 1.0, 9.0, 9.0]), STEPS)  # two members above 7 m
    m = _gc_members(g)
    assert (m["margin"] > 0).mean() == pytest.approx(0.5)
    # The chance constraint sees the two stormy members; the deterministic arm,
    # reading member 0, does not.
    assert _fit(g, "chance_constrained") > _fit(g, "deterministic") + 1.0


def test_series_integrates_to_route_energy():
    g = Grids(*_members([1.0, 2.0]), STEPS)
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seg))])
    s = te.member_series(g, t, lat, op.working_to_signed(wlon), wps=False, power_fn=toy_power_jax)
    assert s["valid"].all()
    e_series = (s["power_kw"] * s["seg_h"]).sum(axis=1) / 1000.0
    np.testing.assert_allclose(e_series, _gc_members(g)["energy_mwh"], rtol=0.05)


def test_cost_drives_the_optimizer():
    g = Grids(*_members([1.0, 3.0, 5.0, 8.0]), STEPS)
    fit, shared = te.make_ensemble_cost(
        g,
        _land(),
        COR,
        objective="joint",
        L=L,
        K=K,
        n_speed=NSP,
        align=ALIGN,
        wps=False,
        power_fn=toy_power_jax,
        **LIMITS,
    )
    x0 = jnp.asarray(op.gc_init_theta(COR, K, NSP))
    cargs = (jnp.float32(0.0), *shared)
    j0 = float(fit(x0[None, :], cargs)[0])
    hp = jc.hyperparams(x0.size, 16)
    _, best = jc.run(x0, fit, 0.1, hp, 16, 30, jax.random.PRNGKey(0), cargs)
    assert np.isfinite(float(best)) and float(best) <= j0 + 1e-3


def _hourly(n=100, lat=LAT, lon=LON, swh=None):
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(n) * np.timedelta64(1, "h")
    shape = (n, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 4.0, np.float32), "v10": np.full(shape, -2.0, np.float32)}
    swh = (
        np.ones(shape, np.float32)
        if swh is None
        else np.broadcast_to(swh, shape).astype(np.float32)
    )
    wave = {**base, "swh": swh, "mwd": np.full(shape, 200.0, np.float32)}
    return wind, wave


def test_single_field_must_cover_the_window_on_its_steps():
    wind, wave = _hourly(60)
    with pytest.raises(ValueError, match="ends before"):
        Grids.from_era5(wind, wave, datetime(2024, 1, 1, 6), 60.0)
    with pytest.raises(ValueError, match="not on a time step"):
        Grids.from_era5(wind, wave, datetime(2024, 1, 1, 6, 20), 24.0)


def test_antimeridian_midpoints_sample_the_weather_at_180():
    """A Pacific route across 180 on a 0-360 grid, inside a calm band, gives the
    same cost, member outcomes and series as an all-calm world."""
    lat = np.arange(30.0, 40.001, 0.5)
    lon = np.arange(140.0, 230.001, 0.5)
    band = np.where((lon >= 150.0) & (lon <= 210.0), 1.0, 8.0)
    calm = Grids.from_era5(*_hourly(lat=lat, lon=lon), datetime(2024, 1, 1), 60.0)
    banded = Grids.from_era5(*_hourly(lat=lat, lon=lon, swh=band), datetime(2024, 1, 1), 60.0)
    cor = op.Corridor("pacific", 35.0, 170.0, 35.0, 200.0, 48.0)
    llat, lwlon = np.arange(30.0, 40.001, 1.0), np.arange(140.0, 230.001, 1.0)
    land = op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )
    theta = jnp.asarray(op.gc_init_theta(cor, K, NSP))[None, :]
    lat_r, wlon_r, seg = op.decode_route(np.asarray(theta[0]), cor, K, L, NSP)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seg))])
    for objective in ("deterministic", "chance_constrained"):
        costs = []
        for g in (calm, banded):
            fit, shared = te.make_ensemble_cost(
                g,
                land,
                cor,
                objective=objective,
                L=L,
                K=K,
                n_speed=NSP,
                align=ALIGN,
                wps=False,
                power_fn=toy_power_jax,
                **LIMITS,
            )
            costs.append(float(fit(theta, (jnp.float32(0.0), *shared))[0]))
        assert costs[1] == pytest.approx(costs[0], rel=1e-6)
    m = te.score_members(
        banded, cor, lat_r, wlon_r, seg, wps=False, power_fn=toy_power_jax, align=ALIGN, **LIMITS
    )
    assert m["max_hs"][0] == pytest.approx(1.0)
    s = te.member_series(
        banded, t, lat_r, op.working_to_signed(wlon_r), wps=False, power_fn=toy_power_jax
    )
    assert np.max(s["swh"][:, s["valid"]]) == pytest.approx(1.0)


def test_port_on_land_keeps_cost_resolution():
    """A port inside the land raster adds nothing to the cost."""
    g = Grids(*_members([1.0, 2.0]), STEPS)
    llat, lwlon = np.arange(35.0, 50.001, 0.25), np.arange(-75.0, 5.001, 0.25)
    coast = np.zeros((llat.size, lwlon.size), np.float32)
    iy, ix = np.abs(llat - COR.o_lat).argmin(), np.abs(lwlon - COR.o_wlon).argmin()
    coast[iy, ix] = 1.0  # only the port's cell: resampled points 0.3 deg out stay clear
    rng = np.random.default_rng(0)
    theta = np.tile(op.gc_init_theta(COR, K, NSP), (64, 1)).astype(np.float32)
    theta[:, -NSP:] = rng.normal(0.0, 0.02, (64, NSP))
    out = []
    for mask in (np.zeros_like(coast), coast):
        land = op.DeviceLand({"lat": llat, "wlon": lwlon, "mask": mask})
        fit, shared = te.make_ensemble_cost(
            g,
            land,
            COR,
            objective="joint",
            L=L,
            K=K,
            n_speed=NSP,
            align=ALIGN,
            wps=False,
            power_fn=toy_power_jax,
            **LIMITS,
        )
        out.append(np.asarray(fit(jnp.asarray(theta), (jnp.float32(0.0), *shared))))
    assert len(np.unique(out[0])) == 64
    np.testing.assert_array_equal(out[1], out[0])
