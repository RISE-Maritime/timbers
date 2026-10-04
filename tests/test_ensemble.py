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
from timbers import optimizer as op  # noqa: E402

COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)
K, L, NSP, ALIGN = 6, 40, 4, 0.25
LAT = np.arange(35.0, 50.001, 0.5)
LON = np.arange(-75.0, 5.001, 0.5)
STEPS = np.array([0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60], np.float32)
LIMITS = dict(hs_lim=7.0, tws_lim=20.0)


def _members(swh):
    """Calm, uniform members; ``swh`` is one value per member."""
    shape = (len(swh), STEPS.size, LAT.size, LON.size)
    wind = dict(lat=LAT, lon=LON, u10=np.full(shape, 4.0, np.float32),
                v10=np.full(shape, -2.0, np.float32))
    wave = dict(lat=LAT, lon=LON, mwd=np.full(shape, 200.0, np.float32),
                swh=np.asarray(swh, np.float32)[:, None, None, None] * np.ones(shape, np.float32))
    return wind, wave


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    return op.DeviceLand({"lat": llat, "wlon": lwlon,
                          "mask": np.zeros((llat.size, lwlon.size), np.float32)})


def _fit(grids, objective, **kw):
    fit, shared = te.make_ensemble_cost(grids, _land(), COR, objective=objective, L=L, K=K,
                                        n_speed=NSP, align=ALIGN, wps=False,
                                        power_fn=toy_power_jax, **LIMITS, **kw)
    theta = jnp.asarray(op.gc_init_theta(COR, K, NSP))[None, :]
    return float(fit(theta, (jnp.float32(0.0), *shared))[0])


def _gc_members(grids):
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    return te.score_members(grids, COR, lat, wlon, seg, wps=False, power_fn=toy_power_jax,
                            align=ALIGN, **LIMITS)


def test_time_index_on_non_uniform_steps():
    ti, tf = te.time_index(jnp.asarray([9.0, 15.0, 0.0]), jnp.asarray(STEPS))
    assert list(np.asarray(ti)) == [3, 4, 0]
    np.testing.assert_allclose(np.asarray(tf), [0.0, 0.5, 0.0], atol=1e-6)


def test_non_finite_fields_are_rejected():
    wind, wave = _members([1.0, 1.0])
    wave["swh"][0, 0, 3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        te.EnsembleGrids(wind, wave, STEPS)


def test_single_field_wraps_as_one_member():
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(100) * np.timedelta64(1, "h")
    shape = (100, LAT.size, LON.size)
    base = dict(lat=LAT, lon=LON, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.ones(shape, np.float32), "v10": np.ones(shape, np.float32)}
    wave = {**base, "swh": np.ones(shape, np.float32), "mwd": np.zeros(shape, np.float32)}
    g = te.as_ensemble(wind, wave, datetime(2024, 1, 1, 6), 48.0)
    assert g.n_members == 1
    np.testing.assert_allclose(np.asarray(g.steps)[:3], [0.0, 1.0, 2.0])
    assert g.nt == 51


def test_objectives_reduce_members_as_documented():
    """Calm weather, no penalties active: deterministic is member 0's energy and
    expected-value the member mean; identical members make all four agree."""
    g = te.EnsembleGrids(*_members([0.5, 2.0, 3.0, 4.0]), STEPS)
    e = _gc_members(g)["energy_mwh"]
    assert _fit(g, "deterministic") == pytest.approx(e[0], rel=1e-5)
    assert _fit(g, "expected_value") == pytest.approx(e.mean(), rel=1e-5)
    same = te.EnsembleGrids(*_members([2.0] * 4), STEPS)
    vals = [_fit(same, o) for o in te.OBJECTIVES]
    assert max(vals) == pytest.approx(min(vals), rel=1e-6)


def test_breach_probability_counts_members():
    g = te.EnsembleGrids(*_members([1.0, 1.0, 9.0, 9.0]), STEPS)   # two members above 7 m
    m = _gc_members(g)
    assert (m["margin"] > 0).mean() == pytest.approx(0.5)
    # The chance constraint sees the two stormy members; the deterministic arm,
    # reading member 0, does not.
    assert _fit(g, "chance_constrained") > _fit(g, "deterministic") + 1.0


def test_series_integrates_to_route_energy():
    g = te.EnsembleGrids(*_members([1.0, 2.0]), STEPS)
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seg))])
    s = te.member_series(g, t, lat, op.working_to_signed(wlon), wps=False,
                         power_fn=toy_power_jax)
    assert s["valid"].all()
    e_series = (s["power_kw"] * s["seg_h"]).sum(axis=1) / 1000.0
    np.testing.assert_allclose(e_series, _gc_members(g)["energy_mwh"], rtol=0.05)


def test_cost_drives_the_optimizer():
    g = te.EnsembleGrids(*_members([1.0, 3.0, 5.0, 8.0]), STEPS)
    fit, shared = te.make_ensemble_cost(g, _land(), COR, objective="joint", L=L, K=K,
                                        n_speed=NSP, align=ALIGN, wps=False,
                                        power_fn=toy_power_jax, **LIMITS)
    x0 = jnp.asarray(op.gc_init_theta(COR, K, NSP))
    cargs = (jnp.float32(0.0), *shared)
    j0 = float(fit(x0[None, :], cargs)[0])
    hp = jc.hyperparams(x0.size, 16)
    _, best = jc.run(x0, fit, 0.1, hp, 16, 30, jax.random.PRNGKey(0), cargs)
    assert np.isfinite(float(best)) and float(best) <= j0 + 1e-3
