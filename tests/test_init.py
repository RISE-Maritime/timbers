"""Initialisation, restarts and warm starts (issue #13).

Properties under test:
- the great-circle start follows the great circle, the chord start is v0.1's;
- ``spread`` restarts start on alternate sides of the great circle;
- where a barrier splits the ocean, the chord start stays in the basin it
  starts in and the default (great-circle) start finds the better one;
- a seeded run is never worse than its seed, and every restart is reported.
"""

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax, toy_power_np  # noqa: E402

from timbers import cmaes as jc  # noqa: E402
from timbers import optimizer as op  # noqa: E402
from timbers import solve as oj  # noqa: E402
from timbers.geo import haversine_m  # noqa: E402
from timbers.land import exclusion_raster  # noqa: E402
from timbers.model import Grids  # noqa: E402

COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)  # 48 h passage
K, L, NSP, ALIGN = 6, 40, 4, 0.25
T0 = np.datetime64("2024-01-01T00:00:00", "s")
BARRIER_LON = -36.5


def _mid_lat(lat, wlon):
    """Latitude where a route crosses the barrier's longitude."""
    return float(lat[np.argmin(np.abs(np.asarray(wlon) - BARRIER_LON))])


# --- starting routes ---------------------------------------------------------
def test_gc_init_follows_great_circle():
    """Decoded, the start lies on the great circle (within the 11 nm spacing of
    the reference points) and is as long; the chord start is ~300 nm south."""
    glat, gwlon = op._great_circle(COR, 1024)
    lat, wlon, _ = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, 200, NSP)
    off = [haversine_m(a, b, glat, gwlon).min() / 1852 for a, b in zip(lat, wlon)]
    assert max(off) < 5.0
    length = haversine_m(lat[:-1], wlon[:-1], lat[1:], wlon[1:]).sum()
    gc = haversine_m(COR.o_lat, COR.o_wlon, COR.d_lat, COR.d_wlon)
    assert length == pytest.approx(gc, rel=1e-3)
    clat, cwlon, _ = op.decode_route(op.chord_init_theta(COR, K, NSP), COR, K, 200, NSP)
    assert _mid_lat(lat, wlon) - _mid_lat(clat, cwlon) > 4.0


def test_chord_init_is_v01():
    """``init="chord"`` is the v0.1 start: control points evenly on the chord."""
    o = np.array(COR.norm(COR.o_lat, COR.o_wlon))
    d = np.array(COR.norm(COR.d_lat, COR.d_wlon))
    fr = np.linspace(0, 1, K)[1:-1]
    v01 = np.concatenate([(o + fr[:, None] * (d - o)).ravel(), np.zeros(NSP)])
    np.testing.assert_array_equal(op.init_thetas(COR, K, NSP, 3, "chord"), np.tile(v01, (3, 1)))


def test_spread_alternates_about_great_circle():
    np.testing.assert_allclose(
        op.spread_offsets(6, 0.3), [0.0, 0.1, -0.1, 0.2, -0.2, 0.3], atol=1e-12
    )
    th = op.init_thetas(COR, K, NSP, 5, "spread", 0.2)
    np.testing.assert_array_equal(th[0], op.gc_init_theta(COR, K, NSP))
    np.testing.assert_array_equal(th[:, -NSP:], 0.0)
    mids = [_mid_lat(*op.decode_route(t, COR, K, L, NSP)[:2]) for t in th]
    assert mids[3] > mids[1] > mids[0] > mids[2] > mids[4]  # 0, +, -, ++, --
    with pytest.raises(ValueError):
        op.init_thetas(COR, K, NSP, 2, "straight")


def test_make_solver_per_instance_starts():
    """Per-instance starts equal to the defaults give the default result; a
    different start or step changes it."""

    def fit(X, cargs):
        return jnp.sum((X - cargs[1]) ** 2, axis=1)

    x0, target = jnp.zeros(3, jnp.float32), jnp.ones(3, jnp.float32)
    solve = jc.make_solver(fit, x0, 0.3, jc.hyperparams(3, 8), 8, 20)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    offs = jnp.zeros(2, jnp.float32)
    ref = solve(keys, offs, (target,))
    same = solve(keys, offs, (target,), jnp.zeros((2, 3)), jnp.full(2, 0.3))
    np.testing.assert_allclose(np.asarray(same[0]), np.asarray(ref[0]), rtol=1e-6)
    moved = solve(keys, offs, (target,), jnp.ones((2, 3)), jnp.full(2, 0.3))
    assert not np.allclose(np.asarray(moved[0]), np.asarray(ref[0]))


# --- two basins ---------------------------------------------------------------
def _two_basins():
    """Calm wind, rough seas south of 44 N, and a land barrier across the
    corridor from 40 to 48.5 N. The chord crosses it nearer its south end, the
    great circle nearer its north end; north of it is the better route."""
    lat = np.arange(35.0, 50.001, 0.5)
    lon = np.arange(-75.0, 5.001, 0.5)
    nt = 73
    times = T0 + np.arange(nt) * np.timedelta64(1, "h")
    shape = (nt, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=times, t0=times[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 3.0, np.float32), "v10": np.zeros(shape, np.float32)}
    swh = np.where(lat[None, :, None] < 44.0, 3.0, 1.0) * np.ones(shape)
    wave = {**base, "swh": swh.astype(np.float32), "mwd": np.full(shape, 270.0, np.float32)}
    llat = np.arange(35.0, 50.001, 0.25)
    lwlon = np.arange(-75.0, 5.001, 0.25)
    wall = (llat[:, None] > 40.0) & (llat[:, None] < 48.5) & (np.abs(lwlon + 36.5) < 2.5)[None, :]
    raster = exclusion_raster(
        {"lat": llat, "wlon": lwlon, "mask": wall}, domain=(35.5, 49.5, -74.5, 4.5)
    )
    return Grids.from_era5(wind, wave), wind, wave, op.DeviceLand(raster)


def _solve(problem, **kw):
    grids, wind, wave, land = problem
    args = dict(wps=False, K=K, L=L, NSP=NSP, ALIGN=ALIGN, n_seeds=4, popsize=32, maxiter=80)
    args.update(power_fn=toy_power_jax, power_fn_host=toy_power_np, **kw)
    return oj.solve_corridor(
        COR, grids, land, op.Penalty(), [datetime(2024, 1, 1)], wind, wave, **args
    )[0]


@pytest.fixture(scope="module")
def two_basins():
    return _two_basins()


@pytest.fixture(scope="module")
def chord_result(two_basins):
    return _solve(two_basins, init="chord")


def test_default_init_finds_the_northern_basin(two_basins, chord_result):
    """Every chord restart ends south of the barrier; the default start ends
    north of it, on a route that costs less."""
    assert all(_mid_lat(r["lat"], r["wlon"]) < 44.0 for r in chord_result["restarts"])
    best = _solve(two_basins)
    assert _mid_lat(best["lat"], best["wlon"]) > 48.5
    assert best["energy_mwh"] < 0.97 * chord_result["energy_mwh"]


def test_seed_track_keeps_its_basin(two_basins, chord_result):
    """A northern seed among chord restarts: its restart stays north, the result
    is north and no worse than the seed itself, and every restart is reported."""
    north = op.gc_init_theta(COR, K, NSP, offset=0.03)
    lat, wlon, seg = op.decode_route(north, COR, K, L, NSP)
    track = (np.r_[0.0, np.cumsum(seg)], lat, op.working_to_signed(wlon))
    seed = op.fit_theta_to_track(COR, *track, K=K, L=L, n_speed=NSP)[0]
    seed_mwh = oj._exact(
        seed, datetime(2024, 1, 1), COR, *two_basins[1:3], K, L, NSP, False, toy_power_np
    )["energy_mwh"]

    res = _solve(two_basins, init="chord", seed_tracks=[[track]])
    assert [r["seeded"] for r in res["restarts"]] == [True, False, False, False]
    assert set(res["restarts"][0]) == {"j", "seeded", "lat", "wlon"}
    assert _mid_lat(res["restarts"][0]["lat"], res["restarts"][0]["wlon"]) > 48.5
    assert _mid_lat(res["lat"], res["wlon"]) > 48.5
    assert res["energy_mwh"] <= seed_mwh
    assert res["energy_mwh"] < chord_result["energy_mwh"]


def test_seed_tracks_are_checked(two_basins):
    with pytest.raises(ValueError, match="one entry per departure"):
        _solve(two_basins, seed_tracks=[None, None])
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    track = (np.r_[0.0, np.cumsum(seg)], lat, op.working_to_signed(wlon))
    with pytest.raises(ValueError, match="more seed tracks"):
        _solve(two_basins, seed_tracks=[[track] * 5])
