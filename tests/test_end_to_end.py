"""Data-free end-to-end tests with an injected toy power model.

Exercises the full pipeline -- device sep-CMA cost, the JAX route-energy
evaluator, the NumPy host scorer, and the production ``solve_corridor`` backend
-- on a tiny synthetic corridor with synthetic weather. No ERA5 data and no
vessel performance model required; the power model is the toy in
``examples/toy_power.py``, injected via ``power_fn``.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax, toy_power_np  # noqa: E402

from timbers import cmaes as jc  # noqa: E402
from timbers import fitness as jrf  # noqa: E402
from timbers import model as jm  # noqa: E402
from timbers import optimizer as op  # noqa: E402
from timbers import solve as oj  # noqa: E402
from timbers.scoring import evaluate_route, evaluate_route_full  # noqa: E402

# --- tiny synthetic problem ------------------------------------------------
COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)  # 48 h passage
K, L, NSP, ALIGN = 6, 40, 4, 0.25
T0 = np.datetime64("2024-01-01T00:00:00", "s")


def _grid():
    lat = np.arange(35.0, 50.001, 0.5, dtype=np.float64)        # ascending
    lon = np.arange(-75.0, 5.001, 0.5, dtype=np.float64)        # signed -> lon_wrap False
    nt = 73
    times = T0 + np.arange(nt) * np.timedelta64(1, "h")
    Y, X = lat.size, lon.size
    tt = np.arange(nt)[:, None, None]
    yy = np.arange(Y)[None, :, None]
    xx = np.arange(X)[None, None, :]
    u10 = (5.0 + 3.0 * np.sin(0.1 * tt + 0.05 * xx)).astype(np.float32) * np.ones((nt, Y, X), np.float32)
    v10 = (-4.0 + 2.0 * np.cos(0.08 * tt + 0.04 * yy)).astype(np.float32) * np.ones((nt, Y, X), np.float32)
    swh = (1.5 + 0.6 * np.sin(0.07 * tt + 0.03 * xx)).astype(np.float32) * np.ones((nt, Y, X), np.float32)
    mwd = np.mod(200.0 + 20.0 * np.cos(0.05 * tt) + 0.0 * xx, 360.0).astype(np.float32) * np.ones((nt, Y, X), np.float32)
    base = dict(lat=lat, lon=lon, times=times, t0=times[0], dt_h=1.0)
    wind = {**base, "u10": u10, "v10": v10}
    wave = {**base, "swh": swh, "mwd": mwd}
    return wind, wave


def _device_grids():
    wind, wave = _grid()
    return jm.DeviceGrids(wind, wave), wind, wave


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    mask = np.zeros((llat.size, lwlon.size), np.float32)        # all water
    return op.DeviceLand({"lat": llat, "wlon": lwlon, "mask": mask})


def test_device_optimizer_reduces_cost_and_pins_endpoints():
    grids, _, _ = _device_grids()
    land = _land()
    pen = op.Penalty()
    fit, x0, shared = jrf.build_fit(COR, grids, land, pen, K, NSP, L, ALIGN,
                                    wps=False, power_fn=toy_power_jax)
    dim = 2 * (K - 2) + NSP
    cargs0 = (jnp.float32(0.0),) + tuple(shared)
    init_cost = float(fit(x0[None, :], cargs0)[0])

    hp = jc.hyperparams(dim, 16)
    solver = jc.make_solver(fit, x0, 0.1, hp, 16, 30)
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    best_x, best_f = solver(keys, jnp.asarray([0.0], jnp.float32), shared)
    best_x, best_f = np.asarray(best_x), np.asarray(best_f)

    assert np.isfinite(best_f).all()
    assert best_f[0] <= init_cost + 1e-3                        # never worse than init

    lat, wlon, seg_dt = op.decode_route(best_x[0], COR, K, L, NSP)
    np.testing.assert_allclose([lat[0], wlon[0]], [COR.o_lat, COR.o_wlon], atol=1e-3)
    np.testing.assert_allclose([lat[-1], wlon[-1]], [COR.d_lat, COR.d_wlon], atol=1e-3)
    assert seg_dt.min() > 0.0
    np.testing.assert_allclose(seg_dt.sum(), COR.hours, rtol=1e-4)


def test_route_energy_model_path():
    grids, _, _ = _device_grids()
    lat, wlon, seg_dt = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    signed = op.working_to_signed(wlon)
    e = jm.route_energy(grids, jnp.asarray(lat, jnp.float32),
                        jnp.asarray(signed, jnp.float32),
                        jnp.asarray(seg_dt, jnp.float32), 0.0, False, toy_power_jax)
    e = float(e)
    assert np.isfinite(e) and e > 0.0


def test_host_scorer_path():
    _, wind, wave = _device_grids()
    lat, wlon, seg_dt = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    signed = op.working_to_signed(wlon)
    acc = np.concatenate([[0.0], np.cumsum(seg_dt)])
    wps_pts = [(T0.item() + timedelta(hours=float(a)), float(la), float(lo))
               for a, la, lo in zip(acc, lat, signed)]
    e = evaluate_route(wind, wave, wps_pts, toy_power_np, wps=False)
    assert np.isfinite(e) and e > 0.0
    full = evaluate_route_full(wind, wave, wps_pts, toy_power_np, wps=False)
    assert set(full) == {"energy_mwh", "max_wind_mps", "max_hs_m", "sailed_distance_nm"}
    assert full["energy_mwh"] > 0.0


def test_solve_corridor_backend():
    grids, wind, wave = _device_grids()
    land = _land()
    deps = [datetime(2024, 1, 1, 0, 0, 0)]
    out = oj.solve_corridor(
        COR, grids, land, op.Penalty(), deps, wind, wave,
        wps=False, K=K, L=L, NSP=NSP, ALIGN=ALIGN,
        n_seeds=2, popsize=16, maxiter=20,
        power_fn=toy_power_jax, power_fn_host=toy_power_np, polish=False,
    )
    assert len(out) == 1
    r = out[0]
    assert np.isfinite(r["energy_mwh"]) and r["energy_mwh"] > 0.0
    np.testing.assert_allclose(r["seg_dt"].sum(), COR.hours, rtol=1e-4)
