"""Minimal runnable demo: optimize one route on synthetic weather + a toy power model.

TIMBERS ships no vessel power model or case definitions; you provide your own.
This script wires up a tiny synthetic corridor and the toy power model in
``toy_power.py`` so the full pipeline (device sep-CMA cost -> host scorer) can be
run with no external data.

    python examples/run_toy.py
"""

from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from timbers import cmaes as jc
from timbers import fitness as jrf
from timbers import model as jm
from timbers import optimizer as op
from timbers import solve as oj
from toy_power import toy_power_jax, toy_power_np

COR = op.Corridor("demo", 43.6, -4.0, 40.6, -69.0, 48.0)  # ports + 48 h passage
K, L, NSP, ALIGN = 6, 40, 4, 0.25
T0 = np.datetime64("2024-01-01T00:00:00", "s")


def synthetic_grids():
    lat = np.arange(35.0, 50.001, 0.5)
    lon = np.arange(-75.0, 5.001, 0.5)          # signed lon -> no antimeridian wrap
    nt = 73
    times = T0 + np.arange(nt) * np.timedelta64(1, "h")
    Y, X = lat.size, lon.size
    ones = np.ones((nt, Y, X), np.float32)
    tt = np.arange(nt)[:, None, None]
    u10 = (5.0 + 3.0 * np.sin(0.1 * tt)).astype(np.float32) * ones
    v10 = (-4.0 + 2.0 * np.cos(0.08 * tt)).astype(np.float32) * ones
    swh = (1.5 + 0.6 * np.sin(0.07 * tt)).astype(np.float32) * ones
    mwd = np.mod(200.0 + 20.0 * np.cos(0.05 * tt), 360.0).astype(np.float32) * ones
    base = dict(lat=lat, lon=lon, times=times, t0=times[0], dt_h=1.0)
    return {**base, "u10": u10, "v10": v10}, {**base, "swh": swh, "mwd": mwd}


def main():
    wind, wave = synthetic_grids()
    grids = jm.DeviceGrids(wind, wave)
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    land = op.DeviceLand({"lat": llat, "wlon": lwlon,
                          "mask": np.zeros((llat.size, lwlon.size), np.float32)})

    out = oj.solve_corridor(
        COR, grids, land, op.Penalty(), [datetime(2024, 1, 1)], wind, wave,
        wps=False, K=K, L=L, NSP=NSP, ALIGN=ALIGN,
        n_seeds=4, popsize=64, maxiter=120,
        power_fn=toy_power_jax, power_fn_host=toy_power_np, polish=True,
    )
    r = out[0]
    print(f"toy energy : {r['energy_mwh']:.3f} MWh")
    print(f"max Hs     : {r['max_hs_m']:.2f} m")
    print(f"max wind   : {r['max_wind_mps']:.2f} m/s")
    print(f"distance   : {r['sailed_distance_nm']:.1f} nm")
    print(f"passage    : {r['seg_dt'].sum():.1f} h over {L} waypoints")


if __name__ == "__main__":
    main()
