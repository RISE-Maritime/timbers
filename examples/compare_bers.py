"""TIMBERS vs BERS: isolate the explicit-speed lever on the toy problem.

TIMBERS is a strict superset of the BERS reference method: ``n_speed = 0`` makes
the time-allocation profile uniform, recovering BERS's implicit uniform-speed,
geometry-only model. So a single code path gives both methods -- the only knob
that changes is whether speed is a decision variable.

This runs a 2x2 ablation (uniform vs explicit speed) x (Stage 1 only vs + polish)
on the synthetic corridor + toy power model from ``run_toy.py`` and prints the
energy ladder. The TIMBERS contribution is the uniform -> explicit-speed delta,
with geometry, optimizer, seeds and polish held identical.

NOTE: this is the toy power model, so the *magnitudes* are illustrative of the
method difference -- not the real-vessel or any benchmark numbers.

    PYTHONPATH=examples python examples/compare_bers.py
"""

import numpy as np
from datetime import datetime

from run_toy import COR, K, L, ALIGN, synthetic_grids

from timbers import model as jm
from timbers import optimizer as op
from timbers import solve as oj
from toy_power import toy_power_jax, toy_power_np

NSP_TIMBERS = 6                 # speed-profile log-weights for TIMBERS mode
N_SEEDS, POPSIZE, MAXITER = 4, 64, 150


def run(n_speed: int, polish: bool) -> float:
    wind, wave = synthetic_grids(storm=True)
    grids = jm.DeviceGrids(wind, wave)
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    land = op.DeviceLand({"lat": llat, "wlon": lwlon,
                          "mask": np.zeros((llat.size, lwlon.size), np.float32)})
    out = oj.solve_corridor(
        COR, grids, land, op.Penalty(), [datetime(2024, 1, 1)], wind, wave,
        wps=False, K=K, L=L, NSP=n_speed, ALIGN=ALIGN,
        n_seeds=N_SEEDS, popsize=POPSIZE, maxiter=MAXITER, base_seed=0,
        power_fn=toy_power_jax, power_fn_host=toy_power_np, polish=polish,
    )
    return out[0]["energy_mwh"]


def main():
    grid = {(nsp, pol): run(nsp, pol)
            for nsp in (0, NSP_TIMBERS) for pol in (False, True)}
    print(f"{'':22}{'uniform speed (BERS)':>22}{'explicit speed (TIMBERS)':>26}")
    for pol, label in ((False, "Stage 1 only"), (True, "+ gradient polish")):
        b, t = grid[(0, pol)], grid[(NSP_TIMBERS, pol)]
        red = 100.0 * (b - t) / b
        print(f"{label:22}{b:>22.3f}{t:>26.3f}   ({red:+.1f}% from speed)")
    print("\nenergy in MWh (toy power model -- magnitudes illustrative)")


if __name__ == "__main__":
    main()
