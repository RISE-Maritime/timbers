"""Risk-aware extension demo: deterministic vs robust routing under forecast error.

Uses the synthetic storm scenario + toy power model from ``run_toy.py``. For one
departure it optimizes two routes:

  * deterministic -- minimize nominal energy (the standard cost,
    ``optimizer.build_fit``);
  * robust        -- minimize nominal energy + expected limit-exceedance over a
    forecast-error surrogate ensemble (``ensemble.make_ensemble_cost`` with
    ``perturbations`` and ``safety_mode="mean"``).

Both are then scored across the SAME perturbation ensemble with
``ensemble.score_members`` (the fragility diagnostic), and we report nominal energy,
nominal max Hs, and the fraction of perturbations whose max Hs exceeds the limit
-- i.e. how often the route would be infeasible if the forecast is a bit off.

The robust route should trade a little nominal energy for a markedly lower
exceedance fraction. NOTE: toy power model + constructed scenario, so the
magnitudes are illustrative of the mechanism, not real-vessel numbers.

    PYTHONPATH=examples python examples/run_risk.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from run_toy import COR, K, L, ALIGN, synthetic_grids

from timbers import cmaes as jc
from timbers import ensemble as te
from timbers import optimizer as op
from timbers.model import Grids
from toy_power import toy_power_jax

NSP = 6
HS_LIM, US_LIM = 7.0, 20.0  # hard feasibility limits
POP, ITERS = 64, 150
DEP_OFF = 0.0


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    return op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )


def _optimize(fit, shared, x0, dim, seed=0):
    """Single-instance sep-CMA solve of a ``(fit, shared)`` batched cost."""
    hp = jc.hyperparams(dim, POP)
    bx, _ = jc.run(
        jnp.asarray(x0, jnp.float32),
        fit,
        0.1,
        hp,
        POP,
        ITERS,
        jax.random.PRNGKey(seed),
        (jnp.float32(DEP_OFF), *shared),
    )
    return np.asarray(bx)


def main():
    wind, wave = synthetic_grids(storm=True)
    grids = Grids.from_era5(wind, wave)
    land = _land()
    dim = 2 * (K - 2) + NSP
    x0 = op.gc_init_theta(COR, K, NSP)

    # Forecast-error surrogate ensemble; row 0 is the nominal (0,0,0,1,1).
    perts = te.perturbation_grid(
        dlat=(0.0, -0.3, 0.3), dt=(0.0, -3.0, 3.0), hs=(1.0, 1.12)
    )  # 18 members

    det = op.build_fit(COR, grids, land, op.Penalty(), K, NSP, L, ALIGN, False, toy_power_jax)
    rob = te.make_ensemble_cost(
        grids,
        land,
        COR,
        objective="chance_constrained",
        safety_mode="mean",
        perturbations=perts,
        L=L,
        K=K,
        n_speed=NSP,
        align=ALIGN,
        wps=False,
        power_fn=toy_power_jax,
        hs_lim=HS_LIM,
        tws_lim=US_LIM,
    )

    print("optimizing deterministic route ...")
    det = _optimize(*det, x0, dim)
    print("optimizing robust route ...")
    rob = _optimize(*rob, x0, dim)

    def report(name, theta):
        lat, wlon, seg = op.decode_route(theta, COR, K, L, NSP)
        m = te.score_members(
            grids,
            COR,
            lat,
            wlon,
            seg,
            wps=False,
            power_fn=toy_power_jax,
            align=ALIGN,
            hs_lim=HS_LIM,
            tws_lim=US_LIM,
            dep_off=DEP_OFF,
            perturbations=perts,
        )
        E, Hs = m["energy_mwh"], m["max_hs"]
        exceed = 100.0 * float((Hs > HS_LIM).mean())
        print(f"{name:14}{E[0]:>12.1f}{Hs[0]:>13.2f}{exceed:>16.0f}%")

    print(f"\n{'route':14}{'nom MWh':>12}{'nom maxHs':>13}{'Hs>limit (ens)':>17}")
    report("deterministic", det)
    report("robust", rob)
    print(
        f"\nlimit Hs = {HS_LIM} m; ensemble = {len(perts)} perturbations "
        "(toy power -- illustrative)"
    )


if __name__ == "__main__":
    main()
