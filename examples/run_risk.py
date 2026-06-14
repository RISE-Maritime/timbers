"""Risk-aware extension demo: deterministic vs robust routing under forecast error.

Uses the synthetic storm scenario + toy power model from ``run_toy.py``. For one
departure it optimizes two routes:

  * deterministic -- minimize nominal energy (the standard cost,
    ``optimizer.make_batched_cost``);
  * robust        -- minimize nominal energy + expected limit-exceedance over a
    forecast-error surrogate ensemble (``risk.make_robust_cost``).

Both are then scored across the SAME perturbation ensemble with
``risk.make_scorer`` (the fragility diagnostic), and we report nominal energy,
nominal max Hs, and the fraction of perturbations whose max Hs exceeds the limit
-- i.e. how often the route would be infeasible if the forecast is a bit off.

The robust route should trade a little nominal energy for a markedly lower
exceedance fraction. NOTE: toy power model + constructed scenario, so the
magnitudes are illustrative of the mechanism, not real-vessel numbers.

    PYTHONPATH=examples python examples/run_risk.py
"""

from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from run_toy import COR, K, L, ALIGN, synthetic_grids

from timbers import cmaes as jc
from timbers import model as jm
from timbers import optimizer as op
from timbers import risk as rk
from toy_power import toy_power_jax

NSP = 6
HS_LIM, US_LIM = 7.0, 20.0          # hard feasibility limits
POP, ITERS = 64, 150
DEP_OFF = 0.0


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    return op.DeviceLand({"lat": llat, "wlon": lwlon,
                          "mask": np.zeros((llat.size, lwlon.size), np.float32)})


def _optimize(cost_fn, x0, dim, seed=0):
    """Single-instance sep-CMA solve against a batched cost fn(theta_batch, dep_off)."""
    hp = jc.hyperparams(dim, POP)
    fit = lambda X, cargs: cost_fn(X, cargs[0])          # noqa: E731
    bx, _ = jc.run(jnp.asarray(x0, jnp.float32), fit, 0.1, hp, POP, ITERS,
                   jax.random.PRNGKey(seed), (jnp.float32(DEP_OFF),))
    return np.asarray(bx)


def main():
    wind, wave = synthetic_grids(storm=True)
    grids = jm.DeviceGrids(wind, wave)
    land = _land()
    dim = 2 * (K - 2) + NSP
    x0 = op.gc_init_theta(COR, K, NSP)

    # Forecast-error surrogate ensemble; row 0 is the nominal (0,0,0,1,1).
    perts = rk.perturbation_grid(dlat=(0.0, -0.3, 0.3), dt=(0.0, -3.0, 3.0),
                                 hs=(1.0, 1.12))                      # 18 members

    det_cost = op.make_batched_cost(grids, land, COR, L, False, op.Penalty(), K,
                                    toy_power_jax, n_speed=NSP, align_dt_h=ALIGN)
    rob_cost = rk.make_robust_cost(grids, land, COR, L, False, K, NSP, ALIGN, perts,
                                   toy_power_jax, hs_lim=HS_LIM, us_lim=US_LIM,
                                   aH=4.0, aU=2.0, lam_env=1.0)

    print("optimizing deterministic route ...")
    det = _optimize(det_cost, x0, dim)
    print("optimizing robust route ...")
    rob = _optimize(rob_cost, x0, dim)

    scorer = rk.make_scorer(grids, COR, False, toy_power_jax, align=ALIGN)

    def report(name, theta):
        lat, wlon, seg = op.decode_route(theta, COR, K, L, NSP)
        E, Hs, _ = scorer(jnp.asarray(lat, jnp.float32),
                          jnp.asarray(op.working_to_signed(wlon), jnp.float32),
                          jnp.asarray(seg, jnp.float32), DEP_OFF, perts)
        E, Hs = np.asarray(E), np.asarray(Hs)
        exceed = 100.0 * float((Hs > HS_LIM).mean())
        print(f"{name:14}{E[0]:>12.1f}{Hs[0]:>13.2f}{exceed:>16.0f}%")

    print(f"\n{'route':14}{'nom MWh':>12}{'nom maxHs':>13}{'Hs>limit (ens)':>17}")
    report("deterministic", det)
    report("robust", rob)
    print(f"\nlimit Hs = {HS_LIM} m; ensemble = {len(perts)} perturbations "
          "(toy power -- illustrative)")


if __name__ == "__main__":
    main()
