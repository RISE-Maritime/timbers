"""Routing under power-model uncertainty: the ship, not the weather, is uncertain.

Uses the synthetic storm scenario and the toy power model from ``run_toy.py``,
with the weather taken as known and no seakeeping limits, so that only energy
and the shaft-power ceiling are at stake. The power model's parameters are
uncertain: the calm-water level (fouling), the added resistance in waves and
the speed exponent of the hull term, 18 draws each held for the whole voyage
(``ensemble.model_param_grid``, passed as ``model_params``). For one departure
it optimizes four routes:

  * nominal   -- nominal energy with the nominal model;
  * mean      -- expected energy over the draws (``objective="joint"``);
  * cvar      -- the mean energy of the worst 20% of draws (``cost_mode="cvar"``);
  * ceiling   -- nominal energy, with the share of draws that need more than
    the shaft-power ceiling pushed toward 10% (``power_eps=0.1``).

Each route is then scored over the same draws: on the device
(``score_members``: energy for the nominal model, the mean and the CVaR over the
draws, and the share of draws that reach the ceiling), and sailed on the host
under the ceiling one draw at a time (``evaluate_route_saturated`` with
``model_draws``), which gives the delay the ceiling causes.

What to expect: the spread is large (the mean is about 5% and the CVaR about
11% above the nominal energy) but the routes barely move. The calm-water and
wave scales enter the power linearly and average out of the mean; the hull term
dominates, so no route avoids it. Only the ceiling, where it binds, changes the
route, and only a little: the CVaR penalty is gentle for a small excess, and
the slowdown it would prevent is minutes. The value here is the risk measured
on the plan.
NOTE: toy power model and constructed scenario, so the magnitudes are
illustrative of the mechanism, not real-vessel numbers.

    PYTHONPATH=examples python examples/run_model_risk.py
"""

from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from run_toy import COR, K, L, ALIGN, synthetic_grids

from timbers import cmaes as jc
from timbers import ensemble as te
from timbers import optimizer as op
from timbers.model import Grids
from timbers.scoring import evaluate_route_saturated
from toy_power import toy_power_jax, toy_power_np

NSP = 6
HS_LIM = US_LIM = float("inf")  # no seakeeping limits: energy and the ceiling only
P_MAX = 160_000.0  # shaft-power ceiling, kW
POP, ITERS, SEEDS = 64, 300, 4
DEP = datetime(2024, 1, 1)

# 18 draws; the nominal value first on every axis, so draw 0 is the nominal model.
DRAWS = te.model_param_grid(calm=(1.0, 1.1), wave=(1.0, 0.6, 1.6), n=(3.0, 2.7, 3.3))


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    return op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )


def _optimize(grids, land, x0, **kw):
    """Best of ``SEEDS`` sep-CMA-ES restarts of one objective."""
    fit, shared = te.make_ensemble_cost(
        grids,
        land,
        COR,
        L=L,
        K=K,
        n_speed=NSP,
        align=ALIGN,
        wps=False,
        power_fn=toy_power_jax,
        hs_lim=HS_LIM,
        tws_lim=US_LIM,
        safety_mode="cvar",
        **kw,
    )
    hp = jc.hyperparams(x0.size, POP)
    cargs = (jnp.float32(0.0), *shared)
    runs = [
        jc.run(jnp.asarray(x0, jnp.float32), fit, 0.1, hp, POP, ITERS, jax.random.PRNGKey(s), cargs)
        for s in range(SEEDS)
    ]
    return np.asarray(min(runs, key=lambda r: float(r[1]))[0])


def main():
    wind, wave = synthetic_grids(storm=True)
    grids = Grids.from_era5(wind, wave)
    land = _land()
    x0 = op.gc_init_theta(COR, K, NSP)

    # The routes differ only in how energy is reduced over the draws and whether
    # the power ceiling is constrained. CVaR safety, as the ceiling is infeasible
    # in some draws and "prob" would then not see by how much.
    routes = {
        "nominal": dict(objective="chance_constrained"),
        "mean": dict(objective="joint", model_params=DRAWS),
        "cvar": dict(objective="joint", model_params=DRAWS, cost_mode="cvar", cost_eps=0.2),
        "ceiling": dict(
            objective="chance_constrained", model_params=DRAWS, p_lim=P_MAX, power_eps=0.1
        ),
    }
    print(f"{len(te.model_draws(DRAWS))} power-model draws; ceiling {P_MAX / 1000:.0f} MW")
    print(
        f"\n{'route':9}{'nom MWh':>9}{'mean MWh':>10}{'CVaR20':>9}"
        f"{'P>ceil':>8}{'mean delay':>12}{'late>1h':>9}"
    )
    for name, kw in routes.items():
        theta = _optimize(grids, land, x0, **kw)
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
            p_lim=P_MAX,
            model_params=DRAWS,
        )
        e = m["energy_mwh"]
        k = int(np.ceil(0.2 * e.size))
        delays = np.array(
            [
                evaluate_route_saturated(
                    wind,
                    wave,
                    DEP,
                    lat,
                    op.working_to_signed(wlon),
                    seg,
                    partial(toy_power_np, params=d),
                    p_max=P_MAX,
                )["delay_h"]
                for d in te.model_draws(DRAWS)
            ]
        )
        print(
            f"{name:9}{e[0]:>9.0f}{e.mean():>10.0f}{np.sort(e)[-k:].mean():>9.0f}"
            f"{(m['power_margin'] > 0).mean():>8.0%}{delays.mean():>10.2f} h"
            f"{(delays > 1.0).mean():>9.0%}"
        )
    print("\ntoy power model -- illustrative")


if __name__ == "__main__":
    main()
