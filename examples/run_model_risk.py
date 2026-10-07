"""Routing under power-model uncertainty: the ship, not the weather, is uncertain.

Uses the synthetic storm scenario and the toy power model from ``run_toy.py``,
with the weather taken as known and no seakeeping limits, so that only energy
and the shaft-power ceiling are at stake. The ship's model is the toy with a
stronger wave term (``wave=4``, so that the storm adds tens of per cent to the
power, not a few). Its uncertain parameters are the calm-water level, the wave
coefficient and the speed exponent of the hull term, each on an axis centred on
the nominal value, 27 draws held for the whole voyage
(``ensemble.model_param_grid``, passed as ``model_params``).

The design is fixed before the result is seen:

* the ceiling is 95% of the peak power of the nominal plan under the nominal
  model, so the nominal plan itself would have to slow down in the storm;
* every route has 12 speed weights over the 48 h passage, so the speed profile
  can resolve a storm a few hours long;
* the routes form two groups. Two use the nominal model only: ``nominal``
  ignores the ceiling, and ``margin`` keeps clear of it with the soft penalty
  that starts at 93% of it, which every objective pays for a finite ``p_lim``.
  Three use the draws and add, opt-in, a chance constraint on the ceiling over
  them (``power_eps=0.1`` under ``safety_mode="cvar"``: the worst 10% of draws
  must stay within the ceiling): ``draws``, nominal energy
  (``chance_constrained``); ``draws mean``, the expected energy (``joint``);
  and ``draws cvar``, the mean energy of the worst 20% of draws
  (``cost_mode="cvar"``).

Each route is then scored over the same draws: on the device
(``score_members``: nominal energy, the mean and CVaR20 over the draws, and the
share of draws that reach the ceiling), and sailed on the host under the
ceiling one draw at a time (``evaluate_route_saturated`` with ``model_draws``),
which gives the delay the ceiling causes. Comparing the groups shows what the
draws add over a plan that only knows the ceiling.

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

NSP = 12
HS_LIM = US_LIM = float("inf")  # no seakeeping limits: energy and the ceiling only
CEILING_FRAC = 0.95  # of the nominal plan's peak power
POP, ITERS, SEEDS = 64, 300, 4
DEP = datetime(2024, 1, 1)

# The nominal value first on every axis, so draw 0 is the nominal model, and
# each axis symmetric about it, so the mean model is the nominal one.
NOMINAL = te.model_param_grid(wave=(4.0,))
DRAWS = te.model_param_grid(calm=(1.0, 0.9, 1.1), wave=(4.0, 2.0, 6.0), n=(3.0, 2.5, 3.5))


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
        eps=0.1,
        **kw,
    )
    hp = jc.hyperparams(x0.size, POP)
    cargs = (jnp.float32(0.0), *shared)
    runs = [
        jc.run(jnp.asarray(x0, jnp.float32), fit, 0.1, hp, POP, ITERS, jax.random.PRNGKey(s), cargs)
        for s in range(SEEDS)
    ]
    theta = np.asarray(min(runs, key=lambda r: float(r[1]))[0])
    return op.decode_route(theta, COR, K, L, NSP)


def _score(grids, route, wind, wave, p_max):
    lat, wlon, seg = route
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
        p_lim=p_max,
        model_params=DRAWS,
    )
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
                p_max=p_max,
            )["delay_h"]
            for d in te.model_draws(DRAWS)
        ]
    )
    return m, np.maximum(delays, 0.0)


def main():
    wind, wave = synthetic_grids(storm=True)
    grids = Grids.from_era5(wind, wave)
    land = _land()
    x0 = op.gc_init_theta(COR, K, NSP)

    nominal = _optimize(grids, land, x0, objective="deterministic", model_params=NOMINAL)
    lat, wlon, seg = nominal
    peak = te.score_members(
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
        model_params=NOMINAL,
    )["max_power"][0]
    p_max = CEILING_FRAC * float(peak)

    ceil = dict(p_lim=p_max)
    drawn = dict(model_params=DRAWS, p_lim=p_max, power_eps=0.1)
    routes = {
        "nominal": nominal,
        "margin": _optimize(
            grids, land, x0, objective="deterministic", model_params=NOMINAL, **ceil
        ),
        "draws": _optimize(grids, land, x0, objective="chance_constrained", **drawn),
        "draws mean": _optimize(grids, land, x0, objective="joint", **drawn),
        "draws cvar": _optimize(
            grids,
            land,
            x0,
            objective="joint",
            cost_mode="cvar",
            cost_eps=0.2,
            **drawn,
        ),
    }

    n = len(te.model_draws(DRAWS))
    print(f"{n} power-model draws; ceiling {p_max / 1000:.1f} MW (95% of the nominal plan's peak)")
    print(
        f"\n{'route':12}{'nom MWh':>9}{'mean MWh':>10}{'CVaR20':>9}{'P>ceil':>8}"
        f"{'mean delay':>12}{'max delay':>11}{'late>1h':>9}"
    )
    for name, route in routes.items():
        m, delays = _score(grids, route, wind, wave, p_max)
        e = m["energy_mwh"]
        k = int(np.ceil(0.2 * e.size))
        print(
            f"{name:12}{e[0]:>9.0f}{e.mean():>10.0f}{np.sort(e)[-k:].mean():>9.0f}"
            f"{(m['power_margin'] > 0).mean():>8.0%}{delays.mean():>10.2f} h"
            f"{delays.max():>9.2f} h{(delays > 1.0).mean():>9.0%}"
        )
    print("\ntoy power model -- illustrative")


if __name__ == "__main__":
    main()
