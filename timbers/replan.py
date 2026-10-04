"""Sail a voyage with re-planning on each forecast cycle.

A plan made once at departure is followed to arrival whatever the weather does.
:func:`sail_with_replanning` instead plans, sails one forecast cycle on the
verifying weather, and plans again from wherever the ship then is, with the
forecast issued at that time. This is the receding-horizon (recourse) setting a
routing service operates in, and lets the two architectures be compared on the
same truth.

Three rules govern the loop:

* **Re-plan from the realised position.** The ship sails under a shaft-power
  ceiling (:func:`timbers.scoring.evaluate_route_saturated`) and is generally
  off its planned schedule, so each plan starts from where the ship actually is.
* **The arrival time does not move.** Each plan is given the time remaining to
  the original arrival, so a ship that falls behind plans faster legs rather
  than resetting its schedule. When the remaining time is shorter than the
  calm-water passage at ``p_max``, it is floored there and the plan is a best
  effort.
* **Each cycle needs only the remaining horizon.** The forecast provider is
  asked for ``remaining_h`` hours from the cycle time, not for its full range.

The forecast is supplied as a callable, so any source fits: an archive of
ensemble forecasts, a single deterministic forecast, or the verifying field
itself for a perfect-information (hindsight) voyage::

    forecast = lambda t, hours: ensemble.as_ensemble(era5_wind, era5_wave, t, hours)
"""

from __future__ import annotations

import gc
from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np

from . import cmaes as jc
from . import optimizer as op
from .ensemble import make_ensemble_cost, score_members
from .geo import gc_distance_nm
from .scoring import evaluate_route_saturated, v_max_for_power

MAX_LEGS = 100      # bounds a voyage that never arrives


def _plan(grids, land, cor, *, objective, K, L, n_speed, seeds, iters, pop,
          sigma0, cost_kw):
    """Best of ``seeds`` sep-CMA-ES restarts from the great-circle chord."""
    fit, shared = make_ensemble_cost(grids, land, cor, objective=objective,
                                     L=L, K=K, n_speed=n_speed, **cost_kw)
    hp = jc.hyperparams(2 * (K - 2) + n_speed, pop)
    x0 = jnp.asarray(op.gc_init_theta(cor, K, n_speed), jnp.float32)
    best, best_j = None, np.inf
    for s in range(seeds):
        bx, bf = jc.run(x0, fit, sigma0, hp, pop, iters, jax.random.PRNGKey(s),
                        (jnp.float32(0.0), *shared))
        if float(bf) < best_j:
            best, best_j = np.asarray(bx), float(bf)
    return best, best_j


def _sailed_window(seg_dt, hours):
    """Index ``j`` and segment durations of the first ``hours`` of a plan."""
    t_cum = np.concatenate([[0.0], np.cumsum(seg_dt)])
    j = int(np.clip(np.searchsorted(t_cum, hours, side="left"), 1, len(seg_dt)))
    sub = seg_dt[:j].copy()
    sub[-1] -= t_cum[j] - hours
    return j, sub


def sail_with_replanning(origin, destination, hours, dep, forecast, truth_wind,
                         truth_wave, land, *, objective, power_fn, power_fn_host,
                         hs_lim, tws_lim, p_max=np.inf, plan_limit_scale=1.0,
                         cycle_h=24.0, K=6, L=40, n_speed=8, align=0.25, wps=False,
                         eps=0.1, seeds=16, iters=300, pop=64, sigma0=0.1,
                         record_promise=False, clear_jax_caches=False,
                         verbose=False, **cost_kw):
    """Plan, sail one cycle on the verifying weather, re-plan. Returns the voyage.

    Parameters
    ----------
    origin, destination : (lat, lon)
        End points, signed longitude.
    hours : float
        Contractual passage time from ``dep``.
    dep : datetime
        Departure time; cycle ``k`` is issued at ``dep + k * cycle_h``.
    forecast : callable
        ``forecast(cycle_time, hours) -> EnsembleGrids`` covering at least
        ``hours`` from ``cycle_time``.
    truth_wind, truth_wave : dict
        Verifying weather the ship sails on (e.g. from
        :func:`timbers.era5.load_era5`).
    land : DeviceLand
        Exclusion raster for planning.
    objective : str
        One of :data:`timbers.ensemble.OBJECTIVES`.
    power_fn, power_fn_host : callable
        The same power model for JAX (planning) and NumPy (sailing).
    hs_lim, tws_lim : float
        Seakeeping limits on significant wave height (m) and true wind speed
        (m/s). They are reported against in ``max_hs_m`` / ``max_wind_mps`` and
        used for the promise.
    p_max : float
        Shaft-power ceiling in kW, for planning and sailing.
    plan_limit_scale : float
        Plan against ``plan_limit_scale`` times the limits; below 1 this is a
        safety margin. The promise is still computed against the true limits.
    cycle_h : float
        Hours between forecast cycles, and hence sailed per plan.
    K, L, n_speed, align, wps, eps, **cost_kw
        Passed to :func:`timbers.ensemble.make_ensemble_cost`, with the same
        values for every leg.
    seeds, iters, pop, sigma0
        Each plan is the best of ``seeds`` sep-CMA-ES restarts.
    record_promise : bool
        Per leg, the share of members in which the plan breaches a limit over
        the window it is then sailed (``p_hat`` in each leg record). Each leg
        scores a differently shaped sub-route and so compiles anew; leave it
        off when only outcomes are needed.
    clear_jax_caches : bool
        Clear JAX's compilation caches after each leg. Plans of different
        length compile separately, and on a memory-limited host a whole
        voyage's executables alongside a large ensemble may not fit.

    Returns
    -------
    dict
        ``energy_mwh``, ``actual_hours``, ``delay_h``, ``arrived``,
        ``max_hs_m``, ``max_wind_mps``, ``max_power_kw``, ``n_legs``, ``track``
        (realised ``t_h``, ``lat``, ``lon`` on the voyage clock) and ``legs``,
        one record per cycle.
    """
    o_lat, o_lon = origin
    d_lat, d_lon = destination
    v_calm = float(v_max_for_power(power_fn_host, 0.0, 0.0, 0.0, 0.0, wps, p_max))
    cost_kw = dict(align=align, wps=wps, power_fn=power_fn,
                   hs_lim=plan_limit_scale * hs_lim,
                   tws_lim=plan_limit_scale * tws_lim, p_lim=p_max, eps=eps,
                   **cost_kw)

    lat_now, lon_now = o_lat, o_lon
    elapsed = energy = 0.0
    max_hs = max_tws = max_p = 0.0
    legs, track = [], dict(t_h=[0.0], lat=[o_lat], lon=[o_lon])
    arrived = False

    for leg in range(MAX_LEGS):
        remaining_nm = gc_distance_nm(lat_now, lon_now, d_lat, d_lon)
        if remaining_nm < 1.0:
            arrived = True
            break
        floor_h = remaining_nm * 1852.0 / (v_calm * 3600.0)
        remaining_h = float(max(hours - elapsed, floor_h))

        cycle_t = dep + timedelta(hours=leg * cycle_h)
        grids = forecast(cycle_t, remaining_h)
        cor = op.Corridor(f"leg{leg}", lat_now, lon_now, d_lat, d_lon, remaining_h)
        best, best_j = _plan(grids, land, cor, objective=objective, K=K, L=L,
                             n_speed=n_speed, seeds=seeds, iters=iters, pop=pop,
                             sigma0=sigma0, cost_kw=cost_kw)
        rlat, rwlon, seg = op.decode_route(best, cor, K, L, n_speed)

        p_hat = None
        if record_promise:
            j, sub = _sailed_window(seg, min(cycle_h, remaining_h))
            sub_cor = op.Corridor(f"leg{leg}_sailed", float(rlat[0]),
                                  float(rwlon[0]), float(rlat[j]),
                                  float(rwlon[j]), float(sub.sum()))
            mem = score_members(grids, sub_cor, rlat[:j + 1], rwlon[:j + 1], sub,
                                wps=wps, power_fn=power_fn, align=align,
                                hs_lim=hs_lim, tws_lim=tws_lim, p_lim=p_max)
            p_hat = float((mem["margin"] > 0).mean())

        r = evaluate_route_saturated(truth_wind, truth_wave, dep, rlat,
                                     op.working_to_signed(rwlon), seg,
                                     power_fn_host, wps=wps, p_max=p_max,
                                     max_hours=cycle_h, t_offset_h=elapsed)
        track["t_h"] += [elapsed + t for t in r["track"]["t_h"][1:]]
        track["lat"] += r["track"]["lat"][1:]
        track["lon"] += r["track"]["lon"][1:]

        energy += r["energy_mwh"]
        elapsed += r["actual_hours"]
        lat_now, lon_now = r["end_lat"], r["end_lon"]
        max_hs = max(max_hs, r["max_hs_m"])
        max_tws = max(max_tws, r["max_wind_mps"])
        max_p = max(max_p, r["max_power_kw"])
        legs.append(dict(leg=leg, cycle=cycle_t, planned_hours=remaining_h,
                         sailed_hours=r["actual_hours"], elapsed_h=elapsed,
                         energy_mwh=r["energy_mwh"], end_lat=lat_now,
                         end_lon=lon_now, max_hs_m=r["max_hs_m"],
                         max_wind_mps=r["max_wind_mps"],
                         saturated_frac=r["saturated_frac"], p_hat=p_hat,
                         objective_value=best_j, theta=best))
        if verbose:
            print(f"leg {leg:2d} {cycle_t:%m-%d %HZ}  {remaining_nm:6.0f} nm to go, "
                  f"plan {remaining_h:5.1f} h, sailed {r['actual_hours']:4.1f} h, "
                  f"{r['energy_mwh']:6.1f} MWh", flush=True)
        del grids
        if clear_jax_caches:
            jax.clear_caches()
            gc.collect()
        if r["arrived"]:
            arrived = True
            break

    return dict(objective=objective, energy_mwh=energy, actual_hours=elapsed,
                delay_h=elapsed - hours, arrived=arrived, max_hs_m=max_hs,
                max_wind_mps=max_tws, max_power_kw=max_p, n_legs=len(legs),
                track=track, legs=legs)
