#!/usr/bin/env python
"""Production GPU-native backend: solve a whole corridor/case with batched sep-CMA-ES.

One chunked, vmapped GPU dispatch over (departures x seeds). The batch is CHUNKED
(cap per corridor) because a larger cost (more aligned timesteps M, larger popsize)
raises the per-instance working set and OOMs the GPU above some batch size.

Returns, per departure, a result dict compatible with scoring / track writing:
  {lat, wlon, seg_dt, theta, energy_mwh, max_hs_m, max_wind_mps}.
"""

from __future__ import annotations

from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np

from . import cmaes as jc
from . import fitness as jrf
from . import optimizer as op
from . import polish as pol
from .scoring import evaluate_route_full

# vmap working set scales ~ popsize * M (M = aligned timesteps). The budget caps
# popsize * M * chunk; tune down if the GPU OOMs (leaves margin for the grids +
# allocator fragmentation).
_CHUNK_BUDGET = 1.0e7


def auto_chunk(cor, popsize, align):
    M = max(1, round(cor.hours / align))
    return int(np.clip(_CHUNK_BUDGET / (popsize * M), 1, 64))


def _exact(theta, dep, cor, wind, wave, K, L, NSP, wps, power_fn_host):
    """Exact host scorer on a single chosen route -> build-compatible result dict."""
    lat, wlon, seg_dt = op.decode_route(np.asarray(theta), cor, K, L, NSP)
    acc = np.concatenate([[0.0], np.cumsum(seg_dt)])
    t = [dep + timedelta(hours=float(a)) for a in acc]
    s = evaluate_route_full(wind, wave, list(zip(t, lat, op.working_to_signed(wlon))),
                            power_fn_host, wps=wps)
    return dict(lat=lat, wlon=wlon, seg_dt=seg_dt, theta=np.asarray(theta),
                energy_mwh=s["energy_mwh"], max_hs_m=s["max_hs_m"],
                max_wind_mps=s["max_wind_mps"], sailed_distance_nm=s["sailed_distance_nm"])


def solve_corridor(cor, grids, land, pen, deps, wind, wave, *, wps, K, L, NSP, ALIGN,
                   n_seeds, popsize, maxiter, power_fn, power_fn_host,
                   hs_max=float("inf"), tws_max=float("inf"),
                   sigma0=0.1, chunk=None, base_seed=0, topk=3, polish=False):
    """Batched best-of-``n_seeds`` over all ``deps``. Returns list of result dicts.

    ``power_fn`` is the device (JAX) power model used in the GPU cost;
    ``power_fn_host`` is the NumPy power model used by the exact host scorer. Both
    have signature ``(tws, twa, swh, mwa, v, wps) -> kW`` (see ``examples/toy_power``).
    ``hs_max``/``tws_max`` are the hard feasibility limits used to select among
    seeds (default: unconstrained). ``polish=True`` applies the local-refinement
    (gradient lateral+speed polish) to each selected route.
    """
    chunk = chunk or auto_chunk(cor, popsize, ALIGN)
    dim = 2 * (K - 2) + NSP
    fit, x0, shared = jrf.build_fit(cor, grids, land, pen, K, NSP, L, ALIGN, wps, power_fn)
    hp = jc.hyperparams(dim, popsize)
    solve = jc.make_solver(fit, x0, sigma0, hp, popsize, maxiter)

    D = len(deps)
    dep_offs = np.array([float((np.datetime64(dp) - wind["t0"]) / np.timedelta64(1, "h"))
                         for dp in deps], np.float32)
    B = D * n_seeds
    dep_offs_B = np.repeat(dep_offs, n_seeds)                       # (B,)
    keys = jax.random.split(jax.random.PRNGKey(base_seed), B)       # (B,2)

    best_x = np.empty((B, dim), np.float32)
    best_f = np.empty(B, np.float32)
    for lo in range(0, B, chunk):                                   # chunk to stay in GPU mem
        hi = min(lo + chunk, B)
        bx, bf = solve(keys[lo:hi], jnp.asarray(dep_offs_B[lo:hi]), shared)
        best_x[lo:hi] = np.asarray(bx); best_f[lo:hi] = np.asarray(bf)
    best_x = best_x.reshape(D, n_seeds, dim)
    best_f = best_f.reshape(D, n_seeds)

    # Pre-rank seeds by the solver's penalized cost (~ energy + feasibility) and exact-score
    # only the top ``topk`` per dep: cuts the host scorer ~n_seeds/topk x without a second
    # GPU kernel (a metrics jit can OOM host RAM for the heaviest configs).
    polisher = (pol.make_polisher(grids, land, cor, pen, wps, K, L, NSP, ALIGN,
                                  power_fn, power_fn_host, hs_max=hs_max, tws_max=tws_max)
                if polish else None)
    out = []
    for d, dep in enumerate(deps):
        order = np.argsort(best_f[d])[:topk]
        cands = [_exact(best_x[d, s], dep, cor, wind, wave, K, L, NSP, wps, power_fn_host)
                 for s in order]
        feas = [c for c in cands if c["max_hs_m"] <= hs_max and c["max_wind_mps"] <= tws_max]
        pool = feas or cands
        best = min(pool, key=lambda c: c["energy_mwh"] if feas else
                   (c["max_hs_m"], c["energy_mwh"]))
        if polisher is not None:
            best = polisher(best["lat"], best["wlon"], best["seg_dt"], best["theta"],
                            float(dep_offs[d]), dep, wind, wave)
        out.append(best)
    return out
