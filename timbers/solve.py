#!/usr/bin/env python
"""Production GPU-native backend: solve a whole corridor/case with batched sep-CMA-ES.

One chunked, vmapped GPU dispatch over (departures x seeds). The batch is CHUNKED
(cap per corridor) because a larger cost (more aligned timesteps M, larger popsize)
raises the per-instance working set and OOMs the GPU above some batch size.

Returns, per departure, a result dict compatible with scoring / track writing:
  {lat, wlon, seg_dt, theta, energy_mwh, max_hs_m, max_wind_mps, ..., restarts}.
"""

from __future__ import annotations

from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np

from . import cmaes as jc
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
    s = evaluate_route_full(
        wind, wave, list(zip(t, lat, op.working_to_signed(wlon))), power_fn_host, wps=wps
    )
    return dict(
        lat=lat,
        wlon=wlon,
        seg_dt=seg_dt,
        theta=np.asarray(theta),
        energy_mwh=s["energy_mwh"],
        max_hs_m=s["max_hs_m"],
        max_wind_mps=s["max_wind_mps"],
        max_power_kw=s["max_power_kw"],
        sailed_distance_nm=s["sailed_distance_nm"],
    )


def solve_corridor(
    cor,
    grids,
    land,
    pen,
    deps,
    wind,
    wave,
    *,
    wps,
    K,
    L,
    NSP,
    ALIGN,
    n_seeds,
    popsize,
    maxiter,
    power_fn,
    power_fn_host,
    hs_max=float("inf"),
    tws_max=float("inf"),
    p_max=float("inf"),
    sigma0=0.1,
    init="gc",
    spread=0.15,
    seed_tracks=None,
    sigma_warm=0.02,
    chunk=None,
    base_seed=0,
    topk=3,
    polish=False,
    land_sub=None,
):
    """Batched best-of-``n_seeds`` over all ``deps``. Returns list of result dicts.

    ``power_fn`` is the device (JAX) power model used in the GPU cost;
    ``power_fn_host`` is the NumPy power model used by the exact host scorer. Both
    have signature ``(tws, twa, swh, mwa, v, wps) -> kW`` (see ``examples/toy_power``).
    ``hs_max``/``tws_max``/``p_max`` are the hard feasibility limits used to
    select among seeds (default: unconstrained). ``p_max`` is a shaft-power
    ceiling in kW; a route needing more cannot hold its planned schedule. ``polish=True`` applies the local-refinement
    (gradient lateral+speed polish) to each selected route. ``land_sub`` sets how
    densely land is checked between track points (``None``: at most one raster
    cell apart, see :func:`op.land_substeps`; 1: at the points only, as v0.1).

    Cold restarts start from ``op.init_thetas(..., init, spread)`` at step
    ``sigma0``: by default the great circle; ``init="spread"`` bends restarts
    north and south of it, and ``init="chord"`` reproduces v0.1.

    ``seed_tracks`` warm-starts the search from known routes: one entry per
    departure, each ``None`` or a list of tracks ``(t_h, lat, lon)`` (hours after
    departure, signed longitude) running port to port over ``cor.hours``. Each
    track is fitted with :func:`op.fit_theta_to_track` and takes one of the
    ``n_seeds`` restarts, run at the smaller step ``sigma_warm`` so that it
    refines the seed's basin rather than leaving it; the remaining restarts start
    cold. The fitted seeds are also scored themselves, so a seeded result is
    never worse than its seed.

    Each result carries ``restarts``: per restart, its final penalized cost
    ``j``, ``seeded`` and decoded ``lat``/``wlon``, to show whether the restarts
    agree on a basin.
    """
    chunk = chunk or auto_chunk(cor, popsize, ALIGN)
    dim = 2 * (K - 2) + NSP
    fit, shared = op.build_fit(cor, grids, land, pen, K, NSP, L, ALIGN, wps, power_fn, land_sub)
    hp = jc.hyperparams(dim, popsize)
    D = len(deps)
    seed_tracks = [None] * D if seed_tracks is None else list(seed_tracks)
    if len(seed_tracks) != D:
        raise ValueError(f"seed_tracks needs one entry per departure ({D}), got {len(seed_tracks)}")
    seeds = [
        [op.fit_theta_to_track(cor, *tr, K=K, L=L, n_speed=NSP)[0] for tr in (st or [])]
        for st in seed_tracks
    ]
    if max(map(len, seeds), default=0) > n_seeds:
        raise ValueError(f"more seed tracks than restarts (n_seeds={n_seeds}) for a departure")
    x0s = np.empty((D, n_seeds, dim), np.float32)
    sig = np.full((D, n_seeds), sigma0, np.float32)
    seeded = np.zeros((D, n_seeds), bool)
    for d, sd in enumerate(seeds):
        n_s = len(sd)
        if n_s:
            x0s[d, :n_s], sig[d, :n_s], seeded[d, :n_s] = sd, sigma_warm, True
        x0s[d, n_s:] = op.init_thetas(cor, K, NSP, n_seeds - n_s, init, spread)
    solve = jc.make_solver(fit, x0s[0, 0], sigma0, hp, popsize, maxiter)
    x0s, sig = x0s.reshape(-1, dim), sig.reshape(-1)

    # Offsets on the device grids' own clock: grids cut to a window (from_era5
    # with start) do not start at the host grid's t0.
    dep_offs = np.array([grids.hours_after_t0(dp) for dp in deps], np.float32)
    B = D * n_seeds
    dep_offs_B = np.repeat(dep_offs, n_seeds)  # (B,)
    keys = jax.random.split(jax.random.PRNGKey(base_seed), B)  # (B,2)

    best_x = np.empty((B, dim), np.float32)
    best_f = np.empty(B, np.float32)
    for lo in range(0, B, chunk):  # chunk to stay in GPU mem
        hi = min(lo + chunk, B)
        bx, bf = solve(keys[lo:hi], jnp.asarray(dep_offs_B[lo:hi]), shared, x0s[lo:hi], sig[lo:hi])
        best_x[lo:hi] = np.asarray(bx)
        best_f[lo:hi] = np.asarray(bf)
    best_x = best_x.reshape(D, n_seeds, dim)
    best_f = best_f.reshape(D, n_seeds)

    # Pre-rank seeds by the solver's penalized cost (~ energy + feasibility) and exact-score
    # only the top ``topk`` per dep: cuts the host scorer ~n_seeds/topk x without a second
    # GPU kernel (a metrics jit can OOM host RAM for the heaviest configs).
    polisher = (
        pol.make_polisher(
            grids,
            land,
            cor,
            pen,
            wps,
            K,
            L,
            NSP,
            ALIGN,
            power_fn,
            power_fn_host,
            hs_max=hs_max,
            tws_max=tws_max,
            land_sub=land_sub,
        )
        if polish
        else None
    )
    out = []
    for d, dep in enumerate(deps):
        order = np.argsort(best_f[d])[:topk]
        cands = [
            _exact(x, dep, cor, wind, wave, K, L, NSP, wps, power_fn_host)
            for x in [best_x[d, s] for s in order] + seeds[d]
        ]
        feas = [
            c
            for c in cands
            if c["max_hs_m"] <= hs_max
            and c["max_wind_mps"] <= tws_max
            and c.get("max_power_kw", 0.0) <= p_max
        ]
        pool = feas or cands
        best = min(
            pool, key=lambda c: c["energy_mwh"] if feas else (c["max_hs_m"], c["energy_mwh"])
        )
        if polisher is not None:
            best = polisher(
                best["lat"],
                best["wlon"],
                best["seg_dt"],
                best["theta"],
                float(dep_offs[d]),
                dep,
                wind,
                wave,
            )
        best["restarts"] = [
            dict(j=float(best_f[d, s]), seeded=bool(seeded[d, s]))
            | dict(zip(("lat", "wlon"), op.decode_route(best_x[d, s], cor, K, L, NSP)[:2]))
            for s in range(n_seeds)
        ]
        out.append(best)
    return out
