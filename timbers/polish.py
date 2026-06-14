#!/usr/bin/env python
"""Gradient local-refinement polish (stage 2).

Given a converged route, give every waypoint a lateral offset along the route normal and
co-refine the speed profile; gentle manual-Adam on the same aligned cost (+ curvature reg);
keep the best SCORED-feasible iterate (seeded at t=0 so divergence is a no-op). Additive
on top of best-of-N. Grids threaded as TRACED args so the jit never bakes large constants.
"""

from __future__ import annotations

from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np

from . import optimizer as op
from .scoring import evaluate_route_full


def make_polisher(grids, land, cor, pen, wps, K, L, NSP, ALIGN, power_fn, power_fn_host,
                  hs_max=float("inf"), tws_max=float("inf"),
                  steps=600, lr0=3e-3, clip=0.05, lam_s=3e4):
    """``power_fn`` is the device (JAX) power model used in the gradient cost;
    ``power_fn_host`` is the NumPy power model used by the host scorer that picks
    the best scored-feasible iterate."""
    n_geo = 2 * (K - 2)
    dargs = (grids.dt_h, grids.nt, grids.lon_wrap)
    FIELDS = (grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos)
    AXES = (grids.wlat, grids.wlon, grids.slat, grids.slon)
    LAND = (land.mask, land.lat, land.wlon)
    end_mask = jnp.ones(L).at[0].set(0.0).at[-1].set(0.0)

    def cost(p, dep_off, P0, normals, fields, axes, land_arrs):
        d = p[:L] * end_mask
        seg_dt = op.time_alloc(p[L:], cor.hours, L, NSP)
        P = P0 + d[:, None] * normals
        c, *_ = op._route_cost(fields, axes, land_arrs, cor, *dargs,
                               P[:, 0], P[:, 1], seg_dt, dep_off, L, wps, pen,
                               power_fn, ALIGN)
        curv = d[2:] - 2 * d[1:-1] + d[:-2]
        return c + lam_s * jnp.sum(curv ** 2)

    vg = jax.jit(jax.value_and_grad(cost))

    def route_inputs(lat, wlon, theta):
        nlat, nwlon = cor.norm(np.asarray(lat), np.asarray(wlon))
        P0 = np.stack([nlat, nwlon], 1).astype(np.float32)
        tang = np.gradient(P0, axis=0)
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        speed0 = np.asarray(theta[n_geo:], np.float32)
        return jnp.asarray(P0), jnp.asarray(nrm.astype(np.float32)), jnp.asarray(speed0)

    def decode(p, P0, normals):
        d = (p[:L] * end_mask)
        P = P0 + d[:, None] * normals
        seg_dt = np.asarray(op.time_alloc(p[L:], cor.hours, L, NSP))
        lat, wlon = cor.denorm(np.asarray(P[:, 0]), np.asarray(P[:, 1]))
        return lat, wlon, seg_dt

    def polish(lat, wlon, seg_dt, theta, dep_off, dep, wind, wave):
        """Refine one route; returns a build-compatible result dict (>= the input route)."""
        P0, normals, speed0 = route_inputs(lat, wlon, theta)
        p = jnp.concatenate([jnp.zeros(L, jnp.float32), speed0])

        def score_p(pp):
            la, wl, sd = decode(pp, P0, normals)
            acc = np.concatenate([[0.0], np.cumsum(sd)])
            t = [dep + timedelta(hours=float(a)) for a in acc]
            s = evaluate_route_full(wind, wave, list(zip(t, la, op.working_to_signed(wl))),
                                    power_fn_host, wps=wps)
            return s, la, wl, sd

        s0, la0, wl0, sd0 = score_p(p)
        best = (s0["energy_mwh"], la0, wl0, sd0, s0) if (
            s0["max_hs_m"] <= hs_max and s0["max_wind_mps"] <= tws_max) else None
        m = v = jnp.zeros_like(p)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for tstep in range(1, steps + 1):
            _, g = vg(p, dep_off, P0, normals, FIELDS, AXES, LAND)
            gn = jnp.linalg.norm(g)
            g = jnp.where(gn > clip, g * clip / gn, g)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh, vh = m / (1 - b1 ** tstep), v / (1 - b2 ** tstep)
            p = p - lr0 * (1.0 - tstep / (steps + 1)) * mh / (jnp.sqrt(vh) + eps)
            if tstep % 50 == 0 or tstep == steps:
                s, la, wl, sd = score_p(p)
                if s["max_hs_m"] <= hs_max and s["max_wind_mps"] <= tws_max and \
                        (best is None or s["energy_mwh"] < best[0]):
                    best = (s["energy_mwh"], la, wl, sd, s)
        if best is None:                                   # nothing feasible -> keep input
            la, wl, sd, s = la0, wl0, sd0, s0
        else:
            _, la, wl, sd, s = best
        return dict(lat=la, wlon=wl, seg_dt=sd, theta=theta,
                    energy_mwh=s["energy_mwh"], max_hs_m=s["max_hs_m"],
                    max_wind_mps=s["max_wind_mps"], sailed_distance_nm=s["sailed_distance_nm"])

    return polish
