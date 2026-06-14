#!/usr/bin/env python
"""Reusable route-cost fitness for the GPU-native sep-CMA solver (timbers.cmaes.make_solver).

Builds a (fit, x0, shared) triple for any corridor/config. ``fit(X, cargs)`` with
cargs=(dep_off, fields, axes, land_arrs); the large grid arrays live in ``shared`` and are
passed as TRACED, UNMAPPED vmap args (never closed-over) so jitting the solver doesn't bake
them in as multi-GB host constants. wps/pen/cor/ALIGN are small -> fine as constants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import optimizer as op


def build_fit(cor, grids, land, pen, K, NSP, L, ALIGN, wps, power_fn):
    o_n = jnp.asarray(cor.norm(cor.o_lat, cor.o_wlon), jnp.float32)
    d_n = jnp.asarray(cor.norm(cor.d_lat, cor.d_wlon), jnp.float32)
    r = jnp.linspace(0.0, 1.0, L, dtype=jnp.float32)
    ngeo = 2 * (K - 2)
    dargs = (grids.dt_h, grids.nt, grids.lon_wrap)
    fields = (grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos)
    axes = (grids.wlat, grids.wlon, grids.slat, grids.slon)
    land_arrs = (land.mask, land.lat, land.wlon)
    shared = (fields, axes, land_arrs)

    def one(theta, dep_off, fld, ax, lnd):
        interior = theta[:ngeo].reshape(K - 2, 2)
        ctrl = jnp.concatenate([o_n[None, :], interior, d_n[None, :]], axis=0)
        pts = op.bezier(ctrl, r)
        seg_dt = op.time_alloc(theta[ngeo:], cor.hours, L, NSP)
        cost, *_ = op._route_cost(fld, ax, lnd, cor, *dargs,
                                  pts[:, 0], pts[:, 1], seg_dt, dep_off, L, wps, pen,
                                  power_fn, ALIGN)
        return cost

    def fit(X, cargs):
        dep_off, fld, ax, lnd = cargs
        return jax.vmap(one, in_axes=(0, None, None, None, None))(X, dep_off, fld, ax, lnd)

    x0 = jnp.asarray(op.gc_init_theta(cor, K, NSP), jnp.float32)
    return fit, x0, shared
