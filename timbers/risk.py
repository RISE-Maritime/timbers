#!/usr/bin/env python
"""Risk-aware extension: perturbation-fragility scoring and chance-constrained cost.

Both treat forecast error as a perturbation ensemble over the weather sampling
(spatial shift dlat/dlon, temporal shift dt_h, amplitude scale on Hs/wind)
while the route itself is held fixed:

* ``make_scorer`` re-scores a FIXED route under each perturbation, quantifying
  its fragility (a route optimized to ride just below a wave-height limit on the
  reanalysis violates it with high probability the moment the weather differs).
  Returns, per perturbation: energy, max_hs, max_tws.
* ``make_robust_cost`` builds an optimizable chance-constrained objective:
  NOMINAL energy + a risk penalty over the ensemble, penalizing Hs/TWS
  exceedance from the HARD limits (``hs_lim``/``us_lim``) -- the safety buffer is
  set by the local forecast spread instead of a hand-tuned ``hs_soft``.

      J(theta) = E_nominal
               + lam_env * mean_pert[exp(aH*(Hs_p-hs_lim)+) + exp(aU*(TWS_p-us_lim)+) - 2]
               + lam_land * land

The route geometry/speed is shared across perturbations; only the weather
sampling is perturbed. vmapped over (population x ensemble) on the GPU.

Both entry points take an injected ``power_fn(tws, twa_deg, swh, mwa_deg, v, wps)
-> kW`` (JAX arrays). See ``examples/toy_power.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import model as jm
from . import optimizer as op


# ---------------------------------------------------------------------------
# Perturbation-fragility (fixed route, perturbed weather)
# ---------------------------------------------------------------------------
def make_scorer(grids, cor, wps, power_fn, align=0.25):
    """Return scorer(lat, wlon_signed, seg_dt, dep_off, perts) -> (E, maxHs, maxTws) per pert.

    ``perts`` is (Pn, 5): columns (dlat, dlon, dt_h, hs_scale, wind_scale). The route is
    resampled to a uniform-time grid (scorer-aligned) so max_hs matches the scorer's max.
    """
    M = int(round(cor.hours / align))
    dt_h, nt, lon_wrap = grids.dt_h, grids.nt, grids.lon_wrap
    # grids threaded as TRACED args (not closed-over) so the jit doesn't bake ~4GB constants.
    SHARED = ((grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos),
              (grids.wlat, grids.wlon, grids.slat, grids.slon))

    def one(lat, wlon_s, seg_dt, dep_off, pert, fields, axes):
        u10f, v10f, swhf, msf, mcf = fields
        wlat, wlon_ax, slat, slon = axes
        dlat, dlon, dtim, hs_sc, w_sc = pert
        t_cum = jnp.concatenate([jnp.zeros(1, lat.dtype), jnp.cumsum(seg_dt)])
        tau = jnp.linspace(0.0, cor.hours, M + 1).astype(lat.dtype)
        rlat = jnp.interp(tau, t_cum, lat)
        rlon = jnp.interp(tau, t_cum, wlon_s)            # signed lon
        seg = jnp.full((M,), cor.hours / M, lat.dtype)
        # ship geometry/speed from the ACTUAL route (unperturbed)
        dist = jm._haversine_m(rlat[:-1], rlon[:-1], rlat[1:], rlon[1:])
        v = dist / (seg * 3600.0)
        bearing = jm._bearing_deg(rlat[:-1], rlon[:-1], rlat[1:], rlon[1:])
        mid_lat = (rlat[:-1] + rlat[1:]) / 2
        mid_lon = (rlon[:-1] + rlon[1:]) / 2
        cum = jnp.cumsum(seg)
        smid = dep_off + cum - seg / 2
        # PERTURBED weather query: shifted position/time, scaled amplitude
        qlat = mid_lat + dlat
        qlon = mid_lon + dlon
        qlon_w = jnp.where(qlon < 0, qlon + 360.0, qlon) if lon_wrap else qlon
        qt = smid + dtim
        ti, tf = jm._time_index(qt, dt_h, nt)
        u10 = jm._interp(u10f, wlat, wlon_ax, qlat, qlon_w, ti, tf, nt) * w_sc
        v10 = jm._interp(v10f, wlat, wlon_ax, qlat, qlon_w, ti, tf, nt) * w_sc
        swh = jm._interp(swhf, slat, slon, qlat, qlon_w, ti, tf, nt) * hs_sc
        ms = jm._interp(msf, slat, slon, qlat, qlon_w, ti, tf, nt)
        mc = jm._interp(mcf, slat, slon, qlat, qlon_w, ti, tf, nt)
        mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)
        tws = jnp.sqrt(u10 ** 2 + v10 ** 2)
        wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
        twa = jnp.mod(wind_from - bearing, 360.0)
        mwa = jnp.mod(mwd - bearing, 360.0)
        p = power_fn(tws, twa, swh, mwa, v, wps)
        energy = jnp.sum(p * seg) / 1000.0
        return jnp.stack([energy, jnp.max(swh), jnp.max(tws)])

    @jax.jit
    def _run(lat, wlon_s, seg_dt, dep_off, perts, fields, axes):
        out = jax.vmap(one, in_axes=(None, None, None, None, 0, None, None))(
            lat, wlon_s, seg_dt, dep_off, perts, fields, axes)
        return out[:, 0], out[:, 1], out[:, 2]

    def scorer(lat, wlon_s, seg_dt, dep_off, perts):
        f, a = SHARED
        return _run(lat, wlon_s, seg_dt, dep_off, jnp.asarray(perts), f, a)

    return scorer


def perturbation_grid(dlat=(0.0,), dlon=(0.0,), dt=(0.0,), hs=(1.0,), wind=(1.0,)):
    """Cartesian product of perturbation axes -> (Pn, 5)."""
    import itertools

    import numpy as np
    rows = [tuple(p) for p in itertools.product(dlat, dlon, dt, hs, wind)]
    return np.array(rows, np.float32)

# ---------------------------------------------------------------------------
# Chance-constrained route cost
# ---------------------------------------------------------------------------
def make_robust_cost(grids, land, cor, L, wps, K, n_speed, align, perts, power_fn, *,
                     hs_lim=7.0, us_lim=20.0, aH=8.0, aU=3.0,
                     lam_env=30.0, lam_land=100.0, nominal_idx=0):
    """fn(theta_batch, dep_off) -> J (Ppop,). ``perts`` (Pn,5): dlat,dlon,dt,hs_sc,wind_sc."""
    o_n = jnp.asarray(cor.norm(cor.o_lat, cor.o_wlon), jnp.float32)
    d_n = jnp.asarray(cor.norm(cor.d_lat, cor.d_wlon), jnp.float32)
    rr = jnp.linspace(0.0, 1.0, L, dtype=jnp.float32)
    n_geo = 2 * (K - 2)
    M = int(round(cor.hours / align))
    dt_h, nt, lon_wrap = grids.dt_h, grids.nt, grids.lon_wrap
    pert_arr = jnp.asarray(perts, jnp.float32)
    FIELDS = (grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos)
    AXES = (grids.wlat, grids.wlon, grids.slat, grids.slon)
    LAND = (land.mask, land.lat, land.wlon)

    def one(theta, dep_off, fields, axes, land_arrs):
        u10f, v10f, swhf, msf, mcf = fields
        wlat, wlon_ax, slat, slon = axes
        lmask, llat, lwlon = land_arrs
        interior = theta[:n_geo].reshape(K - 2, 2)
        ctrl = jnp.concatenate([o_n[None, :], interior, d_n[None, :]], axis=0)
        pts = op.bezier(ctrl, rr)
        seg_dt0 = op.time_alloc(theta[n_geo:], cor.hours, L, n_speed)
        lat, wlon = cor.denorm(pts[:, 0], pts[:, 1])            # working coords
        # scorer-align resample to uniform time
        t_cum = jnp.concatenate([jnp.zeros(1, lat.dtype), jnp.cumsum(seg_dt0)])
        tau = jnp.linspace(0.0, cor.hours, M + 1).astype(lat.dtype)
        rlat = jnp.interp(tau, t_cum, lat)
        rlon = jnp.interp(tau, t_cum, wlon)                     # working lon
        seg = jnp.full((M,), cor.hours / M, lat.dtype)
        glon = op.working_to_signed(rlon)                      # signed for weather/dist
        dist = jm._haversine_m(rlat[:-1], glon[:-1], rlat[1:], glon[1:])
        v = dist / (seg * 3600.0)
        bearing = jm._bearing_deg(rlat[:-1], glon[:-1], rlat[1:], glon[1:])
        mid_lat = (rlat[:-1] + rlat[1:]) / 2
        mid_lon = (glon[:-1] + glon[1:]) / 2                    # signed
        cum = jnp.cumsum(seg)
        smid = dep_off + cum - seg / 2

        def per_pert(pert):
            dlat, dlon, dtim, hs_sc, w_sc = pert
            qlat = mid_lat + dlat
            qlon = mid_lon + dlon
            qlon_w = jnp.where(qlon < 0, qlon + 360.0, qlon) if lon_wrap else qlon
            ti, tf = jm._time_index(smid + dtim, dt_h, nt)
            u10 = jm._interp(u10f, wlat, wlon_ax, qlat, qlon_w, ti, tf, nt) * w_sc
            v10 = jm._interp(v10f, wlat, wlon_ax, qlat, qlon_w, ti, tf, nt) * w_sc
            swh = jm._interp(swhf, slat, slon, qlat, qlon_w, ti, tf, nt) * hs_sc
            ms = jm._interp(msf, slat, slon, qlat, qlon_w, ti, tf, nt)
            mc = jm._interp(mcf, slat, slon, qlat, qlon_w, ti, tf, nt)
            mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)
            tws = jnp.sqrt(u10 ** 2 + v10 ** 2)
            wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
            twa = jnp.mod(wind_from - bearing, 360.0)
            mwa = jnp.mod(mwd - bearing, 360.0)
            p = power_fn(tws, twa, swh, mwa, v, wps)
            energy = jnp.sum(p * seg) / 1000.0
            env = jnp.sum(jnp.exp(aH * jnp.maximum(swh - hs_lim, 0.0))
                          + jnp.exp(aU * jnp.maximum(tws - us_lim, 0.0)) - 2.0)
            return energy, env

        energies, envs = jax.vmap(per_pert)(pert_arr)          # (Pn,), (Pn,)
        nominal_E = energies[nominal_idx]
        risk = jnp.mean(envs)                                  # expected ensemble exceedance
        p_land = jnp.sum(op._sample_mask(lmask, llat, lwlon, rlat, rlon))
        return nominal_E + lam_env * risk + lam_land * p_land

    batched = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None)))
    return lambda tb, doff: batched(tb, jnp.float32(doff), FIELDS, AXES, LAND)
