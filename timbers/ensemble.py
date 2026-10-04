"""Route optimisation and scoring over a forecast ensemble.

Where :mod:`timbers.risk` builds a surrogate ensemble by perturbing one weather
field, this module works with a real one: fields with a member axis, such as the
51 members of ECMWF ENS. One route cost serves four objectives that differ only
in how the per-member outcomes are reduced to a scalar:

    ==================  =======================  =========================
                        safety from member 0     safety over all members
    ==================  =======================  =========================
    cost from member 0  ``deterministic``        ``chance_constrained``
    cost, member mean   ``expected_value``       ``joint``
    ==================  =======================  =========================

Member 0 is the nominal member (the control forecast in ENS). Parameterisation,
resampling, interpolation, power model and land penalty are shared, so a
difference between objectives is a difference in formulation.

The expectation is taken over per-member costs, not over the weather field:
power is convex in wave height, so averaging the weather first would understate
the cost of exactly the storm cases an ensemble is meant to represent.

Forecast steps need not be uniform (ENS is 3-hourly to 144 h, then 6-hourly);
time interpolation searches the actual step vector.

Typical use::

    grids = EnsembleGrids(wind, wave, steps)          # fields (member, time, y, x)
    fit, shared = make_ensemble_cost(grids, land, cor, objective="joint", ...)
    best, best_j = cmaes.run(x0, fit, 0.1, hp, pop, iters, key,
                             (jnp.float32(dep_off), *shared))
    lat, wlon, seg = optimizer.decode_route(best, cor, K, L, n_speed)
    members = score_members(grids, cor, lat, wlon, seg, ...)   # per-member outcomes
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import model as jm
from . import optimizer as op

OBJECTIVES = ("deterministic", "chance_constrained", "expected_value", "joint")


class EnsembleGrids:
    """Device-resident weather fields with a member axis.

    The ensemble counterpart of :class:`timbers.model.DeviceGrids`: every field
    is ``(member, time, lat, lon)`` and the time axis is given by ``steps``, the
    lead time of each slice in hours since the forecast's base time, which may
    be non-uniform. ``wind`` holds ``u10``, ``v10``, ``lat``, ``lon``; ``wave``
    holds ``swh``, ``mwd`` (degrees), ``lat``, ``lon``. Wind and wave may be on
    different spatial grids.

    ``consume=True`` removes the field arrays from ``wind`` and ``wave`` as they
    are moved to the device, which roughly halves peak host memory for a large
    ensemble; it is off by default because it empties the caller's dicts.

    Fields must be finite. Land-masked wave cells are typically NaN in decoded
    GRIB; fill them first, for example with
    :func:`timbers.seafill.fill_from_nearest_sea`. A NaN objective would not
    raise inside the optimizer: the candidate would be silently ranked out.
    """

    def __init__(self, wind: dict, wave: dict, steps, consume: bool = False):
        take = (lambda d, k: d.pop(k)) if consume else (lambda d, k: d[k])
        self.steps = jnp.asarray(np.asarray(steps, np.float32))
        self.wlat = jnp.asarray(wind["lat"], jnp.float32)
        self.wlon = jnp.asarray(wind["lon"], jnp.float32)
        self.slat = jnp.asarray(wave["lat"], jnp.float32)
        self.slon = jnp.asarray(wave["lon"], jnp.float32)
        self.u10 = jnp.asarray(take(wind, "u10"), jnp.float32)
        self.v10 = jnp.asarray(take(wind, "v10"), jnp.float32)
        self.swh = jnp.asarray(take(wave, "swh"), jnp.float32)
        # Direction as sine and cosine so interpolation does not wrap at 360.
        # Computed in float32 one array at a time to bound host memory.
        mwd = np.radians(np.asarray(take(wave, "mwd")).astype(np.float32, copy=False))
        self.mwd_sin = jnp.asarray(np.sin(mwd, dtype=np.float32))
        self.mwd_cos = jnp.asarray(np.cos(mwd, dtype=np.float32))
        del mwd
        for name, arr in (("u10", self.u10), ("v10", self.v10), ("swh", self.swh),
                          ("mwd_sin", self.mwd_sin), ("mwd_cos", self.mwd_cos)):
            if not bool(jnp.all(jnp.isfinite(arr))):
                n = int(jnp.sum(~jnp.isfinite(arr)))
                raise ValueError(f"{name} has {n:,} non-finite values; fill land-masked "
                                 "cells before use (timbers.seafill)")
        self.n_members = self.u10.shape[0]
        self.nt = self.u10.shape[1]
        self.lon_wrap = bool(np.asarray(wind["lon"])[0] >= 0
                             and np.asarray(wind["lon"])[-1] > 180)


def as_ensemble(wind: dict, wave: dict, start, hours: float) -> EnsembleGrids:
    """A single gridded field as a one-member :class:`EnsembleGrids`.

    ``wind`` and ``wave`` are grids as returned by :func:`timbers.era5.load_era5`
    (uniform ``dt_h``). The result starts at ``start`` (a datetime) and covers
    ``hours``, so a route can be optimised on known weather (for example ERA5,
    to obtain a perfect-information solution) with the same cost and scoring
    code as a forecast ensemble.
    """
    t0 = np.datetime64(start.replace(tzinfo=None), "s")
    out = {}
    for name, g, keys in (("wind", wind, ("u10", "v10")), ("wave", wave, ("swh", "mwd"))):
        dt = float(g["dt_h"])
        i0 = int(round(float((t0 - g["t0"]) / np.timedelta64(1, "h")) / dt))
        if i0 < 0:
            raise ValueError(f"{name} grid starts after {start}")
        n = int(hours / dt) + 3
        out[name] = {k: np.asarray(g[k][i0:i0 + n])[None] for k in keys}
        out[name]["lat"], out[name]["lon"] = g["lat"], g["lon"]
    n = min(out["wind"]["u10"].shape[1], out["wave"]["swh"].shape[1])
    for side, keys in (("wind", ("u10", "v10")), ("wave", ("swh", "mwd"))):
        for k in keys:
            out[side][k] = out[side][k][:, :n]
    steps = np.arange(n, dtype=np.float32) * float(wind["dt_h"])
    return EnsembleGrids(out["wind"], out["wave"], steps)


# --- building blocks ----------------------------------------------------------
# Penalties grow exponentially up to a knee and linearly beyond it, with the
# slope matched at the join: sharp near a limit, where sharpness herds routes
# against the boundary, and never flat however bad a candidate is. A clipped
# exponential would overflow-protect but leave a plateau with no ranking signal.
_KNEE = 12.0
_KNEE_VAL = float(np.exp(_KNEE))


def _sexp(x):
    """exp(x) below the knee, linear continuation above; continuous slope."""
    lo = jnp.exp(jnp.minimum(x, _KNEE))
    return jnp.where(x <= _KNEE, lo, _KNEE_VAL * (1.0 + (x - _KNEE)))


# The number of integration points M fixes array shapes. Snapping it to a
# geometric ladder bounds the number of distinct compiled kernels when the same
# cost is built for many passages of different length (as in re-planning, where
# the remaining time shrinks every cycle), so the compilation cache is reused
# instead of growing. The step lands within sqrt(1.5) of the requested ``align``.
_M_LADDER = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536)


def _quantised_M(hours, align):
    m = hours / align
    return min(_M_LADDER, key=lambda c: abs(math.log(c / m)))


def time_index(hours, steps):
    """Bracketing index and fraction of ``hours`` on an ascending step vector."""
    nt = steps.shape[0]
    ti = jnp.clip(jnp.searchsorted(steps, hours, side="right") - 1, 0, nt - 2)
    t0, t1 = steps[ti], steps[ti + 1]
    return ti, jnp.clip((hours - t0) / jnp.maximum(t1 - t0, 1e-6), 0.0, 1.0)


def _cvar(x, eps):
    """Mean of the worst ``ceil(eps * M)`` members (CVaR at level 1 - eps)."""
    k = max(1, int(np.ceil(eps * x.shape[-1])))
    return jnp.mean(jnp.sort(x, axis=-1)[..., -k:], axis=-1)


def _p_hat(margins, sharpness):
    """Smoothed fraction of members whose margin is positive."""
    return jnp.mean(jax.nn.sigmoid(sharpness * margins), axis=-1)


def make_ensemble_cost(grids, land, cor, *, objective, L, K, n_speed, align, wps,
                       power_fn, hs_lim, tws_lim, p_lim=float("inf"), eps=0.1,
                       lam_env=30.0, lam_land=1e6, a_env=6.0, soft_frac=0.93,
                       safety_mode="prob", sharpness=50.0):
    """Return ``(fit, shared)`` for :func:`timbers.cmaes.run`.

    ``fit(theta_batch, cargs)`` with ``cargs = (dep_off, *shared)``, where
    ``dep_off`` is the departure time in hours after the ensemble's base time.
    The fields travel in ``shared`` rather than being closed over, so they are
    not baked into the compiled generation loop as constants (which would hold
    them in device memory twice).

    The route is resampled to a uniform time grid of about ``align`` hours, as
    the scorer integrates, so the optimised quantity is the scored one. For each
    member the cost computes energy (MWh), the worst normalised seakeeping
    margin ``max(Hs/hs_lim, TWS/tws_lim) - 1`` along the route, and a soft
    penalty that rises from ``soft_frac`` of each limit (Hs, TWS and, if finite,
    shaft power ``p_lim``). The objective then combines them:

    * cost: member 0's energy, or the member mean (``expected_value``,
      ``joint``);
    * safety: member 0's soft penalty, or an ensemble constraint
      (``chance_constrained``, ``joint``) penalising the exceedance of ``eps``.

    ``safety_mode`` selects the ensemble constraint. ``"prob"`` (default)
    constrains the smoothed fraction of members that breach a limit, so ``eps``
    is the violation probability. ``"cvar"`` constrains the CVaR at level
    ``1 - eps`` of the member margins, the standard convex relaxation; it bounds
    the violation probability only when it is satisfied, and where the
    constraint is infeasible it measures tail severity rather than frequency.
    ``eps`` is resolved only to ``1 / n_members``.

    Shaft power enters only the soft penalty, never the seakeeping margin: a
    power excess makes a voyage slow, not unsafe, and is better represented as
    late arrival (:func:`timbers.scoring.evaluate_route_saturated`).

    ``land`` is a :class:`timbers.optimizer.DeviceLand`; the land term is the
    summed raster along the route, zero at sea, and ``lam_land = 1e6`` makes it
    a hard constraint under rank-based selection.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; one of {OBJECTIVES}")
    if safety_mode not in ("prob", "cvar"):
        raise ValueError(f"unknown safety_mode {safety_mode!r}")
    use_ensemble_cost = objective in ("expected_value", "joint")
    use_ensemble_safety = objective in ("chance_constrained", "joint")

    o_n = jnp.asarray(cor.norm(cor.o_lat, cor.o_wlon), jnp.float32)
    d_n = jnp.asarray(cor.norm(cor.d_lat, cor.d_wlon), jnp.float32)
    rr = jnp.linspace(0.0, 1.0, L, dtype=jnp.float32)
    n_geo = 2 * (K - 2)
    M = _quantised_M(cor.hours, align)
    # The penalties are sums over the M points while energy is time-weighted;
    # rescaling by the M the requested align implies keeps the penalty weight
    # independent of where M lands on the ladder.
    pen_scale = (cor.hours / align) / M
    lon_wrap = grids.lon_wrap
    finite_p = bool(np.isfinite(p_lim))

    fields = (grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos)
    axes = (grids.wlat, grids.wlon, grids.slat, grids.slon, grids.steps)
    land_arrs = (land.mask, land.lat, land.wlon)

    def one(theta, dep_off, fields, axes, land_arrs):
        wlat, wlon_ax, slat, slon, steps = axes
        lmask, llat, lwlon = land_arrs
        interior = theta[:n_geo].reshape(K - 2, 2)
        ctrl = jnp.concatenate([o_n[None, :], interior, d_n[None, :]], axis=0)
        pts = op.bezier(ctrl, rr)
        seg_dt0 = op.time_alloc(theta[n_geo:], cor.hours, L, n_speed)
        lat, wlon = cor.denorm(pts[:, 0], pts[:, 1])

        t_cum = jnp.concatenate([jnp.zeros(1, lat.dtype), jnp.cumsum(seg_dt0)])
        tau = jnp.linspace(0.0, cor.hours, M + 1).astype(lat.dtype)
        rlat = jnp.interp(tau, t_cum, lat)
        rlon = jnp.interp(tau, t_cum, wlon)
        seg = jnp.full((M,), cor.hours / M, lat.dtype)

        glon = op.working_to_signed(rlon)
        v = jm._haversine_m(rlat[:-1], glon[:-1], rlat[1:], glon[1:]) / (seg * 3600.0)
        bearing = jm._bearing_deg(rlat[:-1], glon[:-1], rlat[1:], glon[1:])
        mid_lat = (rlat[:-1] + rlat[1:]) / 2
        mid_lon = (glon[:-1] + glon[1:]) / 2
        mid_lon_q = jnp.where(mid_lon < 0, mid_lon + 360.0, mid_lon) if lon_wrap else mid_lon
        ti, tf = time_index(dep_off + jnp.cumsum(seg) - seg / 2, steps)

        def per_member(u10m, v10m, swhm, msm, mcm):
            nt = u10m.shape[0]
            u10 = jm._interp(u10m, wlat, wlon_ax, mid_lat, mid_lon_q, ti, tf, nt)
            v10 = jm._interp(v10m, wlat, wlon_ax, mid_lat, mid_lon_q, ti, tf, nt)
            swh = jm._interp(swhm, slat, slon, mid_lat, mid_lon_q, ti, tf, nt)
            ms = jm._interp(msm, slat, slon, mid_lat, mid_lon_q, ti, tf, nt)
            mc = jm._interp(mcm, slat, slon, mid_lat, mid_lon_q, ti, tf, nt)
            mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)
            tws = jnp.sqrt(u10 ** 2 + v10 ** 2)
            wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
            p = power_fn(tws, jnp.mod(wind_from - bearing, 360.0), swh,
                         jnp.mod(mwd - bearing, 360.0), v, wps)
            energy = jnp.sum(p * seg) / 1000.0
            margin = jnp.max(jnp.stack([swh / hs_lim, tws / tws_lim])) - 1.0
            terms = (_sexp(a_env * jnp.maximum(swh / (soft_frac * hs_lim) - 1.0, 0.0))
                     + _sexp(a_env * jnp.maximum(tws / (soft_frac * tws_lim) - 1.0, 0.0)) - 2.0)
            if finite_p:
                terms = terms + _sexp(a_env * jnp.maximum(p / (soft_frac * p_lim) - 1.0, 0.0)) - 1.0
            return energy, margin, pen_scale * jnp.sum(terms)

        energies, margins, softs = jax.vmap(per_member)(*fields)
        cost = jnp.mean(energies) if use_ensemble_cost else energies[0]
        if use_ensemble_safety:
            if safety_mode == "cvar":
                excess = jnp.maximum(_cvar(margins, eps), 0.0)
            else:
                excess = jnp.maximum(_p_hat(margins, sharpness) - eps, 0.0)
            risk = _sexp(a_env * excess) - 1.0
        else:
            risk = softs[0]
        p_land = pen_scale * jnp.sum(op._sample_mask(lmask, llat, lwlon, rlat, rlon))
        return cost + lam_env * risk + lam_land * p_land

    batched = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None)))

    def fit(theta_batch, cargs):
        dep_off, f, a, la = cargs
        return batched(theta_batch, dep_off, f, a, la)

    return fit, (fields, axes, land_arrs)


def score_members(grids, cor, lat, wlon, seg_dt, *, wps, power_fn, align,
                  hs_lim, tws_lim, p_lim=float("inf"), dep_off=0.0):
    """Per-member outcomes of a fixed route, sampled as the cost samples it.

    ``lat``, ``wlon``, ``seg_dt`` as returned by
    :func:`timbers.optimizer.decode_route`. Returns host arrays over members:
    ``energy_mwh``, ``max_hs``, ``max_tws``, ``max_power``, ``margin`` (worst
    normalised Hs/TWS margin; positive means a breach) and ``power_margin``.
    ``(margin > 0).mean()`` is the ensemble's breach probability for the route.
    """
    lat = jnp.asarray(lat, jnp.float32)
    wlon = jnp.asarray(wlon, jnp.float32)
    seg_dt = jnp.asarray(seg_dt, jnp.float32)
    M = _quantised_M(cor.hours, align)
    t_cum = jnp.concatenate([jnp.zeros(1, lat.dtype), jnp.cumsum(seg_dt)])
    tau = jnp.linspace(0.0, cor.hours, M + 1).astype(lat.dtype)
    rlat = jnp.interp(tau, t_cum, lat)
    rlon = jnp.interp(tau, t_cum, wlon)
    seg = jnp.full((M,), cor.hours / M, lat.dtype)
    glon = op.working_to_signed(rlon)
    v = jm._haversine_m(rlat[:-1], glon[:-1], rlat[1:], glon[1:]) / (seg * 3600.0)
    bearing = jm._bearing_deg(rlat[:-1], glon[:-1], rlat[1:], glon[1:])
    mid_lat = (rlat[:-1] + rlat[1:]) / 2
    mid_lon = (glon[:-1] + glon[1:]) / 2
    mid_lon_q = jnp.where(mid_lon < 0, mid_lon + 360.0, mid_lon) if grids.lon_wrap else mid_lon
    ti, tf = time_index(dep_off + jnp.cumsum(seg) - seg / 2, grids.steps)

    def one(u10m, v10m, swhm, msm, mcm):
        nt = u10m.shape[0]
        u10 = jm._interp(u10m, grids.wlat, grids.wlon, mid_lat, mid_lon_q, ti, tf, nt)
        v10 = jm._interp(v10m, grids.wlat, grids.wlon, mid_lat, mid_lon_q, ti, tf, nt)
        swh = jm._interp(swhm, grids.slat, grids.slon, mid_lat, mid_lon_q, ti, tf, nt)
        ms = jm._interp(msm, grids.slat, grids.slon, mid_lat, mid_lon_q, ti, tf, nt)
        mc = jm._interp(mcm, grids.slat, grids.slon, mid_lat, mid_lon_q, ti, tf, nt)
        mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)
        tws = jnp.sqrt(u10 ** 2 + v10 ** 2)
        wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
        p = power_fn(tws, jnp.mod(wind_from - bearing, 360.0), swh,
                     jnp.mod(mwd - bearing, 360.0), v, wps)
        return jnp.stack([jnp.sum(p * seg) / 1000.0, jnp.max(swh), jnp.max(tws), jnp.max(p),
                          jnp.max(jnp.stack([swh / hs_lim, tws / tws_lim])) - 1.0,
                          jnp.max(p) / p_lim - 1.0])

    out = jax.jit(jax.vmap(one))(grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos)
    return dict(energy_mwh=np.asarray(out[:, 0]), max_hs=np.asarray(out[:, 1]),
                max_tws=np.asarray(out[:, 2]), max_power=np.asarray(out[:, 3]),
                margin=np.asarray(out[:, 4]), power_margin=np.asarray(out[:, 5]))


@partial(jax.jit, static_argnames=("power_fn", "wps"))
def _series_core(u10f, v10f, swhf, msf, mcf, wlat, wlon, slat, slon, steps,
                 mid_lat, mid_lon_q, bearing, v, smid, *, power_fn, wps):
    ti, tf = time_index(smid, steps)

    def one(u10m, v10m, swhm, msm, mcm):
        nt = u10m.shape[0]
        u10 = jm._interp(u10m, wlat, wlon, mid_lat, mid_lon_q, ti, tf, nt)
        v10 = jm._interp(v10m, wlat, wlon, mid_lat, mid_lon_q, ti, tf, nt)
        swh = jm._interp(swhm, slat, slon, mid_lat, mid_lon_q, ti, tf, nt)
        ms = jm._interp(msm, slat, slon, mid_lat, mid_lon_q, ti, tf, nt)
        mc = jm._interp(mcm, slat, slon, mid_lat, mid_lon_q, ti, tf, nt)
        mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)
        tws = jnp.sqrt(u10 ** 2 + v10 ** 2)
        wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
        p = power_fn(tws, jnp.mod(wind_from - bearing, 360.0), swh,
                     jnp.mod(mwd - bearing, 360.0), v, wps)
        return jnp.stack([p, swh, tws])

    return jax.vmap(one)(u10f, v10f, swhf, msf, mcf)


def member_series(grids, t_h, lat, lon, *, wps, power_fn, pad_to=512):
    """Per-member, per-segment weather and power along a timed track.

    ``t_h`` is hours since the ensemble's base time and ``lon`` is signed; the
    track is evaluated segment by segment as given, without resampling, so the
    result can be aggregated over any window (for example by voyage day).
    Several tracks can be evaluated in one call by concatenating them and
    discarding the joining segments. The segment count is padded to a multiple
    of ``pad_to`` so one compiled kernel serves tracks of similar length.

    Returns host arrays: ``power_kw``, ``swh``, ``tws`` of shape
    ``(member, segment)``; ``seg_h`` and ``t_mid_h`` per segment; and ``valid``,
    False where a segment lies outside the forecast's time range (those values
    are clamped and should be ignored).
    """
    t_h = np.asarray(t_h, np.float64)
    lat = np.asarray(lat, np.float64)
    glon = np.asarray(lon, np.float64)
    n = len(t_h) - 1
    npad = -(-n // pad_to) * pad_to
    seg = t_h[1:] - t_h[:-1]
    la0, lo0, la1, lo1 = (jnp.asarray(x, jnp.float32)
                          for x in (lat[:-1], glon[:-1], lat[1:], glon[1:]))
    dist = np.asarray(jm._haversine_m(la0, lo0, la1, lo1))
    bearing = np.asarray(jm._bearing_deg(la0, lo0, la1, lo1))
    v = dist / (np.maximum(seg, 1e-6) * 3600.0)
    mid_lat = (lat[:-1] + lat[1:]) / 2
    mid_lon = (glon[:-1] + glon[1:]) / 2
    mid_lon_q = np.where(mid_lon < 0, mid_lon + 360.0, mid_lon) if grids.lon_wrap else mid_lon
    smid = (t_h[:-1] + t_h[1:]) / 2

    def pad(x):
        return jnp.asarray(np.pad(x, (0, npad - n), mode="edge"), jnp.float32)

    out = np.asarray(_series_core(
        grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos,
        grids.wlat, grids.wlon, grids.slat, grids.slon, grids.steps,
        pad(mid_lat), pad(mid_lon_q), pad(bearing), pad(v), pad(smid),
        power_fn=power_fn, wps=wps))[:, :, :n]
    steps = np.asarray(grids.steps)
    return dict(power_kw=out[:, 0], swh=out[:, 1], tws=out[:, 2], seg_h=seg, t_mid_h=smid,
                valid=(smid >= steps[0]) & (smid <= steps[-1]))
