#!/usr/bin/env python
"""TiMBERS route parameterization and penalized cost (GPU/JAX).

A route is a degree-(K-1) Bezier curve with fixed endpoints at the ports;
CMA-ES searches the interior control points (in a normalized frame so sigma0
has a corridor-independent meaning) JOINTLY with an explicit time-allocation
profile: ``n_speed`` log-weights are interpolated to the L-1 segments and
exp-normalized into per-segment durations summing to the fixed passage time
(``time_alloc``). ``n_speed=0`` recovers the implicit uniform-speed model of
the BERS reference method (arXiv 2605.31533), of which this is an extension.
The batched penalized cost is evaluated for the whole population on the GPU.

Cost:  J = energy_MWh + lambda_env * P_env + lambda_land * P_land
  P_env  = sum_seg [exp(aH*[Hs-hs_soft]+) + exp(aU*[TWS-us_soft]+) - 2]
  P_land = sum_wp  land_fraction(waypoint)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import model as jm
from .model import DeviceGrids

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Corridor geometry (working-longitude frame, continuous across antimeridian)
# ---------------------------------------------------------------------------
@dataclass
class Corridor:
    name: str
    o_lat: float
    o_wlon: float
    d_lat: float
    d_wlon: float
    hours: float
    # normalization scale (same for both axes to preserve aspect ratio)
    scale: float = field(init=False)

    def __post_init__(self):
        self.scale = max(abs(self.d_lat - self.o_lat), abs(self.d_wlon - self.o_wlon))

    def norm(self, lat, wlon):
        return (
            (lat - self.o_lat) / self.scale,
            (wlon - self.o_wlon) / self.scale,
        )

    def denorm(self, nlat, nwlon):
        return (nlat * self.scale + self.o_lat, nwlon * self.scale + self.o_wlon)


def working_to_signed(wlon):
    return ((wlon + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Bezier (De Casteljau)
# ---------------------------------------------------------------------------
def bezier(ctrl, r):
    """De Casteljau eval. ctrl (K,2), r (L,) -> (L,2)."""
    K = ctrl.shape[0]
    c = jnp.broadcast_to(ctrl, (r.shape[0],) + ctrl.shape)  # (L,K,2)
    w = r[:, None, None]
    for _ in range(K - 1):
        c = c[:, :-1, :] * (1 - w) + c[:, 1:, :] * w
    return c[:, 0, :]  # (L,2)


# ---------------------------------------------------------------------------
# Land sampling (bilinear from raster)
# ---------------------------------------------------------------------------
class DeviceLand:
    def __init__(self, land: dict):
        self.lat = jnp.asarray(land["lat"], jnp.float32)
        self.wlon = jnp.asarray(land["wlon"], jnp.float32)
        self.mask = jnp.asarray(land["mask"], jnp.float32)  # (Y,X), 1=land

    def sample(self, lat, wlon):
        yi, yf = jm._frac(self.lat, lat)
        xi, xf = jm._frac(self.wlon, wlon)
        m = self.mask
        c0 = m[yi, xi] * (1 - xf) + m[yi, xi + 1] * xf
        c1 = m[yi + 1, xi] * (1 - xf) + m[yi + 1, xi + 1] * xf
        return c0 * (1 - yf) + c1 * yf


# ---------------------------------------------------------------------------
# Penalized cost
# ---------------------------------------------------------------------------
@dataclass
class Penalty:
    # Soft env penalty: exp(aH*[swh-hs_soft]+) + exp(aU*[tws-us_soft]+) - 2 per
    # segment. hs_soft/us_soft sit *below* the application's hard feasibility
    # limits to keep a safety margin, since minimizing energy rewards tall
    # *following* seas and must be actively counteracted.
    lambda_env: float = 30.0
    lambda_land: float = 100.0
    aH: float = 6.0
    aU: float = 2.0
    hs_soft: float = 6.5
    us_soft: float = 19.0


def _route_cost(fields, axes, land_arrs, cor, dt_h, nt, lon_wrap,
                nlat, nwlon, seg_dt, dep_off, L, wps, pen: Penalty, power_fn,
                align_dt_h=0.0):
    """Penalized cost for one route given normalized waypoint coords (L,).

    ``seg_dt`` is the (L-1,) array of per-segment durations (hours); uniform for
    the implicit-speed model, or a learned allocation for the explicit-speed
    model (always summing to the fixed passage time).

    ``align_dt_h`` > 0 makes the cost match the scorer: the L-point track is
    linearly resampled to a uniform time grid of that step (e.g. 0.25 h) and the
    energy/Hs are integrated there. This optimizes the scored quantity and lets
    the Hs penalty see the sub-segment peaks the scorer sees (so we can ride Hs
    up to 7 safely). With 0, the cost integrates on the native L segments.

    ``fields`` and ``axes`` are passed as traced arguments (NOT closures) so XLA
    treats the large grid arrays as parameters rather than baking them into the
    HLO as host-resident constants.
    """
    u10f, v10f, swhf, msf, mcf = fields
    wlat, wlon_ax, slat, slon = axes
    lmask, llat, lwlon = land_arrs

    lat, wlon = cor.denorm(nlat, nwlon)
    if align_dt_h > 0.0:
        # Replicate the scorer: linearly resample the track to uniform time.
        t_cum = jnp.concatenate([jnp.zeros(1, lat.dtype), jnp.cumsum(seg_dt)])
        M = int(round(cor.hours / align_dt_h))
        tau = jnp.linspace(0.0, cor.hours, M + 1).astype(lat.dtype)
        lat = jnp.interp(tau, t_cum, lat)
        wlon = jnp.interp(tau, t_cum, wlon)
        seg_dt = jnp.full((M,), cor.hours / M, lat.dtype)
    glon = working_to_signed(wlon)  # signed lon for weather

    glon_w = jnp.where(glon < 0, glon + 360.0, glon) if lon_wrap else glon

    mid_lat = (lat[:-1] + lat[1:]) / 2
    mid_lon = (glon_w[:-1] + glon_w[1:]) / 2
    cum = jnp.cumsum(seg_dt)
    seg_mid_h = dep_off + cum - seg_dt / 2
    ti, tf = jm._time_index(seg_mid_h, dt_h, nt)

    u10 = jm._interp(u10f, wlat, wlon_ax, mid_lat, mid_lon, ti, tf, nt)
    v10 = jm._interp(v10f, wlat, wlon_ax, mid_lat, mid_lon, ti, tf, nt)
    swh = jm._interp(swhf, slat, slon, mid_lat, mid_lon, ti, tf, nt)
    ms = jm._interp(msf, slat, slon, mid_lat, mid_lon, ti, tf, nt)
    mc = jm._interp(mcf, slat, slon, mid_lat, mid_lon, ti, tf, nt)
    mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)

    dist = jm._haversine_m(lat[:-1], glon[:-1], lat[1:], glon[1:])
    v = dist / (seg_dt * 3600.0)
    bearing = jm._bearing_deg(lat[:-1], glon[:-1], lat[1:], glon[1:])

    tws = jnp.sqrt(u10**2 + v10**2)
    wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
    twa = jnp.mod(wind_from - bearing, 360.0)
    mwa = jnp.mod(mwd - bearing, 360.0)

    p = power_fn(tws, twa, swh, mwa, v, wps)
    energy = jnp.sum(p * seg_dt) / 1000.0

    over_h = jnp.maximum(swh - pen.hs_soft, 0.0)
    over_u = jnp.maximum(tws - pen.us_soft, 0.0)
    p_env = jnp.sum(jnp.exp(pen.aH * over_h) + jnp.exp(pen.aU * over_u) - 2.0)

    p_land = jnp.sum(_sample_mask(lmask, llat, lwlon, lat, wlon))

    cost = energy + pen.lambda_env * p_env + pen.lambda_land * p_land
    return cost, energy, p_env, p_land


def _sample_mask(mask, lat_ax, wlon_ax, lat, wlon):
    yi, yf = jm._frac(lat_ax, lat)
    xi, xf = jm._frac(wlon_ax, wlon)
    c0 = mask[yi, xi] * (1 - xf) + mask[yi, xi + 1] * xf
    c1 = mask[yi + 1, xi] * (1 - xf) + mask[yi + 1, xi + 1] * xf
    return c0 * (1 - yf) + c1 * yf


# ---------------------------------------------------------------------------
# Time allocation (explicit speed) -- distributes the fixed passage time over
# the L-1 segments. n_speed=0 recovers the implicit uniform-Delta-t model.
# ---------------------------------------------------------------------------
def time_alloc(speed_params, hours, L, n_speed):
    """(n_speed,) log-weights -> (L-1,) segment durations summing to ``hours``.

    Log-weights are interpolated to segment centres and exp-normalized, so the
    profile is smooth and strictly positive. ``speed_params == 0`` (the init)
    gives uniform durations == the implicit model.
    """
    n = L - 1
    if n_speed == 0:
        return jnp.full((n,), hours / n, jnp.float32)
    seg_r = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n
    ctrl_r = jnp.linspace(0.0, 1.0, n_speed, dtype=jnp.float32)
    lw = jnp.interp(seg_r, ctrl_r, speed_params)
    w = jnp.exp(lw)
    return hours * w / jnp.sum(w)


# ---------------------------------------------------------------------------
# Batched penalized cost
# ---------------------------------------------------------------------------
def make_batched_cost(grids, land, cor, L, wps, pen, K, power_fn, n_speed=0,
                      align_dt_h=0.0):
    """Return fn(theta_batch, dep_off) -> cost_batch (P,) on device.

    ``theta`` packs the 2*(K-2) interior Bezier coords followed by ``n_speed``
    time-allocation log-weights (explicit speed; n_speed=0 = implicit).
    ``power_fn`` is the injected ``(tws, twa, swh, mwa, v, wps) -> kW`` model.
    ``align_dt_h`` > 0 integrates the cost on a uniform time grid (scorer-aligned).
    ``dep_off`` is a *traced* argument, so the kernel compiles once per
    (corridor, wps) and is reused across all departures.
    """
    o_n = jnp.asarray(cor.norm(cor.o_lat, cor.o_wlon), jnp.float32)
    d_n = jnp.asarray(cor.norm(cor.d_lat, cor.d_wlon), jnp.float32)
    r = jnp.linspace(0.0, 1.0, L, dtype=jnp.float32)
    fields = (grids.u10, grids.v10, grids.swh, grids.mwd_sin, grids.mwd_cos)
    axes = (grids.wlat, grids.wlon, grids.slat, grids.slon)
    land_arrs = (land.mask, land.lat, land.wlon)
    dt_h, nt, lon_wrap = grids.dt_h, grids.nt, grids.lon_wrap
    n_geo = 2 * (K - 2)

    def one(theta, dep_off, fields, axes, land_arrs):
        interior = theta[:n_geo].reshape(K - 2, 2)
        ctrl = jnp.concatenate([o_n[None, :], interior, d_n[None, :]], axis=0)
        pts = bezier(ctrl, r)  # (L,2) normalized (nlat, nwlon)
        seg_dt = time_alloc(theta[n_geo:], cor.hours, L, n_speed)
        cost, *_ = _route_cost(fields, axes, land_arrs, cor, dt_h, nt, lon_wrap,
                               pts[:, 0], pts[:, 1], seg_dt, dep_off, L, wps, pen,
                               power_fn, align_dt_h)
        return cost

    batched = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None)))
    return lambda tb, dep_off: batched(
        tb, jnp.float32(dep_off), fields, axes, land_arrs
    )


def decode_route(theta, cor, K, L, n_speed=0):
    """theta -> (lats, wlons, seg_dt) in geographic working coords."""
    o_n = np.array(cor.norm(cor.o_lat, cor.o_wlon))
    d_n = np.array(cor.norm(cor.d_lat, cor.d_wlon))
    n_geo = 2 * (K - 2)
    interior = theta[:n_geo].reshape(K - 2, 2)
    ctrl = jnp.asarray(np.vstack([o_n, interior, d_n]), jnp.float32)
    r = jnp.linspace(0.0, 1.0, L, dtype=jnp.float32)
    pts = np.array(bezier(ctrl, r))
    lat, wlon = cor.denorm(pts[:, 0], pts[:, 1])
    seg_dt = np.array(time_alloc(
        jnp.asarray(theta[n_geo:], jnp.float32), cor.hours, L, n_speed))
    return lat, wlon, seg_dt


def gc_init_theta(cor, K, n_speed=0):
    """Interior control points along the straight chord + flat speed profile."""
    o_n = np.array(cor.norm(cor.o_lat, cor.o_wlon))
    d_n = np.array(cor.norm(cor.d_lat, cor.d_wlon))
    fr = np.linspace(0, 1, K)[1:-1]
    interior = o_n[None, :] + fr[:, None] * (d_n - o_n)[None, :]
    return np.concatenate([interior.ravel(), np.zeros(n_speed)])
