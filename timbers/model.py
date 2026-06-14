#!/usr/bin/env python
"""GPU/JAX route-energy evaluator for the optimizer.

Evaluates the scorer's physics on-device so CMA-ES can evaluate large batches
of candidate routes cheaply and the polish stage can take gradients. The energy
path mirrors ``timbers.scoring.evaluate_route_full``: weather is interpolated
trilinearly at segment midpoints, power comes from an injected power model
(``power_fn``), and energy is ``sum(P * dt)``.

The power model is not part of this library; pass your own ``power_fn`` with the
signature ``power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`` (operating on
JAX arrays for this device path). See ``examples/toy_power.py``.

Grids are pushed to the device as float32 via :class:`DeviceGrids`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Grid container
# ---------------------------------------------------------------------------
class DeviceGrids:
    """Wind+wave corridor grids resident on the JAX device (float32)."""

    def __init__(self, wind: dict, wave: dict):
        self.t0_wind = wind["t0"]
        self.dt_h = float(wind["dt_h"])
        # axes
        self.wlat = jnp.asarray(wind["lat"], jnp.float32)
        self.wlon = jnp.asarray(wind["lon"], jnp.float32)
        self.slat = jnp.asarray(wave["lat"], jnp.float32)
        self.slon = jnp.asarray(wave["lon"], jnp.float32)
        # fields (T, Y, X)
        self.u10 = jnp.asarray(wind["u10"], jnp.float32)
        self.v10 = jnp.asarray(wind["v10"], jnp.float32)
        self.swh = jnp.asarray(wave["swh"], jnp.float32)
        mwd_rad = np.radians(wave["mwd"].astype(np.float64))
        self.mwd_sin = jnp.asarray(np.sin(mwd_rad), jnp.float32)
        self.mwd_cos = jnp.asarray(np.cos(mwd_rad), jnp.float32)
        self.nt = self.u10.shape[0]
        # wave/wind share the same time base (built together by repackage)
        self.lon_wrap = bool(wind["lon"][0] >= 0 and wind["lon"][-1] > 180)


# ---------------------------------------------------------------------------
# Trilinear interpolation (lat, lon, time) on a regular grid
# ---------------------------------------------------------------------------
def _frac(coord, values):
    n = coord.shape[0]
    step = coord[1] - coord[0]
    fi = jnp.clip((values - coord[0]) / step, 0.0, n - 1.0)
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, n - 2)
    return i0, fi - i0


def _interp(field, lat_ax, lon_ax, lat, lon, ti, tf, nt):
    yi, yf = _frac(lat_ax, lat)
    xi, xf = _frac(lon_ax, lon)

    def g(dt, dy, dx):
        return field[ti + dt, yi + dy, xi + dx]

    c00 = g(0, 0, 0) * (1 - xf) + g(0, 0, 1) * xf
    c01 = g(0, 1, 0) * (1 - xf) + g(0, 1, 1) * xf
    c10 = g(1, 0, 0) * (1 - xf) + g(1, 0, 1) * xf
    c11 = g(1, 1, 0) * (1 - xf) + g(1, 1, 1) * xf
    c0 = c00 * (1 - yf) + c01 * yf
    c1 = c10 * (1 - yf) + c11 * yf
    return c0 * (1 - tf) + c1 * tf


def _time_index(hours, dt_h, nt):
    tr = jnp.clip(hours / dt_h, 0.0, nt - 1.0)
    ti = jnp.clip(jnp.floor(tr).astype(jnp.int32), 0, nt - 2)
    return ti, tr - ti


# ---------------------------------------------------------------------------
# Route energy
# ---------------------------------------------------------------------------
_R_EARTH = 6_371_000.0


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1r, lat2r = jnp.radians(lat1), jnp.radians(lat2)
    dlat = lat2r - lat1r
    dlon = jnp.radians(lon2 - lon1)
    a = jnp.sin(dlat / 2) ** 2 + jnp.cos(lat1r) * jnp.cos(lat2r) * jnp.sin(dlon / 2) ** 2
    return _R_EARTH * 2 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    lat1r, lat2r = jnp.radians(lat1), jnp.radians(lat2)
    dlon = jnp.radians(lon2 - lon1)
    x = jnp.sin(dlon) * jnp.cos(lat2r)
    y = jnp.cos(lat1r) * jnp.sin(lat2r) - jnp.sin(lat1r) * jnp.cos(lat2r) * jnp.cos(dlon)
    return jnp.mod(jnp.degrees(jnp.arctan2(x, y)), 360.0)


def route_energy(grids: DeviceGrids, lats, lons, seg_dt_h, dep_offset_h, wps: bool,
                 power_fn):
    """Total energy (MWh) for a single polyline route.

    Parameters
    ----------
    lats, lons : (L,) device arrays of waypoint coordinates (lon in deg, signed).
    seg_dt_h   : (L-1,) hours per segment.
    dep_offset_h : float, hours from grid t0 to departure.
    power_fn : callable ``(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`` on JAX arrays.
    """
    lons = jnp.where(lons < 0, lons + 360.0, lons) if grids.lon_wrap else lons
    mid_lat = (lats[:-1] + lats[1:]) / 2
    mid_lon = (lons[:-1] + lons[1:]) / 2
    cum = jnp.cumsum(seg_dt_h)
    seg_mid_h = dep_offset_h + cum - seg_dt_h / 2

    ti, tf = _time_index(seg_mid_h, grids.dt_h, grids.nt)
    u10 = _interp(grids.u10, grids.wlat, grids.wlon, mid_lat, mid_lon, ti, tf, grids.nt)
    v10 = _interp(grids.v10, grids.wlat, grids.wlon, mid_lat, mid_lon, ti, tf, grids.nt)
    swh = _interp(grids.swh, grids.slat, grids.slon, mid_lat, mid_lon, ti, tf, grids.nt)
    ms = _interp(grids.mwd_sin, grids.slat, grids.slon, mid_lat, mid_lon, ti, tf, grids.nt)
    mc = _interp(grids.mwd_cos, grids.slat, grids.slon, mid_lat, mid_lon, ti, tf, grids.nt)
    mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)

    dist = _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    v = dist / (seg_dt_h * 3600.0)
    bearing = _bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])

    tws = jnp.sqrt(u10**2 + v10**2)
    wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
    twa = jnp.mod(wind_from - bearing, 360.0)
    mwa = jnp.mod(mwd - bearing, 360.0)

    p = power_fn(tws, twa, swh, mwa, v, wps)
    return jnp.sum(p * seg_dt_h) / 1000.0
