#!/usr/bin/env python
"""Weather grids on the device and the route-sampling core.

Every cost and scorer on the device path is a reduction over :func:`sample`:
the track is cut into segments, the weather is interpolated trilinearly at each
segment's midpoint and mid-time for every ensemble member (:func:`weather`), and
the injected power model gives the shaft power there (:func:`power`). The two
steps are separate so that the weather is interpolated once when the power model
is evaluated for several draws of its parameters. The optimizer's cost
(:mod:`timbers.optimizer`) reads one member; the ensemble objectives and
scorers (:mod:`timbers.ensemble`) reduce over all of them. The physics mirrors
the host scorer, ``timbers.scoring.evaluate_route_full``.

The power model is not part of this library; pass your own ``power_fn`` with the
signature ``power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`` (operating on
JAX arrays for this device path). For power-model uncertainty it also takes
one draw of its parameters, ``power_fn(..., wps, params)``; see
:mod:`timbers.ensemble`. See ``examples/toy_power.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .geo import bearing_deg, haversine_m, midpoint_lon, to_grid_lon
from .weather import utc_datetime64

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
class Grids:
    """Wind and wave fields on the device, float32, with a member axis.

    Every field is ``(member, time, lat, lon)``. ``steps`` gives the time of each
    slice in hours after ``t0``, and need not be uniform (ECMWF ENS is 3-hourly
    to 144 h, then 6-hourly). A single field such as ERA5 is a one-member grid;
    build it with :meth:`from_era5`.

    ``wind`` holds ``u10``, ``v10``, ``lat``, ``lon``; ``wave`` holds ``swh``,
    ``mwd`` (degrees), ``lat``, ``lon``. Wind and wave may be on different
    spatial grids but share ``steps``. ``t0`` (``datetime64``) is optional and
    only used by :meth:`hours_after_t0`.

    ``consume=True`` removes the field arrays from ``wind`` and ``wave`` as they
    are moved to the device, which roughly halves peak host memory for a large
    ensemble; it is off by default because it empties the caller's dicts.

    Fields must be finite. Land-masked wave cells are typically NaN in decoded
    GRIB; fill them first, for example with
    :func:`timbers.weather.fill_from_nearest_sea`. A NaN objective would not
    raise inside the optimizer: the candidate would be silently ranked out.
    """

    def __init__(self, wind: dict, wave: dict, steps, *, t0=None, consume: bool = False):
        take = (lambda d, k: d.pop(k)) if consume else (lambda d, k: d[k])
        self.t0 = None if t0 is None else np.datetime64(t0, "s")
        self.steps = jnp.asarray(np.asarray(steps, np.float32))
        self.wind_lat = jnp.asarray(wind["lat"], jnp.float32)
        self.wind_lon = jnp.asarray(wind["lon"], jnp.float32)
        self.wave_lat = jnp.asarray(wave["lat"], jnp.float32)
        self.wave_lon = jnp.asarray(wave["lon"], jnp.float32)
        self.u10 = jnp.asarray(take(wind, "u10"), jnp.float32)
        self.v10 = jnp.asarray(take(wind, "v10"), jnp.float32)
        self.swh = jnp.asarray(take(wave, "swh"), jnp.float32)
        # Direction as sine and cosine so interpolation does not wrap at 360.
        # Computed in float32 one array at a time to bound host memory.
        mwd = np.radians(np.asarray(take(wave, "mwd")).astype(np.float32, copy=False))
        self.mwd_sin = jnp.asarray(np.sin(mwd, dtype=np.float32))
        self.mwd_cos = jnp.asarray(np.cos(mwd, dtype=np.float32))
        del mwd
        for name, arr in zip(("u10", "v10", "swh", "mwd_sin", "mwd_cos"), self.fields):
            if arr.ndim != 4:
                raise ValueError(f"{name} must be (member, time, lat, lon), got {arr.shape}")
            if not bool(jnp.all(jnp.isfinite(arr))):
                n = int(jnp.sum(~jnp.isfinite(arr)))
                raise ValueError(
                    f"{name} has {n:,} non-finite values; fill land-masked "
                    "cells before use (timbers.weather.fill_from_nearest_sea)"
                )
        self.n_members, self.nt = self.u10.shape[:2]
        if self.steps.shape[0] != self.nt:
            raise ValueError(f"{self.steps.shape[0]} steps for {self.nt} time slices")
        lon = np.asarray(wind["lon"])
        self.lon_wrap = bool(lon[0] >= 0 and lon[-1] > 180)

    @classmethod
    def from_era5(cls, wind: dict, wave: dict, start=None, hours: float | None = None):
        """A single gridded field (e.g. from ``load_era5``) as a one-member grid.

        Without ``start``, the whole grid is used and ``t0`` is the grid's first
        time; both grids must start then. With ``start`` (a datetime; tz-aware
        is converted to UTC) and ``hours``, the grid is cut to the window
        ``[start, start + hours]`` and ``t0 = start``; ``start`` must fall on a
        time step of both grids and both must cover the window. Anything else
        raises ``ValueError``.
        """
        dt = float(wind["dt_h"])
        if float(wave["dt_h"]) != dt:
            raise ValueError("wind and wave grids must share a time step")
        if (start is None) != (hours is None):
            raise ValueError("pass start and hours together, or neither")
        if start is None:
            if np.datetime64(wave["t0"], "s") != np.datetime64(wind["t0"], "s"):
                raise ValueError("wind and wave grids must start at the same time")
            t0, i0, n = np.datetime64(wind["t0"], "s"), 0, None
        else:
            t0 = utc_datetime64(start)
            n = int(hours / dt) + 3
        out = {}
        for name, g, keys in (("wind", wind, ("u10", "v10")), ("wave", wave, ("swh", "mwd"))):
            if start is not None:
                off = float((t0 - g["t0"]) / np.timedelta64(1, "h")) / dt
                i0 = int(round(off))
                if abs(off - i0) > 1e-6:
                    raise ValueError(f"{start} is not on a time step of the {name} grid")
                if i0 < 0:
                    raise ValueError(f"{name} grid starts after {start}")
            stop = None if n is None else i0 + n
            out[name] = {k: np.asarray(g[k][i0:stop])[None] for k in keys}
            if start is not None and out[name][keys[0]].shape[1] < math.ceil(hours / dt) + 1:
                raise ValueError(f"{name} grid ends before {start} + {hours} h")
            out[name]["lat"], out[name]["lon"] = g["lat"], g["lon"]
        nt = min(out["wind"]["u10"].shape[1], out["wave"]["swh"].shape[1])
        for side, keys in (("wind", ("u10", "v10")), ("wave", ("swh", "mwd"))):
            for k in keys:
                out[side][k] = out[side][k][:, :nt]
        return cls(out["wind"], out["wave"], np.arange(nt) * dt, t0=t0)

    @property
    def fields(self):
        return (self.u10, self.v10, self.swh, self.mwd_sin, self.mwd_cos)

    @property
    def axes(self):
        return (self.wind_lat, self.wind_lon, self.wave_lat, self.wave_lon, self.steps)

    def hours_after_t0(self, when) -> float:
        """Hours from ``t0`` to ``when``, e.g. a departure offset.

        ``when`` is a datetime (naive is taken as UTC; tz-aware is converted) or
        a ``datetime64``.
        """
        if self.t0 is None:
            raise ValueError("these grids have no t0")
        return float((utc_datetime64(when) - self.t0) / np.timedelta64(1, "h"))


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def _frac(coord, values):
    n = coord.shape[0]
    step = coord[1] - coord[0]
    fi = jnp.clip((values - coord[0]) / step, 0.0, n - 1.0)
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, n - 2)
    return i0, fi - i0


def time_index(hours, steps):
    """Bracketing index and fraction of ``hours`` on an ascending step vector."""
    nt = steps.shape[0]
    ti = jnp.clip(jnp.searchsorted(steps, hours, side="right") - 1, 0, nt - 2)
    t0, t1 = steps[ti], steps[ti + 1]
    return ti, jnp.clip((hours - t0) / jnp.maximum(t1 - t0, 1e-6), 0.0, 1.0)


def _interp(field, lat_ax, lon_ax, lat, lon, ti, tf):
    """Trilinear interpolation of a (time, lat, lon) field."""
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


# ---------------------------------------------------------------------------
# Tracks and segments
# ---------------------------------------------------------------------------
# Snapping the number of integration points to a geometric ladder bounds the
# number of distinct compiled kernels when the same cost is built for many
# passages of different length (as in re-planning, where the remaining time
# shrinks every cycle), so the compilation cache is reused instead of growing.
# The step lands within sqrt(1.5) of the requested ``align``.
_M_LADDER = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536)


def n_points(hours: float, align: float, quantise: bool = False) -> int:
    """Number of uniform-time segments of about ``align`` hours in ``hours``.

    ``quantise=True`` snaps it to a fixed ladder of sizes (see ``_M_LADDER``).
    """
    if not quantise:
        return max(1, int(round(hours / align)))
    m = hours / align
    return min(_M_LADDER, key=lambda c: abs(math.log(c / m)))


def resample(lat, wlon, seg_dt, hours: float, m: int):
    """A track resampled to ``m`` segments of equal duration.

    ``lat``, ``wlon`` (continuous working longitude) and ``seg_dt`` (hours) as
    from :func:`timbers.optimizer.decode_route`. Returns ``(lat, wlon, seg_dt)``
    with ``m + 1`` points, so the cost integrates where the scorer does.
    """
    t_cum = jnp.concatenate([jnp.zeros(1, lat.dtype), jnp.cumsum(seg_dt)])
    tau = jnp.linspace(0.0, hours, m + 1).astype(lat.dtype)
    return (
        jnp.interp(tau, t_cum, lat),
        jnp.interp(tau, t_cum, wlon),
        jnp.full((m,), hours / m, lat.dtype),
    )


class Segments(NamedTuple):
    """Per-segment quantities of a timed track, where the weather is sampled."""

    v: object  # speed over ground, m/s
    bearing: object  # degrees
    mid_lat: object
    mid_lon: object  # signed
    t_mid: object  # hours after the grids' t0
    seg_h: object  # duration, hours


def segments(lat, lon, seg_dt, dep_off, xp=jnp) -> Segments:
    """Cut a track into segments. ``lon`` may be signed or continuous working
    longitude; ``dep_off`` is the departure in hours after the grids' ``t0``."""
    glon = to_grid_lon(lon, False)
    dist = haversine_m(lat[:-1], glon[:-1], lat[1:], glon[1:], xp=xp)
    return Segments(
        v=dist / (seg_dt * 3600.0),
        bearing=bearing_deg(lat[:-1], glon[:-1], lat[1:], glon[1:], xp=xp),
        mid_lat=(lat[:-1] + lat[1:]) / 2,
        mid_lon=midpoint_lon(glon[:-1], glon[1:], False),
        t_mid=dep_off + xp.cumsum(seg_dt) - seg_dt / 2,
        seg_h=seg_dt,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
class Weather(NamedTuple):
    """The power model's weather inputs, each ``(member, segment)``."""

    tws: object  # true wind speed, m/s
    twa: object  # true wind angle off the bow, degrees
    swh: object  # significant wave height, m
    mwa: object  # mean wave angle off the bow, degrees


def weather(fields, axes, segs: Segments, lon_wrap: bool, perturbation=None) -> Weather:
    """The weather each segment meets, for every member, relative to its heading.

    ``fields`` and ``axes`` are ``Grids.fields`` and ``Grids.axes``, passed
    explicitly so that a jitted caller can take them as traced arguments rather
    than closing over them (which would bake multi-GB arrays into the compiled
    program as constants).

    ``perturbation`` is an optional ``(dlat, dlon, dt_h, hs_scale, wind_scale)``
    applied to where and when the weather is read, and to its amplitude; the
    route's own geometry and speed are unchanged. It is the forecast-error
    surrogate of :func:`timbers.ensemble.perturbation_grid`.
    """
    wind_lat, wind_lon, wave_lat, wave_lon, steps = axes
    qlat, qlon, qt = segs.mid_lat, segs.mid_lon, segs.t_mid
    hs_scale = wind_scale = 1.0
    if perturbation is not None:
        dlat, dlon, dt, hs_scale, wind_scale = (perturbation[i] for i in range(5))
        qlat, qlon, qt = qlat + dlat, qlon + dlon, qt + dt
    qlon = to_grid_lon(qlon, lon_wrap)
    ti, tf = time_index(qt, steps)

    def one(u10m, v10m, swhm, msm, mcm):
        u10 = _interp(u10m, wind_lat, wind_lon, qlat, qlon, ti, tf) * wind_scale
        v10 = _interp(v10m, wind_lat, wind_lon, qlat, qlon, ti, tf) * wind_scale
        swh = _interp(swhm, wave_lat, wave_lon, qlat, qlon, ti, tf) * hs_scale
        ms = _interp(msm, wave_lat, wave_lon, qlat, qlon, ti, tf)
        mc = _interp(mcm, wave_lat, wave_lon, qlat, qlon, ti, tf)
        mwd = jnp.mod(jnp.degrees(jnp.arctan2(ms, mc)), 360.0)
        tws = jnp.sqrt(u10**2 + v10**2)
        wind_from = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
        twa = jnp.mod(wind_from - segs.bearing, 360.0)
        mwa = jnp.mod(mwd - segs.bearing, 360.0)
        return Weather(tws, twa, swh, mwa)

    return jax.vmap(one)(*fields)


def power(power_fn, w: Weather, v, wps, params=None):
    """Shaft power (kW) per member and segment for the weather ``w``.

    ``power_fn`` sees one member at a time, ``(segment,)`` arrays, as on the
    host. With ``params`` (one draw of the power model's parameters, a pytree)
    it is called as ``power_fn(tws, twa, swh, mwa, v, wps, params)``.
    """
    extra = () if params is None else (params,)
    return jax.vmap(lambda tws, twa, swh, mwa: power_fn(tws, twa, swh, mwa, v, wps, *extra))(*w)


def sample(fields, axes, segs: Segments, power_fn, wps, lon_wrap: bool, perturbation=None):
    """Shaft power (kW), Hs (m) and TWS (m/s) per member and segment.

    :func:`weather` followed by :func:`power` with the nominal power model.
    Returns three ``(member, segment)`` arrays.
    """
    w = weather(fields, axes, segs, lon_wrap, perturbation)
    return power(power_fn, w, segs.v, wps), w.swh, w.tws


def route_energy(grids: Grids, lat, lon, seg_dt, dep_off, wps: bool, power_fn):
    """Energy (MWh) of a timed track for each member, on its native segments."""
    segs = segments(lat, lon, seg_dt, dep_off)
    p, _, _ = sample(grids.fields, grids.axes, segs, power_fn, wps, grids.lon_wrap)
    return jnp.sum(p * seg_dt, axis=-1) / 1000.0
