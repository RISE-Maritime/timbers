#!/usr/bin/env python
"""Route energy scorer (NumPy reference).

Evaluates total energy (MWh) for a single route using ERA5 data and an injected
power model. ``evaluate_route_full`` additionally returns the route diagnostics
(max wind, max Hs, sailed distance) used for feasibility checks.

This NumPy path is the host-side reference the GPU/JAX evaluator
(``timbers.model``) mirrors. The power model is not part of this library; pass
your own ``power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`` operating on
NumPy arrays. See ``examples/toy_power.py``.

Usage
-----
::

    from timbers.era5 import load_era5
    from timbers.scoring import evaluate_route

    wind_grid = load_era5(["wind.nc"])
    wave_grid = load_era5(["waves.nc"])
    energy_mwh = evaluate_route(
        wind_grid, wave_grid,
        waypoints=[(datetime(...), lat, lon), ...],
        wps=True, power_fn=my_power,
    )
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from .era5 import query, query_angle

__all__ = ["evaluate_route", "evaluate_route_full"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between arrays of (lat, lon) pairs."""
    R = 6_371_000.0
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _forward_bearing_deg(lat1, lon1, lat2, lon2):
    """Forward bearing in degrees from point 1 to point 2."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.mod(np.degrees(np.arctan2(x, y)), 360.0)


# ---------------------------------------------------------------------------
# Route evaluation
# ---------------------------------------------------------------------------
def evaluate_route(
    wind_grid: dict,
    wave_grid: dict,
    waypoints: list[tuple[datetime, float, float]],
    power_fn,
    wps: bool = False,
    resample_dt_min: float = 15.0,
) -> float:
    """Evaluate total energy (MWh) for a route.

    Parameters
    ----------
    wind_grid, wave_grid : dict
        Grids from :func:`timbers.era5.load_era5`.
    waypoints : list of (datetime, lat_deg, lon_deg)
        Route waypoints in chronological order.
    power_fn : callable ``(tws, twa_deg, swh, mwa_deg, v, wps) -> kW`` on NumPy arrays.
    wps : bool
        Whether wingsails are deployed.
    resample_dt_min : float
        Resample interval in minutes (for integration accuracy).

    Returns
    -------
    float
        Total energy in MWh.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")

    # Resample to uniform Δt for integration-accuracy independence
    waypoints = _resample(waypoints, resample_dt_min)

    lats = np.array([wp[1] for wp in waypoints])
    lons = np.array([wp[2] for wp in waypoints])

    # Segment dt in hours
    wp_times = np.array(
        [np.datetime64(wp[0]) for wp in waypoints], dtype="datetime64[s]"
    )
    seg_dt_h = ((wp_times[1:] - wp_times[:-1]) / np.timedelta64(1, "h")).astype(
        np.float64
    )
    seg_dt_h = np.maximum(seg_dt_h, 1e-6)

    # Normalize longitudes for ERA5 grid
    grid_lon = wind_grid["lon"]
    if grid_lon[0] >= 0 and grid_lon[-1] > 180:
        lons = np.where(lons < 0, lons + 360, lons)

    # Segment midpoints
    mid_lat = (lats[:-1] + lats[1:]) / 2
    mid_lon = (lons[:-1] + lons[1:]) / 2

    # Time at midpoints (hours since grid t0)
    dep_dt64 = wp_times[0]
    dep_offset_h = float((dep_dt64 - wind_grid["t0"]) / np.timedelta64(1, "h"))
    cum_h = np.cumsum(seg_dt_h)
    seg_mid_h = dep_offset_h + cum_h - seg_dt_h / 2

    # Interpolate weather at midpoints
    u10 = query(wind_grid, "u10", mid_lat, mid_lon, seg_mid_h)
    v10 = query(wind_grid, "v10", mid_lat, mid_lon, seg_mid_h)
    swh = query(wave_grid, "swh", mid_lat, mid_lon, seg_mid_h)
    mwd = query_angle(wave_grid, "mwd", mid_lat, mid_lon, seg_mid_h)

    # Ship speed (m/s)
    seg_dist_m = _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    v_mps = seg_dist_m / (seg_dt_h * 3600.0)

    # Heading (degrees)
    bearing_deg = _forward_bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])

    # TWS and TWA relative to heading
    tws = np.sqrt(u10**2 + v10**2)
    wind_from_deg = np.mod(180.0 + np.degrees(np.arctan2(u10, v10)), 360.0)
    twa_deg = np.mod(wind_from_deg - bearing_deg, 360.0)

    # MWA relative to heading
    mwa_deg = np.mod(mwd - bearing_deg, 360.0)

    # power (kW) at each segment midpoint
    power_kw = power_fn(tws, twa_deg, swh, mwa_deg, v_mps, wps)

    # Energy: sum(P * dt) / 1000
    energy_mwh = float(np.sum(power_kw * seg_dt_h) / 1000.0)
    return energy_mwh


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def _resample(
    waypoints: list[tuple[datetime, float, float]], dt_min: float
) -> list[tuple[datetime, float, float]]:
    """Resample waypoints to approximately uniform time intervals.

    Inserts intermediate points via geodesic (great-circle) interpolation
    on segments longer than ``dt_min`` minutes.
    """
    result = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        t0, lat0, lon0 = waypoints[i]
        t1, lat1, lon1 = waypoints[i + 1]
        seg_min = (t1 - t0).total_seconds() / 60.0
        n_sub = max(1, int(math.ceil(seg_min / dt_min)))
        for k in range(1, n_sub + 1):
            f = k / n_sub
            t = t0 + (t1 - t0) * f
            lat = lat0 + f * (lat1 - lat0)
            # Handle longitude wrapping
            dlon = ((lon1 - lon0 + 180.0) % 360.0) - 180.0
            lon = lon0 + f * dlon
            result.append((t, lat, lon))
    return result


def evaluate_route_full(
    wind_grid: dict,
    wave_grid: dict,
    waypoints: list[tuple[datetime, float, float]],
    power_fn,
    wps: bool = False,
    resample_dt_min: float = 15.0,
) -> dict:
    """Evaluate a route, returning energy and constraint diagnostics.

    Returns a dict with keys: ``energy_mwh``, ``max_wind_mps``, ``max_hs_m``,
    ``sailed_distance_nm``. The energy computation is byte-for-byte the same as
    :func:`evaluate_route`.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")

    waypoints = _resample(waypoints, resample_dt_min)

    lats = np.array([wp[1] for wp in waypoints])
    lons = np.array([wp[2] for wp in waypoints])

    wp_times = np.array(
        [np.datetime64(wp[0]) for wp in waypoints], dtype="datetime64[s]"
    )
    seg_dt_h = ((wp_times[1:] - wp_times[:-1]) / np.timedelta64(1, "h")).astype(
        np.float64
    )
    seg_dt_h = np.maximum(seg_dt_h, 1e-6)

    grid_lon = wind_grid["lon"]
    if grid_lon[0] >= 0 and grid_lon[-1] > 180:
        lons = np.where(lons < 0, lons + 360, lons)

    mid_lat = (lats[:-1] + lats[1:]) / 2
    mid_lon = (lons[:-1] + lons[1:]) / 2

    dep_dt64 = wp_times[0]
    dep_offset_h = float((dep_dt64 - wind_grid["t0"]) / np.timedelta64(1, "h"))
    cum_h = np.cumsum(seg_dt_h)
    seg_mid_h = dep_offset_h + cum_h - seg_dt_h / 2

    u10 = query(wind_grid, "u10", mid_lat, mid_lon, seg_mid_h)
    v10 = query(wind_grid, "v10", mid_lat, mid_lon, seg_mid_h)
    swh = query(wave_grid, "swh", mid_lat, mid_lon, seg_mid_h)
    mwd = query_angle(wave_grid, "mwd", mid_lat, mid_lon, seg_mid_h)

    seg_dist_m = _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    v_mps = seg_dist_m / (seg_dt_h * 3600.0)

    bearing_deg = _forward_bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])

    tws = np.sqrt(u10**2 + v10**2)
    wind_from_deg = np.mod(180.0 + np.degrees(np.arctan2(u10, v10)), 360.0)
    twa_deg = np.mod(wind_from_deg - bearing_deg, 360.0)
    mwa_deg = np.mod(mwd - bearing_deg, 360.0)

    power_kw = power_fn(tws, twa_deg, swh, mwa_deg, v_mps, wps)

    energy_mwh = float(np.sum(power_kw * seg_dt_h) / 1000.0)
    return {
        "energy_mwh": energy_mwh,
        "max_wind_mps": float(np.max(tws)),
        "max_hs_m": float(np.max(swh)),
        "sailed_distance_nm": float(np.sum(seg_dist_m) / 1852.0),
    }
