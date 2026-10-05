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
from datetime import datetime, timedelta

import numpy as np

from .era5 import query, query_angle
from .geo import midpoint_lon

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
    wp_times = np.array([np.datetime64(wp[0]) for wp in waypoints], dtype="datetime64[s]")
    seg_dt_h = ((wp_times[1:] - wp_times[:-1]) / np.timedelta64(1, "h")).astype(np.float64)
    seg_dt_h = np.maximum(seg_dt_h, 1e-6)

    # Normalize longitudes for ERA5 grid
    grid_lon = wind_grid["lon"]
    wrap = bool(grid_lon[0] >= 0 and grid_lon[-1] > 180)

    # Segment midpoints
    mid_lat = (lats[:-1] + lats[1:]) / 2
    mid_lon = midpoint_lon(lons[:-1], lons[1:], wrap)

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
    ``max_power_kw``,
    ``sailed_distance_nm``. The energy computation is byte-for-byte the same as
    :func:`evaluate_route`.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")

    waypoints = _resample(waypoints, resample_dt_min)

    lats = np.array([wp[1] for wp in waypoints])
    lons = np.array([wp[2] for wp in waypoints])

    wp_times = np.array([np.datetime64(wp[0]) for wp in waypoints], dtype="datetime64[s]")
    seg_dt_h = ((wp_times[1:] - wp_times[:-1]) / np.timedelta64(1, "h")).astype(np.float64)
    seg_dt_h = np.maximum(seg_dt_h, 1e-6)

    grid_lon = wind_grid["lon"]
    wrap = bool(grid_lon[0] >= 0 and grid_lon[-1] > 180)

    mid_lat = (lats[:-1] + lats[1:]) / 2
    mid_lon = midpoint_lon(lons[:-1], lons[1:], wrap)

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
        "max_power_kw": float(np.max(power_kw)),
        "sailed_distance_nm": float(np.sum(seg_dist_m) / 1852.0),
    }


# ---------------------------------------------------------------------------
# Scoring under a shaft-power ceiling
# ---------------------------------------------------------------------------
def v_max_for_power(
    power_fn, tws, twa_deg, swh, mwa_deg, wps, p_max, v_hi: float = 20.0, steps: int = 40
):
    """Largest speed (m/s) whose demanded shaft power stays within ``p_max``.

    Requires power to be non-decreasing in speed for fixed weather, so that
    ``sup{v : P(v) <= p_max}`` is well defined. Vectorised over the weather
    arguments; fixed-iteration bisection on ``[0, v_hi]``, so every call costs
    the same and the result always satisfies the ceiling. Returns ``v_hi`` where
    the ceiling cannot be reached within the bracket.
    """
    lo = np.zeros_like(np.asarray(tws, dtype=float))
    hi = np.full_like(lo, v_hi)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        over = power_fn(tws, twa_deg, swh, mwa_deg, mid, wps) > p_max
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    return lo


def _signed(lon):
    """Longitude in [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def _path_arrays(lat, lon):
    """Cumulative along-track distance (m), leg lengths and leg bearings."""
    seg_m = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    brg = _forward_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(seg_m)]), seg_m, brg


def _at_distance(lat, lon, cum_m, d):
    """Position and leg index at along-track distance ``d``."""
    i = int(np.clip(np.searchsorted(cum_m, d, side="right") - 1, 0, len(lat) - 2))
    span = max(cum_m[i + 1] - cum_m[i], 1e-9)
    f = (d - cum_m[i]) / span
    return lat[i] + f * (lat[i + 1] - lat[i]), lon[i] + f * (lon[i + 1] - lon[i]), i


def evaluate_route_saturated(
    wind_grid,
    wave_grid,
    dep,
    lat,
    lon,
    seg_dt_h,
    power_fn,
    *,
    wps=False,
    p_max=np.inf,
    dt_h=0.25,
    max_hours=None,
    t_offset_h=0.0,
    stall_factor=2.0,
):
    """Sail a planned route forward in time under a shaft-power ceiling.

    The ship follows the commanded speed of each leg (leg length over planned
    duration) unless the weather would demand more than ``p_max``, in which case
    it sails at the speed the ceiling allows:

        v_achieved = min(v_commanded, v_max(weather, p_max))

    There is no schedule recovery, so a ship that falls behind stays behind, and
    meets later weather at a later clock time than planned. Time is integrated
    in steps of ``dt_h`` with the midpoint rule.

    Parameters
    ----------
    wind_grid, wave_grid : dict
        Grids as returned by :func:`timbers.era5.load_era5`.
    dep : datetime
        Departure time.
    lat, lon : array_like, length L
        Planned track, signed longitude (e.g. from ``optimizer.decode_route``
        with ``optimizer.working_to_signed``). The track may cross the
        antimeridian; positions are interpolated in continuous longitude.
    seg_dt_h : array_like, length L-1
        Planned leg durations in hours; their sum is the scheduled passage time.
    power_fn : callable
        NumPy power model ``(tws, twa_deg, swh, mwa_deg, v, wps) -> kW``,
        non-decreasing in ``v``.
    p_max : float
        Shaft-power ceiling in kW. ``inf`` follows the schedule exactly.
    max_hours : float or None
        Stop after this many hours and report the position reached, for
        example to sail one forecast cycle before re-planning.
    t_offset_h : float
        Hours already elapsed since ``dep``, so a later leg of a voyage reads
        weather at the right clock time.
    stall_factor : float
        Give up when elapsed time exceeds this multiple of the planned time.

    Returns
    -------
    dict
        ``energy_mwh``, ``max_hs_m``, ``max_wind_mps``, ``max_power_kw``,
        ``sailed_distance_nm`` (planned track length), ``planned_hours``,
        ``actual_hours``, ``delay_h``, ``arrived``, ``arrival`` (datetime),
        ``fraction_done``, ``end_lat``, ``end_lon``, ``saturated_frac`` (share
        of steps limited by the ceiling) and ``track`` (realised ``t_h``,
        ``lat``, ``lon``). ``actual_hours``, ``delay_h`` and ``track["t_h"]``
        count from the start of this call, i.e. exclude ``t_offset_h``;
        ``arrival`` includes it. Longitudes are signed.
    """
    lat = np.asarray(lat, float)
    lon = np.unwrap(np.asarray(lon, float), period=360.0)  # continuous across 180
    seg_dt_h = np.asarray(seg_dt_h, float)
    cum_m, seg_m, brg = _path_arrays(lat, lon)
    total_m = cum_m[-1]
    planned_h = float(seg_dt_h.sum())
    v_cmd_leg = seg_m / np.maximum(seg_dt_h * 3600.0, 1e-9)

    glon = wind_grid["lon"]
    wrap = glon[0] >= 0 and glon[-1] > 180
    dep_off = float(
        (np.datetime64(dep.replace(tzinfo=None), "s") - wind_grid["t0"]) / np.timedelta64(1, "h")
    )
    limit_h = np.inf if max_hours is None else float(max_hours)

    def sample(dist, hours):
        p_lat, p_lon, leg = _at_distance(lat, lon, cum_m, min(dist, total_m))
        p_lon = _signed(p_lon)
        q_lon = p_lon + 360.0 if (wrap and p_lon < 0) else p_lon
        h = dep_off + t_offset_h + hours
        u = float(query(wind_grid, "u10", [p_lat], [q_lon], [h])[0])
        v = float(query(wind_grid, "v10", [p_lat], [q_lon], [h])[0])
        hs = float(query(wave_grid, "swh", [p_lat], [q_lon], [h])[0])
        mwd = float(query_angle(wave_grid, "mwd", [p_lat], [q_lon], [h])[0])
        tws = float(np.hypot(u, v))
        wind_from = np.mod(180.0 + np.degrees(np.arctan2(u, v)), 360.0)
        return (
            tws,
            float(np.mod(wind_from - brg[leg], 360.0)),
            hs,
            float(np.mod(mwd - brg[leg], 360.0)),
            leg,
        )

    def speed(tws, twa, hs, mwa, leg):
        v_cmd = v_cmd_leg[leg]
        if not np.isfinite(p_max):
            return v_cmd, False
        v_cap = float(v_max_for_power(power_fn, tws, twa, hs, mwa, wps, p_max))
        return min(v_cmd, v_cap), v_cap < v_cmd - 1e-9

    d = t_h = energy_kwh = 0.0
    n_sat = n_step = 0
    max_hs = max_tws = max_p = 0.0
    trk_t, trk_lat, trk_lon = [0.0], [lat[0]], [_signed(lon[0])]
    while d < total_m and t_h < min(limit_h, stall_factor * planned_h):
        tws0, twa0, hs0, mwa0, leg0 = sample(d, t_h)
        v0, _ = speed(tws0, twa0, hs0, mwa0, leg0)
        step_h = min(dt_h, (total_m - d) / max(v0 * 3600.0, 1e-9))
        tws, twa, hs, mwa, leg = sample(d + v0 * (step_h / 2) * 3600.0, t_h + step_h / 2)
        v_act, sat = speed(tws, twa, hs, mwa, leg)
        p_kw = float(power_fn(tws, twa, hs, mwa, v_act, wps))
        step_h = min(step_h, (total_m - d) / max(v_act * 3600.0, 1e-9), max(limit_h - t_h, 1e-9))
        energy_kwh += p_kw * step_h
        d += v_act * step_h * 3600.0
        t_h += step_h
        n_sat += int(sat)
        n_step += 1
        max_hs, max_tws, max_p = max(max_hs, hs), max(max_tws, tws), max(max_p, p_kw)
        a, b, _ = _at_distance(lat, lon, cum_m, min(d, total_m))
        trk_t.append(t_h)
        trk_lat.append(a)
        trk_lon.append(_signed(b))

    end_lat, end_lon, _ = _at_distance(lat, lon, cum_m, min(d, total_m))
    return dict(
        energy_mwh=energy_kwh / 1000.0,
        max_hs_m=max_hs,
        max_wind_mps=max_tws,
        max_power_kw=max_p,
        sailed_distance_nm=float(total_m / 1852.0),
        planned_hours=planned_h,
        actual_hours=float(t_h),
        delay_h=float(t_h - planned_h),
        arrived=bool(d >= total_m - 1.0),
        arrival=dep + timedelta(hours=float(t_offset_h + t_h)),
        fraction_done=float(min(d / max(total_m, 1e-9), 1.0)),
        end_lat=float(end_lat),
        end_lon=float(_signed(end_lon)),
        saturated_frac=float(n_sat / max(n_step, 1)),
        track=dict(t_h=trk_t, lat=trk_lat, lon=trk_lon),
    )
