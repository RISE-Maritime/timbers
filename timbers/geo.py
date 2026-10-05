#!/usr/bin/env python
"""Geodesic helpers: great-circle interpolation and distance."""

from __future__ import annotations

import numpy as np

__all__ = ["great_circle_points", "gc_distance_nm", "midpoint_lon"]

_R_EARTH_M = 6_371_000.0


def _to_xyz(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def great_circle_points(
    lat0: float, lon0: float, lat1: float, lon1: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` points along the great circle, evenly spaced in arc length.

    Includes both endpoints (so ``n`` >= 2). Uses spherical linear interpolation
    (slerp) of the endpoint unit vectors. Returns ``(lats_deg, lons_deg)`` with
    longitudes in [-180, 180).
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    p0 = _to_xyz(lat0, lon0)
    p1 = _to_xyz(lat1, lon1)
    dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    omega = np.arccos(dot)
    f = np.linspace(0.0, 1.0, n)
    if omega < 1e-12:  # coincident endpoints
        pts = np.outer(np.ones_like(f), p0)
    else:
        s0 = np.sin((1 - f) * omega) / np.sin(omega)
        s1 = np.sin(f * omega) / np.sin(omega)
        pts = s0[:, None] * p0[None, :] + s1[:, None] * p1[None, :]
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def gc_distance_nm(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Great-circle distance in nautical miles."""
    p0 = _to_xyz(lat0, lon0)
    p1 = _to_xyz(lat1, lon1)
    omega = np.arccos(float(np.clip(np.dot(p0, p1), -1.0, 1.0)))
    return omega * _R_EARTH_M / 1852.0


def midpoint_lon(lon_a, lon_b, wrap: bool):
    """Midpoint of the shorter arc between two longitudes, in a grid's convention.

    Averaging longitudes directly is wrong for a segment across the seam of the
    convention it is written in: in signed longitude the midpoint of 179.9 and
    -179.9 comes out at 0, and in 0-360 longitude the midpoint of 359.9 and 0.1
    comes out at 180. Taking half of the shorter signed difference avoids both.

    Returns longitudes in [0, 360) if ``wrap`` (a grid in 0-360 longitude),
    otherwise in [-180, 180). Arithmetic only, so it works elementwise on NumPy
    and JAX arrays alike.
    """
    d = (lon_b - lon_a + 180.0) % 360.0 - 180.0
    mid = lon_a + d / 2
    if not wrap:
        mid = mid + 180.0
    m = mid % 360.0
    m = m - 360.0 * (m >= 360.0)  # a tiny negative mid rounds to exactly 360.0
    return m if wrap else m - 180.0
