#!/usr/bin/env python
"""Geodesic helpers: great-circle interpolation and distance."""

from __future__ import annotations

import numpy as np

__all__ = ["great_circle_points", "gc_distance_nm"]

_R_EARTH_M = 6_371_000.0


def _to_xyz(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


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
