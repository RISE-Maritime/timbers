"""Spherical geometry shared by the host (NumPy) and device (JAX) paths.

Each function takes the array module as ``xp`` (``numpy`` by default, or
``jax.numpy``), so the scorer and the GPU cost use one implementation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["haversine_m", "bearing_deg", "midpoint_lon", "to_grid_lon", "R_EARTH_M"]

R_EARTH_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2, xp=np):
    """Great-circle distance in metres between (lat, lon) pairs in degrees."""
    lat1r, lat2r = xp.radians(lat1), xp.radians(lat2)
    dlat = lat2r - lat1r
    dlon = xp.radians(lon2 - lon1)
    a = xp.sin(dlat / 2) ** 2 + xp.cos(lat1r) * xp.cos(lat2r) * xp.sin(dlon / 2) ** 2
    return R_EARTH_M * 2 * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2, xp=np):
    """Initial bearing in degrees [0, 360) from point 1 to point 2."""
    lat1r, lat2r = xp.radians(lat1), xp.radians(lat2)
    dlon = xp.radians(lon2 - lon1)
    x = xp.sin(dlon) * xp.cos(lat2r)
    y = xp.cos(lat1r) * xp.sin(lat2r) - xp.sin(lat1r) * xp.cos(lat2r) * xp.cos(dlon)
    return xp.mod(xp.degrees(xp.arctan2(x, y)), 360.0)


def to_grid_lon(lon, wrap: bool):
    """Longitude in a grid's convention: [0, 360) if ``wrap``, else [-180, 180).

    Arithmetic only, so it works on NumPy and JAX arrays alike.
    """
    if not wrap:
        lon = lon + 180.0
    m = lon % 360.0
    m = m - 360.0 * (m >= 360.0)  # a tiny negative input rounds to exactly 360.0
    return m if wrap else m - 180.0


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
    return to_grid_lon(lon_a + d / 2, wrap)
