#!/usr/bin/env python
"""ERA5 grid loader.

Provides the three functions used by ``timbers.scoring``:

    load_era5(paths)            -> grid dict
    query(grid, var, lat, lon, hours)        -> linear interp of a scalar field
    query_angle(grid, var, lat, lon, hours)  -> circular interp of an angle field

Grids are concatenated along time from one or more monthly ERA5 NetCDF files.
The loader is tolerant of both ERA5 file schemas found in the CDS archive:

  * legacy: time variable ``time`` in "hours since 1900-01-01"
  * current: time variable ``valid_time`` in "seconds since 1970-01-01"

Latitude may be stored ascending or descending; longitude is assumed to be in
[0, 360). Interpolation is trilinear (lat, lon, time) on the regular grid,
matching the reference scoring logic.

Corridor grids are produced by ``scripts/download_era5.py``; they are cropped
to a corridor bounding box so a full year fits comfortably in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # netCDF4 is required to read the .nc files
    import netCDF4 as nc
except ImportError as exc:  # pragma: no cover
    raise ImportError("timbers.era5 requires the 'netCDF4' package") from exc

__all__ = ["load_era5", "query", "query_angle"]

# Data variables we know how to load, by file type. Coordinate/metadata
# variables are skipped.
_SKIP_VARS = {
    "number",
    "expver",
    "time",
    "valid_time",
    "latitude",
    "longitude",
    "lat",
    "lon",
}


# ---------------------------------------------------------------------------
# Time decoding
# ---------------------------------------------------------------------------
_UNIT_SECONDS = {"hour": 3600.0, "second": 1.0, "minute": 60.0, "day": 86400.0}


def _times_to_datetime64(tvar) -> np.ndarray:
    """Convert a NetCDF time variable to a ``datetime64[s]`` array."""
    units = getattr(tvar, "units", "")
    vals = np.asarray(tvar[:], dtype=np.float64)
    if "since" not in units:
        raise ValueError(f"Unrecognised time units: {units!r}")
    kind, _, epoch_str = units.partition(" since ")
    kind = kind.strip().lower().rstrip("s")  # "hours" -> "hour"
    if kind not in _UNIT_SECONDS:
        raise ValueError(f"Unrecognised time kind in units {units!r}")

    # Normalise "YYYY-MM-DD HH:MM:SS.s" -> ISO 8601 the way datetime64 wants.
    epoch_str = epoch_str.strip().split(".")[0].replace(" ", "T")
    epoch = np.datetime64(epoch_str).astype("datetime64[s]")

    deltas = (vals * _UNIT_SECONDS[kind]).round().astype("timedelta64[s]")
    return epoch + deltas


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_era5(paths: list[str] | str) -> dict:
    """Load and time-concatenate one or more ERA5 NetCDF files.

    Parameters
    ----------
    paths : str or list of str
        File paths. They are sorted by their first timestamp and concatenated
        along the time axis. All files must share the same lat/lon grid.

    Returns
    -------
    dict
        Keys: ``lat`` (1D), ``lon`` (1D, ascending in [0,360)), ``t0``
        (datetime64[s] of first time), ``times`` (datetime64[s] array),
        ``dt_h`` (mean hourly spacing), plus one (T, Y, X) float32 array per
        data variable (e.g. ``u10``, ``v10``, ``swh``, ``mwd``). Land-masked
        cells are filled with 0.0.
    """
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("load_era5: no paths given")

    per_file = []
    lat = lon = None
    var_names = None
    for p in paths:
        ds = nc.Dataset(p)
        tname = "valid_time" if "valid_time" in ds.variables else "time"
        times = _times_to_datetime64(ds.variables[tname])
        f_lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        f_lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        if lat is None:
            lat, lon = f_lat, f_lon
        elif lat.shape != f_lat.shape or lon.shape != f_lon.shape:
            raise ValueError(f"Grid mismatch between files at {p}")

        names = [
            v
            for v in ds.variables
            if v not in _SKIP_VARS
            and ds.variables[v].dimensions[-2:] == ("latitude", "longitude")
        ]
        if var_names is None:
            var_names = names
        data = {}
        for v in var_names:
            arr = ds.variables[v][:]
            arr = np.ma.filled(arr, 0.0).astype(np.float32)
            data[v] = arr
        per_file.append((times[0], times, data))
        ds.close()

    per_file.sort(key=lambda t: t[0])
    times = np.concatenate([pf[1] for pf in per_file])
    grid = {v: np.concatenate([pf[2][v] for pf in per_file], axis=0) for v in var_names}

    grid["lat"] = lat
    grid["lon"] = lon
    grid["times"] = times
    grid["t0"] = times[0]
    dt = np.diff(times) / np.timedelta64(1, "h")
    grid["dt_h"] = float(np.median(dt)) if dt.size else 1.0
    return grid


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def _frac_index(coord: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower integer index, fractional offset) for a regular axis.

    Handles ascending or descending ``coord``. Values are clamped to the grid.
    """
    n = coord.shape[0]
    step = coord[1] - coord[0]  # signed
    fi = (values - coord[0]) / step
    fi = np.clip(fi, 0.0, n - 1.0)
    i0 = np.floor(fi).astype(np.intp)
    i0 = np.minimum(i0, n - 2)
    frac = fi - i0
    return i0, frac


def _corner_indices(grid, lat, lon, hours):
    """Return the 8 corner index arrays and the (tf, yf, xf) fractions."""
    yi, yf = _frac_index(grid["lat"], np.asarray(lat, dtype=np.float64))
    xi, xf = _frac_index(grid["lon"], np.asarray(lon, dtype=np.float64))
    nt = grid["times"].shape[0]
    ti_real = np.asarray(hours, dtype=np.float64) / grid["dt_h"]
    ti_real = np.clip(ti_real, 0.0, nt - 1.0)
    ti = np.minimum(np.floor(ti_real).astype(np.intp), max(nt - 2, 0))
    tf = ti_real - ti
    return ti, yi, xi, tf, yf, xf


def _combine(corners, tf, yf, xf):
    """Trilinear blend of 8 corner-value arrays keyed (dt, dy, dx)."""
    c00 = corners[(0, 0, 0)] * (1 - xf) + corners[(0, 0, 1)] * xf
    c01 = corners[(0, 1, 0)] * (1 - xf) + corners[(0, 1, 1)] * xf
    c10 = corners[(1, 0, 0)] * (1 - xf) + corners[(1, 0, 1)] * xf
    c11 = corners[(1, 1, 0)] * (1 - xf) + corners[(1, 1, 1)] * xf
    c0 = c00 * (1 - yf) + c01 * yf
    c1 = c10 * (1 - yf) + c11 * yf
    return c0 * (1 - tf) + c1 * tf


def _gather(field, ti, yi, xi):
    """Gather the 8 trilinear corners of ``field`` (T,Y,X) as float64."""
    return {
        (dt, dy, dx): field[ti + dt, yi + dy, xi + dx].astype(np.float64)
        for dt in (0, 1)
        for dy in (0, 1)
        for dx in (0, 1)
    }


def query(grid: dict, var: str, lat, lon, hours) -> np.ndarray:
    """Trilinear interpolation of scalar field ``var`` at scattered points.

    ``hours`` is hours since ``grid['t0']``. ``lat``/``lon``/``hours`` are
    array-like of equal length; returns a 1D float array.
    """
    ti, yi, xi, tf, yf, xf = _corner_indices(grid, lat, lon, hours)
    return _combine(_gather(grid[var], ti, yi, xi), tf, yf, xf)


def query_angle(grid: dict, var: str, lat, lon, hours) -> np.ndarray:
    """Circular (sin/cos) interpolation of an angle field in degrees.

    Gathers raw angle corners and blends their sine/cosine components, so only
    the 8 needed corners are transcendentally transformed (not the whole field).
    Returns degrees in [0, 360).
    """
    ti, yi, xi, tf, yf, xf = _corner_indices(grid, lat, lon, hours)
    raw = _gather(grid[var], ti, yi, xi)
    rad = {k: np.radians(v) for k, v in raw.items()}
    s = _combine({k: np.sin(v) for k, v in rad.items()}, tf, yf, xf)
    c = _combine({k: np.cos(v) for k, v in rad.items()}, tf, yf, xf)
    return np.mod(np.degrees(np.arctan2(s, c)), 360.0)
