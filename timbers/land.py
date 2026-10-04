#!/usr/bin/env python
"""Rasterize Natural Earth land into a corridor mask for land avoidance.

Point-in-polygon against the full coastline is too slow inside the CMA-ES inner
loop, so we precompute a fine binary land raster over a corridor box and sample
it bilinearly (cheap, JAX-friendly).

Longitude convention: a corridor uses a *continuous working longitude* so the
route never jumps across the antimeridian (e.g. a Pacific corridor in 0-360 lon
stays continuous across 180). Sampling against Natural Earth (which is in
[-180, 180)) converts each point's working lon back to signed via
``((wl + 180) % 360) - 180``.

The raster is optionally cached to a ``.npz``. Natural Earth data (public
domain) is fetched by ``scripts/download_natural_earth.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

NE_DIR = Path("data/natural-earth/10m/physical")
LAND_FILES = ["ne_10m_land.json", "ne_10m_minor_islands.json"]
RES_DEG = 0.05


def _load_land_union(ne_dir: Path):
    geoms = []
    for fn in LAND_FILES:
        gj = json.load(open(ne_dir / fn))
        geoms.extend(shape(f["geometry"]) for f in gj["features"])
    return unary_union(geoms)


def build_mask(
    box, res_deg: float = RES_DEG, ne_dir: Path = NE_DIR, cache: Path | None = None
) -> dict:
    """Build (and optionally cache) the land raster for a corridor box.

    ``box`` is ``(lat_min, lat_max, wlon_min, wlon_max)`` in the continuous
    working-longitude frame used by the optimizer. Returns dict with ``lat``
    (ascending), ``wlon`` (ascending working lon), and ``mask`` (Y, X) float32
    with 1.0 = land.
    """
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return {"lat": z["lat"], "wlon": z["wlon"], "mask": z["mask"]}

    lat_min, lat_max, wl_min, wl_max = box
    lat = np.arange(lat_min, lat_max + res_deg / 2, res_deg)
    wlon = np.arange(wl_min, wl_max + res_deg / 2, res_deg)
    Wl, La = np.meshgrid(wlon, lat)  # (Y, X)
    signed = ((Wl + 180.0) % 360.0) - 180.0

    land = _load_land_union(Path(ne_dir))
    mask = contains_xy(land, signed.ravel(), La.ravel()).reshape(La.shape)
    mask = mask.astype(np.float32)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, lat=lat, wlon=wlon, mask=mask)
    return {"lat": lat, "wlon": wlon, "mask": mask}


# How fast the exclusion grows with penetration, per degree of distance from the
# nearest open water. On a 0/1 raster the bilinear sample is flat more than one
# cell inside the mask, so every candidate that has wandered in looks equally
# bad and the search has no signal pointing back to the water. Ramping with
# depth gives that signal: from anywhere inside, moving towards the nearest
# water lowers the cost. Depth is measured in degrees on the raster grid, which
# is not metric; only monotonicity in penetration is needed.
INLAND_RAMP_PER_DEG = 1.0


def exclusion_raster(
    land: dict, *, extra=None, domain=None, ramp: float = INLAND_RAMP_PER_DEG
) -> dict:
    """Combine static exclusions into one raster with no flat interior.

    ``land`` is a raster from :func:`build_mask`. ``extra`` is an optional
    boolean (Y, X) array of further cells the ship must not enter on the same
    grid, for example water shallower than a draught limit derived from a
    bathymetry such as GEBCO. ``domain`` is an optional
    ``(lat_min, lat_max, wlon_min, wlon_max)`` box; cells outside it are
    excluded. Set it inside the weather grid: weather interpolation clamps at
    the grid edge, so a route that leaves the grid reads a constant field and
    the objective there means nothing.

    Returns the same dict layout with ``mask`` no longer binary: open water is
    0.0, and excluded cells carry 1.0 plus ``ramp`` per degree of distance to
    the nearest open water. ``ramp=0`` returns the plain 0/1 union. The penalty
    code samples and sums the raster, so it works unchanged on the result.
    """
    from scipy.ndimage import distance_transform_edt

    lat = np.asarray(land["lat"])
    wlon = np.asarray(land["wlon"])
    m = (np.asarray(land["mask"]) > 0).astype(np.float32)
    if extra is not None:
        extra = np.asarray(extra, bool)
        if extra.shape != m.shape:
            raise ValueError(f"extra {extra.shape} does not match the raster {m.shape}")
        m = np.maximum(m, extra.astype(np.float32))
    if domain is not None:
        la0, la1, wl0, wl1 = domain
        outside = ((lat < la0) | (lat > la1))[:, None] | ((wlon < wl0) | (wlon > wl1))[None, :]
        m = np.maximum(m, outside.astype(np.float32))
    if m.all():
        raise ValueError("every cell is excluded")
    if ramp:
        d = distance_transform_edt(m > 0, sampling=(abs(lat[1] - lat[0]), abs(wlon[1] - wlon[0])))
        m = m + ramp * d.astype(np.float32)
    return {"lat": lat, "wlon": wlon, "mask": m}
