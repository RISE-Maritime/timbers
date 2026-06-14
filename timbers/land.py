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


def build_mask(box, res_deg: float = RES_DEG, ne_dir: Path = NE_DIR,
               cache: Path | None = None) -> dict:
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
