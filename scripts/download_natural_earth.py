#!/usr/bin/env python
"""Fetch the Natural Earth land polygons (public domain) used for the land mask.

Downloads ne_10m_land.json and ne_10m_minor_islands.json from the
natural-earth-geojson mirror into data/natural-earth/10m/physical/.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/martynafford/"
        "natural-earth-geojson/master/10m/physical")
FILES = ["ne_10m_land.json", "ne_10m_minor_islands.json"]
OUT = Path("data/natural-earth/10m/physical")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        dst = OUT / fn
        if dst.exists():
            print(f"skip {dst} (exists)")
            continue
        print(f"fetching {fn} ...")
        urllib.request.urlretrieve(f"{BASE}/{fn}", dst)
        print(f"  -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
