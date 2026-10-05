"""Fill land-masked weather cells from the nearest sea cell.

Wave reanalyses and forecasts leave land cells undefined (ERA5 returns a masked
array, ENS GRIB decodes to NaN). A zero fill would make land read as flat calm,
cheaper in power and never breaching a wave limit, and so reward a route for
cutting across a headland or sheltering behind a coast; it would also bias
wave height low within one cell of every coastline through the interpolation
stencil.

Carrying the nearest sea cell's value inland avoids both without inventing a
hazard: land is not a calm corridor, and a route that strays still reads
plausible weather. Keeping routes off land is the exclusion
penalty's job (``timbers.land.exclusion_raster``), not the wave field's.
"""

from __future__ import annotations

import numpy as np


def nearest_sea_index(masked: np.ndarray):
    """``(yi, xi)`` giving, for every cell of a 2-D grid, the nearest unmasked cell.

    Unmasked cells index themselves. Raises if every cell is masked.
    """
    from scipy.ndimage import distance_transform_edt

    masked = np.asarray(masked, bool)
    if masked.all():
        raise ValueError("every cell is masked, so there is nothing to fill from")
    _, (yi, xi) = distance_transform_edt(masked, return_indices=True)
    return yi, xi


def fill_from_nearest_sea(a: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """``a`` with masked cells replaced by the nearest unmasked cell's value.

    ``a`` is ``(..., Y, X)`` with the grid on the last two axes. ``masked`` is
    boolean, either ``(Y, X)`` or the full shape of ``a``. The nearest-neighbour
    map is built once from the union of the mask over the leading axes, and only
    the cells masked at a given time (or member) are replaced. Returns ``a``
    itself when nothing is masked.
    """
    masked = np.asarray(masked, bool)
    if masked.shape[-2:] != a.shape[-2:]:
        raise ValueError(f"mask {masked.shape} does not match grid {a.shape[-2:]}")
    grid = masked.reshape(-1, *masked.shape[-2:]).any(axis=0) if masked.ndim > 2 else masked
    if not grid.any():
        return a
    yi, xi = nearest_sea_index(grid)
    return np.where(masked, a[..., yi, xi], a)
