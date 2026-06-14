"""A toy ship power model for examples and tests.

TIMBERS ships no vessel power model; you inject your own ``power_fn`` with the
signature ``power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW``. This module is
a deliberately trivial stand-in (made-up round coefficients, not calibrated to
any real ship) so the pipeline can be run and tested end to end.

Two bindings are provided because the device path (``timbers.model`` /
``timbers.optimizer``) operates on JAX arrays while the host scorer
(``timbers.scoring``) operates on NumPy arrays; both share one implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _toy(xp, tws, twa_deg, swh, mwa_deg, v, wps):
    twa = xp.radians(twa_deg)
    mwa = xp.radians(mwa_deg)
    p_hull = 5.0 * v ** 3                              # cubic hull drag
    p_wind = 2.0 * tws * (1.0 - xp.cos(twa))          # headwind costs most
    p_wave = 8.0 * swh ** 2 * v * (1.0 + 0.5 * xp.cos(mwa))  # head seas worst
    p = p_hull + p_wind + p_wave
    if wps:                                            # crude beam-wind sail credit
        p = p - 1.5 * tws * v * xp.abs(xp.sin(twa))
    return xp.maximum(p, 0.0)


def toy_power_np(tws, twa_deg, swh, mwa_deg, v, wps=False):
    """Toy power (kW) on NumPy arrays — for ``timbers.scoring``."""
    return _toy(np, tws, twa_deg, swh, mwa_deg, v, wps)


def toy_power_jax(tws, twa_deg, swh, mwa_deg, v, wps=False):
    """Toy power (kW) on JAX arrays — for ``timbers.model`` / ``timbers.optimizer``."""
    return _toy(jnp, tws, twa_deg, swh, mwa_deg, v, wps)
