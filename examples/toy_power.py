"""A toy ship power model for examples and tests.

TiMBERS ships no vessel power model; you inject your own ``power_fn`` with the
signature ``power_fn(tws, twa_deg, swh, mwa_deg, v, wps) -> kW``. This module is
a deliberately trivial stand-in (made-up round coefficients, not calibrated to
any real ship) so the pipeline can be run and tested end to end.

Two bindings are provided because the device path (``timbers.model`` /
``timbers.optimizer``) operates on JAX arrays while the host scorer
(``timbers.scoring``) operates on NumPy arrays; both share one implementation.

Both also take an optional ``params``, one draw of the model's uncertain
parameters, for power-model uncertainty (``model_params`` in
:mod:`timbers.ensemble`). Its keys, any subset of :data:`NOMINAL`, scale the
calm-water, wave and wind terms and set the speed exponent ``n`` of the hull
term, which keeps its value at ``V_REF``: a different ``n`` changes the shape
of the speed-power curve, not its level.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

NOMINAL = dict(calm=1.0, wave=1.0, wind=1.0, n=3.0)
V_REF = 30.0  # m/s, about the demo corridor's mean speed


def _toy(xp, tws, twa_deg, swh, mwa_deg, v, wps, params=None):
    k = NOMINAL if params is None else {**NOMINAL, **params}
    twa = xp.radians(twa_deg)
    mwa = xp.radians(mwa_deg)
    p_hull = k["calm"] * 5.0 * v**3 * (v / V_REF) ** (k["n"] - 3.0)  # cubic hull drag
    p_wind = k["wind"] * 2.0 * tws * (1.0 - xp.cos(twa))  # headwind costs most
    p_wave = k["wave"] * 8.0 * swh**2 * v * (1.0 + 0.5 * xp.cos(mwa))  # head seas worst
    p = p_hull + p_wind + p_wave
    if wps:  # crude beam-wind sail credit
        p = p - 1.5 * tws * v * xp.abs(xp.sin(twa))
    return xp.maximum(p, 0.0)


def toy_power_np(tws, twa_deg, swh, mwa_deg, v, wps=False, params=None):
    """Toy power (kW) on NumPy arrays — for ``timbers.scoring``."""
    return _toy(np, tws, twa_deg, swh, mwa_deg, v, wps, params)


def toy_power_jax(tws, twa_deg, swh, mwa_deg, v, wps=False, params=None):
    """Toy power (kW) on JAX arrays — for ``timbers.model`` / ``timbers.optimizer``."""
    return _toy(jnp, tws, twa_deg, swh, mwa_deg, v, wps, params)
