"""Route optimisation and scoring under weather uncertainty.

The weather comes as an ensemble: the members of a real forecast (such as the 51
members of ECMWF ENS, a :class:`timbers.model.Grids` with a member axis), or a
surrogate ensemble built by perturbing one field (``perturbations``, see below),
or both. One route cost serves four objectives that differ only in how the
per-member outcomes are reduced to a scalar:

    ==================  =======================  =========================
                        safety from member 0     safety over all members
    ==================  =======================  =========================
    cost from member 0  ``deterministic``        ``chance_constrained``
    cost, member mean   ``expected_value``       ``joint``
    ==================  =======================  =========================

Member 0 is the nominal member (the control forecast in ENS, or the unperturbed
row of a perturbation grid). Parameterisation, resampling, interpolation, power
model and land penalty are shared, so a difference between objectives is a
difference in formulation.

The expectation is taken over per-member costs, not over the weather field:
power is convex in wave height, so averaging the weather first would understate
the cost of exactly the storm cases an ensemble is meant to represent.

**Perturbation surrogate.** Without a real ensemble, forecast error can be
approximated by perturbing where, when and how strongly one field is read:
``perturbation_grid`` builds rows ``(dlat, dlon, dt_h, hs_scale, wind_scale)``,
and passing them as ``perturbations`` makes every (perturbation, member) pair a
member, perturbation-major, so row 0 of a single-member grid is member 0. The
route's own geometry and speed are not perturbed. ``score_members`` then gives
a fixed route's fragility, and ``safety_mode="mean"`` the expected exceedance
penalty over the surrogate.

**Validity domain of the shift/scale surrogate.** Measured against real ECMWF
ENS (51 members, along-route Hs, two North Atlantic storm departures), the
surrogate approximately matches real ensemble spread out to about two days of
lead time (spread ratio 0.75-0.97 at 12-72 h) and understates it beyond:
1.45-1.89x at 72-144 h, 2.4-6.2x at 144-366 h. The mismatch is structural
rather than a tuning issue. Real forecast spread grows with lead time, while
the surrogate's tracks the amplitude of the one field being perturbed, and the
real member distribution is right-skewed (skew 0.4-1.4), which a symmetric
shift and scale cannot produce. Use the surrogate as a mechanism demonstration
or for horizons within about two days. For longer passages, score against real
ensemble members instead.

Typical use::

    grids = Grids(wind, wave, steps)                  # fields (member, time, y, x)
    fit, shared = make_ensemble_cost(grids, land, cor, objective="joint", ...)
    best, best_j = cmaes.run(x0, fit, 0.1, hp, pop, iters, key,
                             (jnp.float32(dep_off), *shared))
    lat, wlon, seg = optimizer.decode_route(best, cor, K, L, n_speed)
    members = score_members(grids, cor, lat, wlon, seg, ...)   # per-member outcomes
"""

from __future__ import annotations

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import model as jm
from . import optimizer as op

OBJECTIVES = ("deterministic", "chance_constrained", "expected_value", "joint")
SAFETY_MODES = ("prob", "cvar", "mean")


def perturbation_grid(dlat=(0.0,), dlon=(0.0,), dt=(0.0,), hs=(1.0,), wind=(1.0,)):
    """Cartesian product of perturbation axes -> ``(Pn, 5)`` float32 rows
    ``(dlat, dlon, dt_h, hs_scale, wind_scale)``.

    With the nominal value first on every axis (the defaults), row 0 is the
    unperturbed field.
    """
    return np.array(list(itertools.product(dlat, dlon, dt, hs, wind)), np.float32)


def _members(fields, axes, segs, power_fn, wps, lon_wrap, perturbations):
    """Power, Hs and TWS ``(member, segment)``; with ``perturbations`` every
    (perturbation, grid member) pair is a member, perturbation-major."""
    if perturbations is None:
        return jm.sample(fields, axes, segs, power_fn, wps, lon_wrap)
    out = jax.vmap(lambda pt: jm.sample(fields, axes, segs, power_fn, wps, lon_wrap, pt))(
        jnp.asarray(perturbations, jnp.float32)
    )
    return tuple(x.reshape(-1, x.shape[-1]) for x in out)


# --- building blocks ----------------------------------------------------------
# Penalties grow exponentially up to a knee and linearly beyond it, with the
# slope matched at the join: sharp near a limit, where sharpness herds routes
# against the boundary, and never flat however bad a candidate is. A clipped
# exponential would overflow-protect but leave a plateau with no ranking signal.
_KNEE = 12.0
_KNEE_VAL = float(np.exp(_KNEE))


def _sexp(x):
    """exp(x) below the knee, linear continuation above; continuous slope."""
    lo = jnp.exp(jnp.minimum(x, _KNEE))
    return jnp.where(x <= _KNEE, lo, _KNEE_VAL * (1.0 + (x - _KNEE)))


def _cvar(x, eps):
    """Mean of the worst ``ceil(eps * M)`` members (CVaR at level 1 - eps)."""
    k = max(1, int(np.ceil(eps * x.shape[-1])))
    return jnp.mean(jnp.sort(x, axis=-1)[..., -k:], axis=-1)


def _p_hat(margins, sharpness):
    """Smoothed fraction of members whose margin is positive."""
    return jnp.mean(jax.nn.sigmoid(sharpness * margins), axis=-1)


def make_ensemble_cost(
    grids,
    land,
    cor,
    *,
    objective,
    L,
    K,
    n_speed,
    align,
    wps,
    power_fn,
    hs_lim,
    tws_lim,
    p_lim=float("inf"),
    eps=0.1,
    lam_env=30.0,
    lam_land=1e6,
    a_env=6.0,
    soft_frac=0.93,
    safety_mode="prob",
    sharpness=50.0,
    perturbations=None,
):
    """Return ``(fit, shared)`` for :func:`timbers.cmaes.run`.

    ``fit(theta_batch, cargs)`` with ``cargs = (dep_off, *shared)``, where
    ``dep_off`` is the departure time in hours after the grids' ``t0``. The
    fields travel in ``shared`` rather than being closed over, so they are not
    baked into the compiled generation loop as constants (which would hold them
    in device memory twice).

    The route is resampled to a uniform time grid of about ``align`` hours, as
    the scorer integrates, so the optimised quantity is the scored one. For each
    member the cost computes energy (MWh), the worst normalised seakeeping
    margin ``max(Hs/hs_lim, TWS/tws_lim) - 1`` along the route, and a soft
    penalty that rises from ``soft_frac`` of each limit (Hs, TWS and, if finite,
    shaft power ``p_lim``). The objective then combines them:

    * cost: member 0's energy, or the member mean (``expected_value``,
      ``joint``);
    * safety: member 0's soft penalty, or an ensemble term (``chance_constrained``,
      ``joint``) chosen by ``safety_mode``.

    ``safety_mode="prob"`` (default) constrains the smoothed fraction of members
    that breach a limit, so ``eps`` is the violation probability. ``"cvar"``
    constrains the CVaR at level ``1 - eps`` of the member margins, the standard
    convex relaxation; it bounds the violation probability only when it is
    satisfied, and where the constraint is infeasible it measures tail severity
    rather than frequency. ``eps`` is resolved only to ``1 / n_members``.
    ``"mean"`` is the members' mean soft penalty, the expected exceedance; it
    has no ``eps``.

    ``perturbations`` (rows from :func:`perturbation_grid`) multiplies the
    members by a forecast-error surrogate; see the module docstring.

    Shaft power enters only the soft penalty, never the seakeeping margin: a
    power excess makes a voyage slow, not unsafe, and is better represented as
    late arrival (:func:`timbers.scoring.evaluate_route_saturated`).

    ``land`` is a :class:`timbers.optimizer.DeviceLand`; the land term is the
    summed raster along the route, zero at sea, and ``lam_land = 1e6`` makes it
    a hard constraint under rank-based selection.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; one of {OBJECTIVES}")
    if safety_mode not in SAFETY_MODES:
        raise ValueError(f"unknown safety_mode {safety_mode!r}; one of {SAFETY_MODES}")
    use_ensemble_cost = objective in ("expected_value", "joint")
    use_ensemble_safety = objective in ("chance_constrained", "joint")

    M = jm.n_points(cor.hours, align, quantise=True)
    # The penalties are sums over the M points while energy is time-weighted;
    # rescaling by the M the requested align implies keeps the penalty weight
    # independent of where M lands on the ladder.
    pen_scale = (cor.hours / align) / M
    lon_wrap = grids.lon_wrap
    finite_p = bool(np.isfinite(p_lim))

    def one(theta, dep_off, fields, axes, land_arrs):
        lat, wlon, seg_dt = op.theta_to_track(theta, cor, K, L, n_speed)
        rlat, rlon, seg = jm.resample(lat, wlon, seg_dt, cor.hours, M)
        segs = jm.segments(rlat, rlon, seg, dep_off)
        p, swh, tws = _members(fields, axes, segs, power_fn, wps, lon_wrap, perturbations)

        energies = jnp.sum(p * seg, axis=-1) / 1000.0
        margins = jnp.maximum(jnp.max(swh, -1) / hs_lim, jnp.max(tws, -1) / tws_lim) - 1.0
        terms = (
            _sexp(a_env * jnp.maximum(swh / (soft_frac * hs_lim) - 1.0, 0.0))
            + _sexp(a_env * jnp.maximum(tws / (soft_frac * tws_lim) - 1.0, 0.0))
            - 2.0
        )
        if finite_p:
            terms = terms + _sexp(a_env * jnp.maximum(p / (soft_frac * p_lim) - 1.0, 0.0)) - 1.0
        softs = pen_scale * jnp.sum(terms, axis=-1)

        cost = jnp.mean(energies) if use_ensemble_cost else energies[0]
        if not use_ensemble_safety:
            risk = softs[0]
        elif safety_mode == "mean":
            risk = jnp.mean(softs)
        else:
            if safety_mode == "cvar":
                excess = jnp.maximum(_cvar(margins, eps), 0.0)
            else:
                excess = jnp.maximum(_p_hat(margins, sharpness) - eps, 0.0)
            risk = _sexp(a_env * excess) - 1.0
        p_land = pen_scale * op.land_term(land_arrs, rlat, rlon)
        return cost + lam_env * risk + lam_land * p_land

    batched = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None)))

    def fit(theta_batch, cargs):
        dep_off, fields, axes, land_arrs = cargs
        return batched(theta_batch, dep_off, fields, axes, land_arrs)

    return fit, (grids.fields, grids.axes, land.arrays)


def score_members(
    grids,
    cor,
    lat,
    wlon,
    seg_dt,
    *,
    wps,
    power_fn,
    align,
    hs_lim,
    tws_lim,
    p_lim=float("inf"),
    dep_off=0.0,
    perturbations=None,
):
    """Per-member outcomes of a fixed route, sampled as the cost samples it.

    ``lat``, ``wlon``, ``seg_dt`` as returned by
    :func:`timbers.optimizer.decode_route`. Returns host arrays over members:
    ``energy_mwh``, ``max_hs``, ``max_tws``, ``max_power``, ``margin`` (worst
    normalised Hs/TWS margin; positive means a breach) and ``power_margin``.
    ``(margin > 0).mean()`` is the ensemble's breach probability for the route.
    With ``perturbations``, members are (perturbation, grid member) pairs as in
    :func:`make_ensemble_cost`, which makes this the route's fragility under the
    surrogate.
    """
    M = jm.n_points(cor.hours, align, quantise=True)

    @jax.jit
    def run(lat, wlon, seg_dt, dep_off, fields, axes):
        rlat, rlon, seg = jm.resample(lat, wlon, seg_dt, cor.hours, M)
        segs = jm.segments(rlat, rlon, seg, dep_off)
        p, swh, tws = _members(fields, axes, segs, power_fn, wps, grids.lon_wrap, perturbations)
        max_hs, max_tws, max_p = jnp.max(swh, -1), jnp.max(tws, -1), jnp.max(p, -1)
        return (
            jnp.sum(p * seg, axis=-1) / 1000.0,
            max_hs,
            max_tws,
            max_p,
            jnp.maximum(max_hs / hs_lim, max_tws / tws_lim) - 1.0,
            max_p / p_lim - 1.0,
        )

    f32 = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
    out = run(f32(lat), f32(wlon), f32(seg_dt), f32(dep_off), grids.fields, grids.axes)
    keys = ("energy_mwh", "max_hs", "max_tws", "max_power", "margin", "power_margin")
    return {k: np.asarray(v) for k, v in zip(keys, out)}


def member_series(grids, t_h, lat, lon, *, wps, power_fn, pad_to=512):
    """Per-member, per-segment weather and power along a timed track.

    ``t_h`` is hours since the grids' ``t0`` and ``lon`` is signed; the track is
    evaluated segment by segment as given, without resampling, so the result can
    be aggregated over any window (for example by voyage day). Several tracks
    can be evaluated in one call by concatenating them and discarding the
    joining segments. The segment count is padded to a multiple of ``pad_to`` so
    one compiled kernel serves tracks of similar length.

    Returns host arrays: ``power_kw``, ``swh``, ``tws`` of shape
    ``(member, segment)``; ``seg_h`` and ``t_mid_h`` per segment; and ``valid``,
    False where a segment lies outside the forecast's time range (those values
    are clamped and should be ignored).
    """
    t_h = np.asarray(t_h, np.float64)
    seg = t_h[1:] - t_h[:-1]
    # Speed uses a floored duration; mid-times come from t_h itself, so a joining
    # segment between concatenated tracks cannot shift its neighbours.
    segs = jm.segments(
        np.asarray(lat, np.float64),
        np.asarray(lon, np.float64),
        np.maximum(seg, 1e-6),
        0.0,
        xp=np,
    )._replace(t_mid=(t_h[:-1] + t_h[1:]) / 2)
    n = len(seg)
    npad = -(-n // pad_to) * pad_to
    padded = jm.Segments(
        *(jnp.asarray(np.pad(x, (0, npad - n), mode="edge"), jnp.float32) for x in segs)
    )
    p, swh, tws = _series(
        grids.fields, grids.axes, padded, power_fn=power_fn, wps=wps, lon_wrap=grids.lon_wrap
    )
    steps = np.asarray(grids.steps)
    t_mid = np.asarray(segs.t_mid)
    return dict(
        power_kw=np.asarray(p)[:, :n],
        swh=np.asarray(swh)[:, :n],
        tws=np.asarray(tws)[:, :n],
        seg_h=seg,
        t_mid_h=t_mid,
        valid=(t_mid >= steps[0]) & (t_mid <= steps[-1]),
    )


@partial(jax.jit, static_argnames=("power_fn", "wps", "lon_wrap"))
def _series(fields, axes, segs, *, power_fn, wps, lon_wrap):
    return jm.sample(fields, axes, segs, power_fn, wps, lon_wrap)
