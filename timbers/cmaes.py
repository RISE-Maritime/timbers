#!/usr/bin/env python
"""GPU-native separable CMA-ES (sep-CMA-ES) in JAX.

Profiling showed the bottleneck of a host-side ES loop is the single-threaded
ask/tell (73% of wall time, GPU idle) and that diagonal CMA is quality-neutral
on this problem. Diagonal CMA has no covariance matrix to eigendecompose --
every update is elementwise -- so the whole generation loop fits inside a jitted lax.scan
with ZERO host round-trips, and the ES state vmaps cleanly over (seeds x departures) so one
dispatch optimizes many departures at once.

sep-CMA-ES: Ros & Hansen (2008), "A Simple Modification in CMA-ES Achieving Linear Time and
Space Complexity" -- standard CMA constants with C constrained diagonal and the (n+2)/3
learning-rate acceleration on the covariance terms.

This module provides the algorithm only (fitness is injected). State per instance:
  m (mean, dim), sigma (step, scalar), c (per-coord variance, dim),
  ps/pc (evolution paths, dim), best_x/best_f, gen, key.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)


def hyperparams(dim, popsize):
    """sep-CMA-ES constants (canonical CMA + (n+2)/3 separable acceleration)."""
    n = float(dim)
    mu = popsize // 2
    w = jnp.log(mu + 0.5) - jnp.log(jnp.arange(1, mu + 1, dtype=jnp.float32))
    w = w / jnp.sum(w)
    mueff = 1.0 / jnp.sum(w ** 2)
    w = jnp.zeros(popsize, jnp.float32).at[:mu].set(w)  # pad -> weighted sum over sorted pop
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = jnp.minimum(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    sep = (n + 2.0) / 3.0                       # separable acceleration
    c1, cmu = c1 * sep, cmu * sep
    scale = jnp.minimum(1.0, 1.0 / (c1 + cmu))  # keep c1+cmu <= 1
    c1, cmu = c1 * scale, cmu * scale
    damps = 1.0 + 2.0 * jnp.maximum(0.0, jnp.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chin = jnp.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
    return dict(mu=mu, w=w, mueff=mueff, cc=cc, cs=cs, c1=c1, cmu=cmu,
                damps=damps, chin=chin, n=n)


def init_state(x0, sigma0, dim, key):
    return dict(
        m=jnp.asarray(x0, jnp.float32), sigma=jnp.float32(sigma0),
        c=jnp.ones(dim, jnp.float32), ps=jnp.zeros(dim, jnp.float32),
        pc=jnp.zeros(dim, jnp.float32), best_x=jnp.asarray(x0, jnp.float32),
        best_f=jnp.float32(jnp.inf), gen=jnp.int32(0), key=key)


def _step(state, fit, hp, popsize, cargs):
    """One generation. ``fit(X, cargs)->(popsize,)`` is the (jittable) batched cost.

    ``cargs`` carries the large grid arrays as TRACED data (not closed-over), so jitting
    ``run`` never bakes them in as multi-GB host constants.
    """
    key, sub = jax.random.split(state["key"])
    dim = state["m"].shape[0]
    Z = jax.random.normal(sub, (popsize, dim))
    sqc = jnp.sqrt(state["c"])
    Y = Z * sqc                                   # (pop, dim)  D*z, D=sqrt(c)
    X = state["m"] + state["sigma"] * Y
    F = fit(X, cargs)                             # (pop,)
    order = jnp.argsort(F)
    w = hp["w"]                                   # (pop,) padded: w[:mu]>0, rest 0
    Ys = Y[order]                                 # (pop, dim) sorted, = (x-m)/sigma
    yw = w @ Ys                                   # (dim,)
    m_new = state["m"] + state["sigma"] * yw
    cs, cc, c1, cmu = hp["cs"], hp["cc"], hp["c1"], hp["cmu"]
    ps = (1 - cs) * state["ps"] + jnp.sqrt(cs * (2 - cs) * hp["mueff"]) * (yw / sqc)
    gen1 = state["gen"] + 1
    hsig = (jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - cs) ** (2 * gen1))
            < (1.4 + 2.0 / (state["m"].shape[0] + 1)) * hp["chin"]).astype(jnp.float32)
    pc = (1 - cc) * state["pc"] + hsig * jnp.sqrt(cc * (2 - cc) * hp["mueff"]) * yw
    rankmu = w @ (Ys ** 2)                        # (dim,)
    c_new = ((1 - c1 - cmu) * state["c"]
             + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * state["c"])
             + cmu * rankmu)
    sigma = state["sigma"] * jnp.exp((cs / hp["damps"]) * (jnp.linalg.norm(ps) / hp["chin"] - 1))
    sigma = jnp.clip(sigma, 1e-8, 1e3)
    fbest_i = F[order[0]]
    improve = fbest_i < state["best_f"]
    best_x = jnp.where(improve, X[order[0]], state["best_x"])
    best_f = jnp.where(improve, fbest_i, state["best_f"])
    return dict(m=m_new, sigma=sigma, c=c_new, ps=ps, pc=pc,
                best_x=best_x, best_f=best_f, gen=gen1, key=key)


@partial(jax.jit, static_argnums=(1, 4, 5))
def run(x0, fit, sigma0, hp, popsize, maxiter, key, cargs):
    """Full sep-CMA-ES solve inside one lax.scan (no host round-trips).

    ``fit`` is a static (hashable) callable (X(pop,dim), cargs)->(pop,). ``cargs`` is a
    traced pytree carrying the cost's large arrays (+ per-instance dep_off). Returns
    (best_x, best_f). vmap this over (seeds x departures): map x0/key/cargs, keep fit/
    sigma0/hp/popsize/maxiter shared.
    """
    dim = x0.shape[0]
    state = init_state(x0, sigma0, dim, key)

    def body(st, _):
        return _step(st, fit, hp, popsize, cargs), st["best_f"]

    final, _ = jax.lax.scan(body, state, None, length=maxiter)
    return final["best_x"], final["best_f"]


def make_solver(fit, x0, sigma0, hp, popsize, maxiter):
    """Batched solver: solve(keys(B,2), dep_offs(B,), shared) -> best_x(B,dim), best_f(B).

    vmaps the whole sep-CMA scan over B = (seeds x departures) instances in ONE dispatch.
    ``shared`` = the large grid arrays (fields, axes, land_arrs), passed UNMAPPED so they
    are not replicated B times; only the per-instance key and dep_off are mapped. ``fit``
    takes (X, cargs) with cargs=(dep_off, fields, axes, land_arrs).
    """
    x0 = jnp.asarray(x0, jnp.float32)

    def single(key, dep_off, shared):
        state = init_state(x0, sigma0, x0.shape[0], key)
        cargs = (dep_off,) + tuple(shared)

        def body(st, _):
            return _step(st, fit, hp, popsize, cargs), st["best_f"]

        final, _ = jax.lax.scan(body, state, None, length=maxiter)
        return final["best_x"], final["best_f"]

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None)))
