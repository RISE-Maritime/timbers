"""Data-free unit tests: parameterization invariants."""

import jax.numpy as jnp
import numpy as np

from timbers.optimizer import Corridor, bezier, gc_init_theta, time_alloc

# A generic example corridor (continuous working-longitude frame).
COR = Corridor("example", 43.6, -4.0, 40.6, -69.0, 354.0)


def test_time_alloc_sums_to_passage_time():
    rng = np.random.default_rng(0)
    for n_speed in (4, 36, 64):
        w = jnp.asarray(rng.standard_normal(n_speed), jnp.float32)
        seg = time_alloc(w, 354.0, 400, n_speed)
        assert seg.shape == (399,)
        assert float(jnp.min(seg)) > 0.0          # strictly positive durations
        np.testing.assert_allclose(float(jnp.sum(seg)), 354.0, rtol=1e-5)


def test_time_alloc_zero_weights_is_uniform():
    seg = time_alloc(jnp.zeros(36, jnp.float32), 354.0, 400, 36)
    np.testing.assert_allclose(np.asarray(seg), 354.0 / 399, rtol=1e-6)


def test_time_alloc_nspeed0_recovers_implicit_model():
    seg = time_alloc(jnp.zeros(0, jnp.float32), 583.0, 100, 0)
    np.testing.assert_allclose(np.asarray(seg), 583.0 / 99, rtol=1e-6)


def test_bezier_endpoints_fixed_at_ports():
    cor = COR
    theta = gc_init_theta(cor, K=10, n_speed=0)
    interior = theta[: 2 * 8].reshape(8, 2)
    o = np.array(cor.norm(cor.o_lat, cor.o_wlon))
    d = np.array(cor.norm(cor.d_lat, cor.d_wlon))
    ctrl = jnp.asarray(np.vstack([o, interior, d]), jnp.float32)
    pts = np.asarray(bezier(ctrl, jnp.linspace(0.0, 1.0, 50, dtype=jnp.float32)))
    np.testing.assert_allclose(pts[0], o, atol=1e-6)
    np.testing.assert_allclose(pts[-1], d, atol=1e-5)


def test_gc_init_theta_speed_weights_zero():
    cor = COR
    theta = gc_init_theta(cor, K=16, n_speed=36)
    assert theta.shape == (2 * 14 + 36,)
    np.testing.assert_array_equal(theta[-36:], 0.0)
