"""Ensemble cost and scoring on a synthetic ensemble.

Properties under test:
- time interpolation follows a non-uniform step vector;
- non-finite fields are rejected at construction;
- a single gridded field wraps as a one-member ensemble;
- the objectives reduce members as documented (member 0 against the mean;
  identical members make all four agree);
- the breach probability of a route counts the members that breach;
- the per-segment series integrates to the route energy;
- the cost drives the optimizer;
- power-model draws multiply the members draw-major, reduce as documented
  (mean, CVaR), travel as traced data, and can enter the safety margin.
"""

import sys
from datetime import datetime
from datetime import timedelta
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax, toy_power_np  # noqa: E402

from timbers import cmaes as jc  # noqa: E402
from timbers import ensemble as te  # noqa: E402
from timbers import model as jm  # noqa: E402
from timbers import optimizer as op  # noqa: E402
from timbers import scoring as sc  # noqa: E402
from timbers.model import Grids  # noqa: E402

COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)
K, L, NSP, ALIGN = 6, 40, 4, 0.25
LAT = np.arange(35.0, 50.001, 0.5)
LON = np.arange(-75.0, 5.001, 0.5)
STEPS = np.array([0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60], np.float32)
LIMITS = dict(hs_lim=7.0, tws_lim=20.0)


def _members(swh):
    """Calm, uniform members; ``swh`` is one value per member."""
    shape = (len(swh), STEPS.size, LAT.size, LON.size)
    wind = dict(
        lat=LAT, lon=LON, u10=np.full(shape, 4.0, np.float32), v10=np.full(shape, -2.0, np.float32)
    )
    wave = dict(
        lat=LAT,
        lon=LON,
        mwd=np.full(shape, 200.0, np.float32),
        swh=np.asarray(swh, np.float32)[:, None, None, None] * np.ones(shape, np.float32),
    )
    return wind, wave


def _land():
    llat = np.arange(35.0, 50.001, 1.0)
    lwlon = np.arange(-75.0, 5.001, 1.0)
    return op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )


def _fit(grids, objective, **kw):
    fit, shared = te.make_ensemble_cost(
        grids,
        _land(),
        COR,
        objective=objective,
        L=L,
        K=K,
        n_speed=NSP,
        align=ALIGN,
        wps=False,
        power_fn=toy_power_jax,
        **LIMITS,
        **kw,
    )
    theta = jnp.asarray(op.gc_init_theta(COR, K, NSP))[None, :]
    return float(fit(theta, (jnp.float32(0.0), *shared))[0])


def _gc_members(grids, **kw):
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    return te.score_members(
        grids, COR, lat, wlon, seg, wps=False, power_fn=toy_power_jax, align=ALIGN, **LIMITS, **kw
    )


def test_time_index_on_non_uniform_steps():
    ti, tf = jm.time_index(jnp.asarray([9.0, 15.0, 0.0]), jnp.asarray(STEPS))
    assert list(np.asarray(ti)) == [3, 4, 0]
    np.testing.assert_allclose(np.asarray(tf), [0.0, 0.5, 0.0], atol=1e-6)


def test_non_finite_fields_are_rejected():
    wind, wave = _members([1.0, 1.0])
    wave["swh"][0, 0, 3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Grids(wind, wave, STEPS)


def test_single_field_wraps_as_one_member():
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(100) * np.timedelta64(1, "h")
    shape = (100, LAT.size, LON.size)
    base = dict(lat=LAT, lon=LON, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.ones(shape, np.float32), "v10": np.ones(shape, np.float32)}
    wave = {**base, "swh": np.ones(shape, np.float32), "mwd": np.zeros(shape, np.float32)}
    g = Grids.from_era5(wind, wave, datetime(2024, 1, 1, 6), 48.0)
    assert g.n_members == 1
    np.testing.assert_allclose(np.asarray(g.steps)[:3], [0.0, 1.0, 2.0])
    assert g.nt == 51


def test_objectives_reduce_members_as_documented():
    """Calm weather, no penalties active: deterministic is member 0's energy and
    expected-value the member mean; identical members make all four agree."""
    g = Grids(*_members([0.5, 2.0, 3.0, 4.0]), STEPS)
    e = _gc_members(g)["energy_mwh"]
    assert _fit(g, "deterministic") == pytest.approx(e[0], rel=1e-5)
    assert _fit(g, "expected_value") == pytest.approx(e.mean(), rel=1e-5)
    same = Grids(*_members([2.0] * 4), STEPS)
    vals = [_fit(same, o) for o in te.OBJECTIVES]
    assert max(vals) == pytest.approx(min(vals), rel=1e-6)


def test_breach_probability_counts_members():
    g = Grids(*_members([1.0, 1.0, 9.0, 9.0]), STEPS)  # two members above 7 m
    m = _gc_members(g)
    assert (m["margin"] > 0).mean() == pytest.approx(0.5)
    # The chance constraint sees the two stormy members; the deterministic arm,
    # reading member 0, does not.
    assert _fit(g, "chance_constrained") > _fit(g, "deterministic") + 1.0


def test_series_integrates_to_route_energy():
    g = Grids(*_members([1.0, 2.0]), STEPS)
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seg))])
    s = te.member_series(g, t, lat, op.working_to_signed(wlon), wps=False, power_fn=toy_power_jax)
    assert s["valid"].all()
    e_series = (s["power_kw"] * s["seg_h"]).sum(axis=1) / 1000.0
    np.testing.assert_allclose(e_series, _gc_members(g)["energy_mwh"], rtol=0.05)


def test_cost_drives_the_optimizer():
    g = Grids(*_members([1.0, 3.0, 5.0, 8.0]), STEPS)
    fit, shared = te.make_ensemble_cost(
        g,
        _land(),
        COR,
        objective="joint",
        L=L,
        K=K,
        n_speed=NSP,
        align=ALIGN,
        wps=False,
        power_fn=toy_power_jax,
        **LIMITS,
    )
    x0 = jnp.asarray(op.gc_init_theta(COR, K, NSP))
    cargs = (jnp.float32(0.0), *shared)
    j0 = float(fit(x0[None, :], cargs)[0])
    hp = jc.hyperparams(x0.size, 16)
    _, best = jc.run(x0, fit, 0.1, hp, 16, 30, jax.random.PRNGKey(0), cargs)
    assert np.isfinite(float(best)) and float(best) <= j0 + 1e-3


def _hourly(n=100, lat=LAT, lon=LON, swh=None):
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(n) * np.timedelta64(1, "h")
    shape = (n, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 4.0, np.float32), "v10": np.full(shape, -2.0, np.float32)}
    swh = (
        np.ones(shape, np.float32)
        if swh is None
        else np.broadcast_to(swh, shape).astype(np.float32)
    )
    wave = {**base, "swh": swh, "mwd": np.full(shape, 200.0, np.float32)}
    return wind, wave


def test_single_field_must_cover_the_window_on_its_steps():
    wind, wave = _hourly(60)
    with pytest.raises(ValueError, match="ends before"):
        Grids.from_era5(wind, wave, datetime(2024, 1, 1, 6), 60.0)
    with pytest.raises(ValueError, match="not on a time step"):
        Grids.from_era5(wind, wave, datetime(2024, 1, 1, 6, 20), 24.0)


def test_antimeridian_midpoints_sample_the_weather_at_180():
    """A Pacific route across 180 on a 0-360 grid, inside a calm band, gives the
    same cost, member outcomes and series as an all-calm world."""
    lat = np.arange(30.0, 40.001, 0.5)
    lon = np.arange(140.0, 230.001, 0.5)
    band = np.where((lon >= 150.0) & (lon <= 210.0), 1.0, 8.0)
    calm = Grids.from_era5(*_hourly(lat=lat, lon=lon), datetime(2024, 1, 1), 60.0)
    banded = Grids.from_era5(*_hourly(lat=lat, lon=lon, swh=band), datetime(2024, 1, 1), 60.0)
    cor = op.Corridor("pacific", 35.0, 170.0, 35.0, 200.0, 48.0)
    llat, lwlon = np.arange(30.0, 40.001, 1.0), np.arange(140.0, 230.001, 1.0)
    land = op.DeviceLand(
        {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    )
    theta = jnp.asarray(op.gc_init_theta(cor, K, NSP))[None, :]
    lat_r, wlon_r, seg = op.decode_route(np.asarray(theta[0]), cor, K, L, NSP)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seg))])
    for objective in ("deterministic", "chance_constrained"):
        costs = []
        for g in (calm, banded):
            fit, shared = te.make_ensemble_cost(
                g,
                land,
                cor,
                objective=objective,
                L=L,
                K=K,
                n_speed=NSP,
                align=ALIGN,
                wps=False,
                power_fn=toy_power_jax,
                **LIMITS,
            )
            costs.append(float(fit(theta, (jnp.float32(0.0), *shared))[0]))
        assert costs[1] == pytest.approx(costs[0], rel=1e-6)
    m = te.score_members(
        banded, cor, lat_r, wlon_r, seg, wps=False, power_fn=toy_power_jax, align=ALIGN, **LIMITS
    )
    assert m["max_hs"][0] == pytest.approx(1.0)
    s = te.member_series(
        banded, t, lat_r, op.working_to_signed(wlon_r), wps=False, power_fn=toy_power_jax
    )
    assert np.max(s["swh"][:, s["valid"]]) == pytest.approx(1.0)


def test_port_on_land_keeps_cost_resolution():
    """A port inside the land raster adds nothing to the cost."""
    g = Grids(*_members([1.0, 2.0]), STEPS)
    llat, lwlon = np.arange(35.0, 50.001, 0.25), np.arange(-75.0, 5.001, 0.25)
    coast = np.zeros((llat.size, lwlon.size), np.float32)
    iy, ix = np.abs(llat - COR.o_lat).argmin(), np.abs(lwlon - COR.o_wlon).argmin()
    coast[iy, ix] = 1.0  # only the port's cell: resampled points 0.3 deg out stay clear
    rng = np.random.default_rng(0)
    theta = np.tile(op.gc_init_theta(COR, K, NSP), (64, 1)).astype(np.float32)
    theta[:, -NSP:] = rng.normal(0.0, 0.02, (64, NSP))
    out = []
    for mask in (np.zeros_like(coast), coast):
        land = op.DeviceLand({"lat": llat, "wlon": lwlon, "mask": mask})
        fit, shared = te.make_ensemble_cost(
            g,
            land,
            COR,
            objective="joint",
            L=L,
            K=K,
            n_speed=NSP,
            align=ALIGN,
            wps=False,
            power_fn=toy_power_jax,
            **LIMITS,
        )
        out.append(np.asarray(fit(jnp.asarray(theta), (jnp.float32(0.0), *shared))))
    assert len(np.unique(out[0])) == 64
    np.testing.assert_array_equal(out[1], out[0])


def test_perturbations_multiply_members_perturbation_major():
    """With perturbations, members are (perturbation, member) pairs in
    perturbation-major order, and the unperturbed row reproduces the members."""
    g = Grids(*_members([1.0, 2.0, 3.0]), STEPS)
    plain = _gc_members(g)
    perts = te.perturbation_grid(hs=(1.0, 2.0))
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    m = te.score_members(
        g,
        COR,
        lat,
        wlon,
        seg,
        wps=False,
        power_fn=toy_power_jax,
        align=ALIGN,
        perturbations=perts,
        **LIMITS,
    )
    np.testing.assert_allclose(m["max_hs"], [1, 2, 3, 2, 4, 6], rtol=1e-6)
    np.testing.assert_allclose(m["energy_mwh"][:3], plain["energy_mwh"], rtol=1e-6)


def test_mean_safety_is_the_mean_member_penalty():
    """safety_mode="mean" averages the members' soft penalties: with identical
    members it equals member 0's, i.e. the deterministic objective, and with one
    stormy member out of four it is a quarter of that member's penalty."""
    same = Grids(*_members([6.9] * 4), STEPS)
    det = _fit(same, "deterministic")
    assert det > _fit(Grids(*_members([1.0] * 4), STEPS), "deterministic")  # penalty active
    assert _fit(same, "chance_constrained", safety_mode="mean") == pytest.approx(det, rel=1e-6)

    # chance_constrained takes energy from member 0 (calm here), so the mixed cost
    # exceeds the all-calm one by the storm member's penalty alone, over 4.
    storm_g, calm_g = Grids(*_members([6.9]), STEPS), Grids(*_members([1.0]), STEPS)
    storm, calm = _fit(storm_g, "deterministic"), _fit(calm_g, "deterministic")
    extra_energy = _gc_members(storm_g)["energy_mwh"][0] - _gc_members(calm_g)["energy_mwh"][0]
    storm_penalty = storm - calm - float(extra_energy)
    mixed = _fit(
        Grids(*_members([1.0, 1.0, 1.0, 6.9]), STEPS), "chance_constrained", safety_mode="mean"
    )
    assert mixed - calm == pytest.approx(storm_penalty / 4, rel=1e-3)


# --- power-model uncertainty --------------------------------------------------
def test_model_draws_multiply_members_draw_major():
    """Members are (draw, weather member) pairs, draw-major; draw 0 at the
    nominal parameters reproduces the members, and a draw without the wave term
    makes every weather member cost the same (they differ only in Hs)."""
    g = Grids(*_members([1.0, 2.0, 3.0]), STEPS)
    plain = _gc_members(g)
    m = _gc_members(g, model_params=te.model_param_grid(wave=(1.0, 0.0)))
    np.testing.assert_allclose(m["max_hs"], [1, 2, 3, 1, 2, 3], rtol=1e-6)
    np.testing.assert_allclose(m["energy_mwh"][:3], plain["energy_mwh"], rtol=1e-6)
    assert np.ptp(m["energy_mwh"][3:]) < 1e-6 * m["energy_mwh"][3]
    assert m["energy_mwh"][3] < m["energy_mwh"][0]

    perts = te.perturbation_grid(hs=(1.0, 2.0))
    both = _gc_members(g, perturbations=perts, model_params=te.model_param_grid(wave=(1.0, 0.5)))
    np.testing.assert_allclose(both["max_hs"], [1, 2, 3, 2, 4, 6] * 2, rtol=1e-6)


def test_cost_reduces_draws_by_mean_or_cvar():
    g = Grids(*_members([2.0]), STEPS)
    draws = te.model_param_grid(calm=(1.0, 1.1, 1.3))
    e = _gc_members(g, model_params=draws)["energy_mwh"]
    assert _fit(g, "expected_value", model_params=draws) == pytest.approx(e.mean(), rel=1e-5)
    worst = _fit(g, "expected_value", model_params=draws, cost_mode="cvar", cost_eps=1 / 3)
    assert worst == pytest.approx(e.max(), rel=1e-5)
    assert _fit(g, "deterministic", model_params=draws) == pytest.approx(e[0], rel=1e-5)


def test_expected_cost_ignores_spread_linear_in_the_parameters():
    """Power linear in a parameter: the expected cost over draws symmetric about
    the nominal is the nominal cost for every route, so it cannot move the
    optimum. A nonlinear parameter (the speed exponent) does change it."""
    g = Grids(*_members([2.0]), STEPS)
    rng = np.random.default_rng(1)
    theta = np.tile(op.gc_init_theta(COR, K, NSP), (8, 1)).astype(np.float32)
    theta[:, -NSP:] = rng.normal(0.0, 0.3, (8, NSP))

    def costs(**kw):
        fit, shared = te.make_ensemble_cost(
            g,
            _land(),
            COR,
            L=L,
            K=K,
            n_speed=NSP,
            align=ALIGN,
            wps=False,
            power_fn=toy_power_jax,
            **LIMITS,
            **kw,
        )
        return np.asarray(fit(jnp.asarray(theta), (jnp.float32(0.0), *shared)))

    nominal = costs(objective="deterministic")
    linear = te.model_param_grid(calm=(1.0, 0.9, 1.1), wave=(1.0, 0.7, 1.3))
    np.testing.assert_allclose(
        costs(objective="expected_value", model_params=linear), nominal, rtol=1e-5
    )
    curved = te.model_param_grid(n=(3.0, 2.7, 3.3))
    assert (
        np.max(np.abs(costs(objective="expected_value", model_params=curved) / nominal - 1)) > 1e-3
    )


def test_model_params_travel_as_traced_data():
    """One compiled cost serves any draws of the same shape: swapping them in
    ``shared`` gives the cost a cost built with them gives."""
    g = Grids(*_members([2.0]), STEPS)
    kw = dict(L=L, K=K, n_speed=NSP, align=ALIGN, wps=False, power_fn=toy_power_jax, **LIMITS)
    a, b = te.model_param_grid(calm=(1.0, 1.2)), te.model_param_grid(calm=(1.5, 2.0))
    fit, shared = te.make_ensemble_cost(
        g, _land(), COR, objective="expected_value", model_params=a, **kw
    )
    theta = jnp.asarray(op.gc_init_theta(COR, K, NSP))[None, :]
    swapped = float(fit(theta, (jnp.float32(0.0), *shared[:-1], (te._as_params(b), None)))[0])
    assert swapped == pytest.approx(_fit(g, "expected_value", model_params=b), rel=1e-6)
    assert swapped > float(fit(theta, (jnp.float32(0.0), *shared))[0])


def test_ceiling_costs_and_cost_terms_match_main():
    """``deterministic`` and ``expected_value`` cost what they did before power
    uncertainty was added (values from main at 18c1b1c), with and without a
    power ceiling; ``chance_constrained`` and ``joint`` match them without a
    ceiling, and with one pay the same nominal soft power penalty (main
    ignored the ceiling there). Seakeeping is within eps here, so the chance
    term is zero."""
    g = Grids(*_members([1.0, 4.0, 6.8]), STEPS)
    rng = np.random.default_rng(3)
    theta = np.tile(op.gc_init_theta(COR, K, NSP), (4, 1)).astype(np.float32)
    theta[:, -NSP:] = rng.normal(0.0, 0.08, (4, NSP))
    main = {
        ("deterministic", False): [7374.592285, 7289.885254, 7440.042480, 7242.279297],
        ("deterministic", True): [26636.042969, 14040.932617, 28386.826172, 9165.552734],
        ("expected_value", False): [7662.017578, 7577.312012, 7727.467773, 7529.706055],
        ("expected_value", True): [26923.466797, 14328.360352, 28674.250000, 9452.979492],
    }
    same_as = {"chance_constrained": "deterministic", "joint": "expected_value"}

    def costs(objective, p_lim):
        fit, shared = te.make_ensemble_cost(
            g,
            _land(),
            COR,
            objective=objective,
            L=L,
            K=K,
            n_speed=NSP,
            align=ALIGN,
            wps=False,
            power_fn=toy_power_jax,
            p_lim=p_lim,
            **LIMITS,
        )
        return np.asarray(fit(jnp.asarray(theta), (jnp.float32(0.0), *shared)))

    for ceiling in (False, True):
        p_lim = 160_000.0 if ceiling else float("inf")
        for objective in te.OBJECTIVES:
            ref = main[(same_as.get(objective, objective), ceiling)]
            np.testing.assert_allclose(costs(objective, p_lim), ref, rtol=2e-6)


def test_power_chance_constraint_is_opt_in():
    """With ``power_eps``, a draw that needs more than ``p_lim`` counts as a
    breach of a power constraint, though the weather is calm; without it only
    the nominal soft penalty applies. A breach of the ceiling does not excuse a
    breach of the seakeeping limits."""
    g = Grids(*_members([1.0]), STEPS)
    draws = te.model_param_grid(calm=(1.0, 1.0, 1.6))
    p_max = float(_gc_members(g, model_params=draws)["max_power"][0])
    m = _gc_members(g, model_params=draws, p_lim=1.2 * p_max)
    assert list(m["power_margin"] > 0) == [False, False, True]
    none = _fit(g, "chance_constrained", model_params=draws, eps=0.1)
    kw = dict(model_params=draws, p_lim=1.2 * p_max, eps=0.1)
    # Draw 0 stays below the soft knee: without power_eps the ceiling is free.
    assert _fit(g, "chance_constrained", **kw) == pytest.approx(none, rel=1e-6)
    for mode in ("prob", "cvar"):
        on = _fit(g, "chance_constrained", safety_mode=mode, power_eps=0.1, **kw)
        assert on > none + 1.0
    # One breaching draw in three is within power_eps = 0.5.
    loose = _fit(g, "chance_constrained", power_eps=0.5, **kw)
    assert loose == pytest.approx(none, rel=1e-6)

    # Every draw over the ceiling: a storm on top must still cost more.
    kw = dict(model_params=draws, p_lim=0.5 * p_max, power_eps=0.1)
    calm = _fit(g, "chance_constrained", **kw)
    storm = _fit(Grids(*_members([9.0]), STEPS), "chance_constrained", **kw)
    e = [_gc_members(Grids(*_members([h]), STEPS))["energy_mwh"][0] for h in (1.0, 9.0)]
    assert storm - calm > (e[1] - e[0]) + 1.0


def test_draw_weights_enter_every_reduction():
    """``model_weights`` weights the mean, the CVaR and the breach probability;
    one-hot weights reduce the ensemble to that draw."""
    g = Grids(*_members([2.0]), STEPS)
    draws = te.model_param_grid(calm=(1.0, 1.2, 1.5))
    e = _gc_members(g, model_params=draws)["energy_mwh"]
    w = np.array([0.5, 0.3, 0.2])
    mean = _fit(g, "expected_value", model_params=draws, model_weights=w)
    assert mean == pytest.approx(np.average(e, weights=w), rel=1e-5)
    one_hot = _fit(g, "expected_value", model_params=draws, model_weights=[0, 1, 0])
    assert one_hot == pytest.approx(e[1], rel=1e-5)
    # Weighted CVaR counts the worst members up to the mass eps, the last in part:
    # with equal weights and eps = 0.5, all of the worst and half of the next.
    cvar = _fit(
        g,
        "expected_value",
        model_params=draws,
        model_weights=np.ones(3),
        cost_mode="cvar",
        cost_eps=0.5,
    )
    assert cvar == pytest.approx((e[2] / 3 + e[1] / 6) / 0.5, rel=1e-5)

    m = _gc_members(g, model_params=draws, model_weights=w)
    np.testing.assert_allclose(m["weight"], w, rtol=1e-6)
    assert _gc_members(g)["weight"] == pytest.approx([1.0])

    # Breach probability: the one breaching draw carries weight 0.2 or 0.5.
    p_max = float(m["max_power"][0])
    kw = dict(model_params=draws, p_lim=1.3 * p_max, eps=0.1, power_eps=0.3)
    light = _fit(g, "chance_constrained", model_weights=[0.4, 0.4, 0.2], **kw)
    heavy = _fit(g, "chance_constrained", model_weights=[0.25, 0.25, 0.5], **kw)
    assert light == pytest.approx(e[0], rel=1e-5) and heavy > light + 1.0


def test_model_options_are_checked():
    g = Grids(*_members([1.0]), STEPS)
    with pytest.raises(ValueError, match="power_eps needs"):
        _fit(g, "joint", power_eps=0.1)
    with pytest.raises(ValueError, match="power_eps needs"):
        _fit(g, "expected_value", p_lim=1e5, power_eps=0.1)
    with pytest.raises(ValueError, match="needs an ensemble cost"):
        _fit(g, "deterministic", cost_mode="cvar")
    with pytest.raises(ValueError, match="model_weights needs"):
        _fit(g, "joint", model_weights=[1.0])
    with pytest.raises(ValueError, match="model_weights must"):
        _fit(g, "joint", model_params=te.model_param_grid(calm=(1.0, 2.0)), model_weights=[1.0])
    with pytest.raises(ValueError, match="leading"):
        _fit(g, "joint", model_params={"calm": np.ones(2), "wave": np.ones(3)})


def test_draws_score_on_the_host_as_on_the_device():
    """``model_draws`` hands the host scorer one draw at a time."""
    wind, wave = _hourly(swh=2.0)
    g = Grids.from_era5(wind, wave)
    draws = te.model_param_grid(calm=(1.0, 1.2), n=(3.0, 3.3))
    dev = _gc_members(g, model_params=draws)["energy_mwh"]
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    t = [datetime(2024, 1, 1) + timedelta(hours=float(h)) for h in np.r_[0, np.cumsum(seg)]]
    wps_ = list(zip(t, lat, op.working_to_signed(wlon)))
    host = [
        sc.evaluate_route(wind, wave, wps_, partial(toy_power_np, params=d))
        for d in te.model_draws(draws)
    ]
    assert len(host) == 4
    np.testing.assert_allclose(host, dev, rtol=0.02)


def test_series_carries_the_draws():
    g = Grids(*_members([1.0, 2.0]), STEPS)
    lat, wlon, seg = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seg))])
    s = te.member_series(
        g,
        t,
        lat,
        op.working_to_signed(wlon),
        wps=False,
        power_fn=toy_power_jax,
        model_params=te.model_param_grid(calm=(1.0, 2.0)),
    )
    assert s["power_kw"].shape == (4, len(seg))
    np.testing.assert_allclose(s["swh"][2:], s["swh"][:2])
    assert np.all(s["power_kw"][2:] > s["power_kw"][:2])
