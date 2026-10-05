"""Routes across a longitude seam sample the weather on the route.

A segment midpoint averaged in the wrong longitude convention lands on the
far side of the globe: in signed longitude a segment across 180 averages to 0,
and on a 0-360 grid a segment across 0 averages to 180. Each case puts a calm
band around the crossing and heavy seas everywhere else, so any sample taken
off the route shows up as a higher maximum Hs, energy or cost than the same
route on uniformly calm weather.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax, toy_power_np  # noqa: E402

from timbers import ensemble as te  # noqa: E402
from timbers import model as jm  # noqa: E402
from timbers import optimizer as op  # noqa: E402
from timbers import risk as rk  # noqa: E402
from timbers.geo import midpoint_lon  # noqa: E402
from timbers.scoring import evaluate_route, evaluate_route_full  # noqa: E402

K, L, NSP, ALIGN = 6, 40, 4, 0.25
CALM, ROUGH = 1.0, 8.0
LAT = np.arange(30.0, 60.001, 0.5)
NT = 73
T = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(NT) * np.timedelta64(1, "h")
DEP = datetime(2024, 1, 1)

# (corridor, grid longitudes, longitude of the crossing)
CASES = {
    "greenwich-0..360": (
        op.Corridor("gw", 48.0, -10.0, 50.0, 10.0, 48.0),
        np.arange(0.0, 359.51, 0.5),
        0.0,
    ),
    "antimeridian-0..360": (
        op.Corridor("pac", 40.0, 170.0, 42.0, 200.0, 48.0),
        np.arange(0.0, 359.51, 0.5),
        180.0,
    ),
    "greenwich-signed": (
        op.Corridor("gw", 48.0, -10.0, 50.0, 10.0, 48.0),
        np.arange(-40.0, 40.01, 0.5),
        0.0,
    ),
}


def _grids(lon, crossing, calm_everywhere):
    """Wind and wave grids with a calm band within 30 degrees of the crossing."""
    near = np.abs((lon - crossing + 180.0) % 360.0 - 180.0) < 30.0
    hs = np.where(near | calm_everywhere, CALM, ROUGH).astype(np.float32)
    shape = (NT, LAT.size, lon.size)
    base = dict(lat=LAT, lon=lon, times=T, t0=T[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 3.0, np.float32), "v10": np.zeros(shape, np.float32)}
    wave = {
        **base,
        "swh": hs[None, None, :] * np.ones(shape, np.float32),
        "mwd": np.zeros(shape, np.float32),
    }
    return wind, wave


def _land():
    wlon = np.arange(-180.0, 360.01, 1.0)
    return op.DeviceLand(
        {"lat": LAT, "wlon": wlon, "mask": np.zeros((LAT.size, wlon.size), np.float32)}
    )


@pytest.fixture(params=list(CASES), scope="module")
def case(request):
    cor, lon, crossing = CASES[request.param]
    lat, wlon, seg = op.decode_route(op.gc_init_theta(cor, K, NSP), cor, K, L, NSP)
    return dict(
        cor=cor,
        lat=np.asarray(lat),
        wlon=np.asarray(wlon),
        seg=np.asarray(seg),
        theta=op.gc_init_theta(cor, K, NSP),
        seam=_grids(lon, crossing, False),
        calm=_grids(lon, crossing, True),
    )


def _waypoints(c):
    t = np.concatenate([[0.0], np.cumsum(c["seg"])])
    slon = op.working_to_signed(c["wlon"])
    return [(DEP + timedelta(hours=float(h)), a, b) for h, a, b in zip(t, c["lat"], slon)]


def test_midpoint_lon_takes_the_short_arc():
    a = np.array([179.9, -179.9, 359.9, 0.1, 10.0, -170.0])
    b = np.array([-179.9, 179.9, 0.1, 359.9, 20.0, 170.0])
    np.testing.assert_allclose(midpoint_lon(a, b, wrap=True), [180, 180, 0, 0, 15, 180], atol=1e-9)
    np.testing.assert_allclose(
        midpoint_lon(a, b, wrap=False), [-180, -180, 0, 0, 15, -180], atol=1e-9
    )
    j = midpoint_lon(jnp.asarray(a), jnp.asarray(b), wrap=True)  # JAX arrays too
    np.testing.assert_allclose(np.asarray(j), [180, 180, 0, 0, 15, 180], atol=1e-4)


def test_host_scorers(case):
    wind, wave = case["seam"]
    full = evaluate_route_full(wind, wave, _waypoints(case), toy_power_np)
    assert full["max_hs_m"] == pytest.approx(CALM)
    calm_e = evaluate_route(*case["calm"], _waypoints(case), toy_power_np)
    assert evaluate_route(wind, wave, _waypoints(case), toy_power_np) == pytest.approx(calm_e)


def test_device_route_energy(case):
    def energy(grids):
        g = jm.DeviceGrids(*grids)
        lons = jnp.asarray(op.working_to_signed(case["wlon"]), jnp.float32)
        return float(
            jm.route_energy(
                g,
                jnp.asarray(case["lat"], jnp.float32),
                lons,
                jnp.asarray(case["seg"], jnp.float32),
                0.0,
                False,
                toy_power_jax,
            )
        )

    assert energy(case["seam"]) == pytest.approx(energy(case["calm"]), rel=1e-5)


def test_optimizer_cost(case):
    def cost(grids):
        fn = op.make_batched_cost(
            jm.DeviceGrids(*grids),
            _land(),
            case["cor"],
            L,
            False,
            op.Penalty(),
            K,
            toy_power_jax,
            n_speed=NSP,
            align_dt_h=ALIGN,
        )
        return float(fn(jnp.asarray(case["theta"], jnp.float32)[None, :], 0.0)[0])

    assert cost(case["seam"]) == pytest.approx(cost(case["calm"]), rel=1e-5)


def test_risk_scorer_and_robust_cost(case):
    perts = rk.perturbation_grid()
    slon = jnp.asarray(op.working_to_signed(case["wlon"]), jnp.float32)
    scorer = rk.make_scorer(
        jm.DeviceGrids(*case["seam"]), case["cor"], False, toy_power_jax, align=ALIGN
    )
    _, max_hs, _ = scorer(
        jnp.asarray(case["lat"], jnp.float32),
        slon,
        jnp.asarray(case["seg"], jnp.float32),
        0.0,
        perts,
    )
    assert float(max_hs[0]) == pytest.approx(CALM)

    def cost(grids):
        fn = rk.make_robust_cost(
            jm.DeviceGrids(*grids),
            _land(),
            case["cor"],
            L,
            False,
            K,
            NSP,
            ALIGN,
            perts,
            toy_power_jax,
        )
        return float(fn(jnp.asarray(case["theta"], jnp.float32)[None, :], 0.0)[0])

    assert cost(case["seam"]) == pytest.approx(cost(case["calm"]), rel=1e-5)


def _ensemble(grids):
    wind, wave = grids
    return te.EnsembleGrids(
        {
            "lat": wind["lat"],
            "lon": wind["lon"],
            "u10": wind["u10"][None],
            "v10": wind["v10"][None],
        },
        {
            "lat": wave["lat"],
            "lon": wave["lon"],
            "swh": wave["swh"][None],
            "mwd": wave["mwd"][None],
        },
        np.arange(NT, dtype=np.float32),
    )


def test_ensemble(case):
    g = _ensemble(case["seam"])
    m = te.score_members(
        g,
        case["cor"],
        case["lat"],
        case["wlon"],
        case["seg"],
        wps=False,
        power_fn=toy_power_jax,
        align=ALIGN,
        hs_lim=7.0,
        tws_lim=20.0,
    )
    assert float(m["max_hs"][0]) == pytest.approx(CALM)
    t = np.concatenate([[0.0], np.cumsum(case["seg"])])
    s = te.member_series(
        g, t, case["lat"], op.working_to_signed(case["wlon"]), wps=False, power_fn=toy_power_jax
    )
    assert float(s["swh"].max()) == pytest.approx(CALM)

    def cost(grids):
        fit, shared = te.make_ensemble_cost(
            _ensemble(grids),
            _land(),
            case["cor"],
            objective="joint",
            L=L,
            K=K,
            n_speed=NSP,
            align=ALIGN,
            wps=False,
            power_fn=toy_power_jax,
            hs_lim=7.0,
            tws_lim=20.0,
        )
        theta = jnp.asarray(case["theta"], jnp.float32)[None, :]
        return float(fit(theta, (jnp.float32(0.0), *shared))[0])

    assert cost(case["seam"]) == pytest.approx(cost(case["calm"]), rel=1e-5)
