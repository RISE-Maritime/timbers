"""Land and domain handling.

Properties under test:
- land-masked weather cells carry sea values, so land never reads as calm;
- the exclusion raster rises with distance from water, with no flat interior;
- land is a hard constraint in the cost and costs nothing to a route at sea.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from toy_power import toy_power_jax  # noqa: E402

from timbers import optimizer as op  # noqa: E402
from timbers.weather import load_era5  # noqa: E402
from timbers.land import exclusion_raster  # noqa: E402
from timbers.model import Grids  # noqa: E402
from timbers.weather import fill_from_nearest_sea  # noqa: E402


def _raster(mask, lat, wlon):
    return {"lat": lat, "wlon": wlon, "mask": mask.astype(np.float32)}


# --- exclusion raster ---------------------------------------------------------
def test_ramp_is_zero_at_sea_and_rises_inland():
    """Square island on a 0.1 degree grid: the waterline cell is one step from
    water (1 + 0.1), the deepest cell three steps (1 + 0.3)."""
    lat = np.arange(40.0, 45.01, 0.1)
    wlon = np.arange(-10.0, -4.99, 0.1)
    mask = np.zeros((lat.size, wlon.size))
    mask[20:26, 3:9] = 1.0
    m = exclusion_raster(_raster(mask, lat, wlon))["mask"]
    assert np.all(m[mask == 0] == 0.0)
    assert m[20, 3] == pytest.approx(1.1)
    assert m[22, 5] == pytest.approx(1.3)
    assert np.all(np.diff(m[22, 3:6]) > 0)  # no flat interior


def test_ramp_zero_gives_the_binary_union():
    lat = np.arange(40.0, 42.01, 0.1)
    wlon = np.arange(-10.0, -7.99, 0.1)
    mask = np.zeros((lat.size, wlon.size))
    mask[5:8, 5:8] = 1.0
    out = exclusion_raster(_raster(mask, lat, wlon), ramp=0.0)
    assert set(np.unique(out["mask"])) == {0.0, 1.0}


def test_domain_is_excluded_and_ramped():
    lat = np.arange(30.0, 40.01, 0.1)
    wlon = np.arange(-10.0, -7.99, 0.1)
    mask = np.zeros((lat.size, wlon.size))
    m = exclusion_raster(_raster(mask, lat, wlon), domain=(35.0, 40.0, -10.0, -8.0))["mask"]
    i35, i34, i33 = (np.abs(lat - x).argmin() for x in (35.0, 34.0, 33.0))
    assert m[i35, 5] == 0.0
    assert 1.0 < m[i34, 5] < m[i33, 5]


def test_extra_exclusion_and_shape_check():
    lat = np.arange(40.0, 42.01, 0.1)
    wlon = np.arange(-10.0, -7.99, 0.1)
    mask = np.zeros((lat.size, wlon.size))
    shallow = np.zeros(mask.shape, bool)
    shallow[10, 10] = True
    m = exclusion_raster(_raster(mask, lat, wlon), extra=shallow)["mask"]
    assert m[10, 10] > 0.0 and m[10, 12] == 0.0
    with pytest.raises(ValueError):
        exclusion_raster(_raster(mask, lat, wlon), extra=shallow[:-1])


# --- nearest-sea fill -----------------------------------------------------------
def test_fill_carries_the_nearest_sea_value():
    a = np.arange(1.0, 10.0).reshape(1, 3, 3)
    masked = np.zeros((3, 3), bool)
    masked[1, 1] = True
    out = fill_from_nearest_sea(a, masked)
    assert out[0, 1, 1] in (2.0, 4.0, 6.0, 8.0)
    assert np.array_equal(out[0][~masked], a[0][~masked])


def test_fill_only_where_masked_at_that_time():
    a = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    masked = np.array([[[False, True]], [[False, False]]])
    out = fill_from_nearest_sea(a, masked)
    assert out[0, 0, 1] == 1.0 and out[1, 0, 1] == 4.0


def test_fill_no_op_and_all_masked():
    a = np.arange(12.0).reshape(2, 2, 3)
    assert fill_from_nearest_sea(a, np.zeros((2, 3), bool)) is a
    with pytest.raises(ValueError):
        fill_from_nearest_sea(a, np.ones((2, 3), bool))


def _write_wave_file(path, nan_land=False):
    ds = nc.Dataset(path, "w")
    ds.createDimension("valid_time", 2)
    ds.createDimension("latitude", 3)
    ds.createDimension("longitude", 3)
    t = ds.createVariable("valid_time", "f8", ("valid_time",))
    t.units = "hours since 2024-01-01 00:00:00"
    t[:] = [0.0, 1.0]
    ds.createVariable("latitude", "f8", ("latitude",))[:] = [45.0, 44.5, 44.0]
    ds.createVariable("longitude", "f8", ("longitude",))[:] = [0.0, 0.5, 1.0]
    dims = ("valid_time", "latitude", "longitude")
    if nan_land:  # no _FillValue; land stored as NaN
        v = ds.createVariable("swh", "f4", dims, fill_value=False)
        data = np.full((2, 3, 3), 2.0, np.float32)
        data[:, 1, 1] = np.nan
    else:
        v = ds.createVariable("swh", "f4", dims, fill_value=-9999.0)
        data = np.ma.masked_array(np.full((2, 3, 3), 2.0, np.float32))
        data[:, 1, 1] = np.ma.masked  # one land cell
    v[:] = data
    ds.close()


@pytest.mark.parametrize("nan_land", [False, True])
def test_load_era5_does_not_make_land_calm(tmp_path, nan_land):
    f = tmp_path / "waves.nc"
    _write_wave_file(f, nan_land)
    near = load_era5(str(f))
    zero = load_era5(str(f), land_fill="zero")
    iy = int(np.argmin(np.abs(near["lat"] - 44.5)))
    ix = int(np.argmin(np.abs(near["lon"] - 0.5)))
    assert np.all(near["swh"][:, iy, ix] == 2.0)  # carried from the sea
    assert np.all(zero["swh"][:, iy, ix] == 0.0)  # zero fill


# --- hard land in the cost ------------------------------------------------------
COR = op.Corridor("example", 43.6, -4.0, 40.6, -69.0, 48.0)
K, L, NSP = 6, 40, 4


def _grids():
    lat = np.arange(35.0, 50.001, 0.5)
    lon = np.arange(-75.0, 5.001, 0.5)
    nt = 73
    t = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(nt) * np.timedelta64(1, "h")
    shape = (nt, lat.size, lon.size)
    base = dict(lat=lat, lon=lon, times=t, t0=t[0], dt_h=1.0)
    wind = {**base, "u10": np.full(shape, 5.0, np.float32), "v10": np.full(shape, -4.0, np.float32)}
    wave = {
        **base,
        "swh": np.full(shape, 1.5, np.float32),
        "mwd": np.full(shape, 200.0, np.float32),
    }
    return Grids.from_era5(wind, wave)


def _costs(raster, theta):
    fit, shared = op.build_fit(
        COR, _grids(), op.DeviceLand(raster), op.Penalty(), K, NSP, L, 0.0, False, toy_power_jax
    )
    theta = jnp.asarray(np.atleast_2d(theta), jnp.float32)
    return np.asarray(fit(theta, (jnp.float32(0.0), *shared)))


def _cost(land, theta):
    return float(_costs(land, theta)[0])


def test_land_is_hard_and_free_at_sea():
    """The great-circle route through an island costs far more than with no
    island, and a land-free raster leaves the cost exactly as without land."""
    llat = np.arange(35.0, 50.001, 0.25)
    lwlon = np.arange(-75.0, 5.001, 0.25)
    sea = {"lat": llat, "wlon": lwlon, "mask": np.zeros((llat.size, lwlon.size), np.float32)}
    island = np.zeros((llat.size, lwlon.size))
    lat_gc, wlon_gc, _ = op.decode_route(op.gc_init_theta(COR, K, NSP), COR, K, L, NSP)
    mid = L // 2
    iy = np.abs(llat - lat_gc[mid]).argmin()
    ix = np.abs(lwlon - wlon_gc[mid]).argmin()
    island[iy - 4 : iy + 5, ix - 4 : ix + 5] = 1.0
    land = exclusion_raster({"lat": llat, "wlon": lwlon, "mask": island})
    theta = op.gc_init_theta(COR, K, NSP)
    c_sea, c_land = _cost(sea, theta), _cost(land, theta)
    assert c_land - c_sea > 1e5
    assert _cost(exclusion_raster(sea), theta) == pytest.approx(c_sea)


def test_port_on_land_keeps_cost_resolution():
    """A port inside the raster adds nothing: candidates that differ only in
    speed profile keep the costs, and so the ranking, they have without land."""
    llat = np.arange(35.0, 50.001, 0.25)
    lwlon = np.arange(-75.0, 5.001, 0.25)
    coast = np.zeros((llat.size, lwlon.size))
    iy = np.abs(llat - COR.o_lat).argmin()
    ix = np.abs(lwlon - COR.o_wlon).argmin()
    coast[iy - 2 : iy + 3, ix - 2 : ix + 3] = 1.0
    land = exclusion_raster({"lat": llat, "wlon": lwlon, "mask": coast})
    sea = {"lat": llat, "wlon": lwlon, "mask": np.zeros_like(coast, np.float32)}
    rng = np.random.default_rng(0)
    theta = np.tile(op.gc_init_theta(COR, K, NSP), (64, 1)).astype(np.float32)
    theta[:, -NSP:] = rng.normal(0.0, 0.02, (64, NSP))

    on_land, free = _costs(land, theta), _costs(sea, theta)
    assert len(np.unique(free)) == 64
    np.testing.assert_array_equal(on_land, free)
