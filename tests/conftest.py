"""Synthetic contract-valid fixtures shared by the postprocess chain tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

N = 8
# Geographic grid straddling lon -119.0 at Mammoth: 4 columns west (land in the
# test water tile), 4 columns east (water). Pixel centres, 0.001 degree spacing.
LON = -119.0035 + 0.001 * np.arange(N)
LAT = 37.6535 - 0.001 * np.arange(N)


def geographic_grid(name: str = "fsnow", values=None) -> xr.DataArray:
    import rioxarray  # noqa: F401  registers the .rio accessor

    data = np.zeros((N, N), dtype=np.float32) if values is None else values
    da = xr.DataArray(
        np.asarray(data, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": LAT, "x": LON},
        name=name,
    )
    return da.rio.write_crs("EPSG:4326")


@pytest.fixture
def grid() -> xr.DataArray:
    return geographic_grid()


@pytest.fixture
def results(grid) -> xr.Dataset:
    rng = np.random.default_rng(0)
    fsnow = grid.copy(data=rng.uniform(0.0, 0.8, (N, N)).astype(np.float32))
    fshade = grid.copy(data=rng.uniform(0.0, 0.3, (N, N)).astype(np.float32))
    grain = grid.copy(data=rng.uniform(100.0, 900.0, (N, N)).astype(np.float32))
    lap = grid.copy(data=rng.uniform(0.0, 500.0, (N, N)).astype(np.float32))
    fsnow.attrs = {"long_name": "Fractional snow endmember", "units": "1"}
    fshade.attrs = {"long_name": "Fractional shade endmember", "units": "1"}
    grain.attrs = {"long_name": "Effective Snow Grain Radius", "units": "um"}
    lap.attrs = {
        "long_name": "Light-Absorbing Particle Concentration in Snow",
        "units": "ppm",
        "lap_type": "dust",
    }
    return xr.Dataset(
        {
            "fsnow": fsnow,
            "fshade": fshade,
            "lap_concentration": lap,
            "grain_radius": grain,
        }
    )


@pytest.fixture
def scene(grid) -> xr.Dataset:
    valid = np.ones((N, N), dtype=bool)
    valid[0, :] = False  # first row is cloud
    return xr.Dataset(
        {"valid": xr.DataArray(valid, dims=("y", "x"), coords=grid.coords)},
        coords=grid.coords,
    )


@pytest.fixture
def water_tile_dir(tmp_path):
    """A tiny EPSG:4326 land/water tile named for the Mammoth 30-degree cell.

    20 x 20 pixels at 0.005 degree covering lon [-119.05, -118.95] and
    lat [37.60, 37.70]; land (1) where lon < -119.0, water (0) east of it.
    """
    import rioxarray  # noqa: F401

    lon = -119.05 + 0.005 * (np.arange(20) + 0.5)
    lat = 37.70 - 0.005 * (np.arange(20) + 0.5)
    land = (lon < -119.0).astype(np.uint8)
    data = np.broadcast_to(land[None, :], (20, 20)).copy()
    tile = xr.DataArray(data, dims=("y", "x"), coords={"y": lat, "x": lon})
    tile = tile.rio.write_crs("EPSG:4326")
    tile.rio.to_raster(tmp_path / "landWater2020_-120_030.tif")
    return tmp_path


def binary_canopy_raster(fill_columns: slice, striped_columns: slice | None = None) -> xr.DataArray:
    """EPSG:4326 binary canopy raster at 0.0005 degree covering the grid + buffer.

    Columns in ``fill_columns`` (in *target* column units, 0..7) are all tree;
    ``striped_columns`` alternate tree/no-tree row-wise so they average to 0.5.
    """
    import rioxarray  # noqa: F401

    n = 2 * N + 8  # 2x oversampling plus a 4-fine-pixel (0.002 degree) margin
    lon = (LON[0] - 0.0005 - 0.002) + 0.0005 * (np.arange(n) + 0.5)
    lat = (LAT[0] + 0.0005 + 0.002) - 0.0005 * (np.arange(n) + 0.5)
    data = np.zeros((n, n), dtype=np.uint8)
    for target_col in range(N)[fill_columns]:
        fine = 4 + 2 * target_col
        data[:, fine : fine + 2] = 1
    if striped_columns is not None:
        for target_col in range(N)[striped_columns]:
            fine = 4 + 2 * target_col
            data[0::2, fine : fine + 2] = 1
            data[1::2, fine : fine + 2] = 0
    raster = xr.DataArray(data, dims=("y", "x"), coords={"y": lat, "x": lon})
    return raster.rio.write_crs("EPSG:4326")
