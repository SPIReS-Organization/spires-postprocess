import numpy as np
import pytest
import xarray as xr

from spires_postprocess.masks import cloud_mask_from_valid, mask_to_uint8


def test_cloud_mask_is_inverse_of_valid(scene, grid):
    mask = cloud_mask_from_valid(scene["valid"], grid)
    assert mask.dtype == bool
    assert mask.dims == ("y", "x")
    assert mask.name == "cloud_mask"
    assert bool(mask.isel(y=0).all())
    assert not bool(mask.isel(y=slice(1, None)).any())
    assert mask.attrs["source"] == "spires_io.valid"


def test_cloud_mask_rejects_non_boolean(scene, grid):
    with pytest.raises(ValueError, match="boolean"):
        cloud_mask_from_valid(scene["valid"].astype("uint8"), grid)


def test_cloud_mask_rejects_coordinate_mismatch(scene, grid):
    shifted = scene["valid"].assign_coords(x=scene["x"] + 1.0)
    with pytest.raises(ValueError, match="does not match"):
        cloud_mask_from_valid(shifted, grid)


def test_mask_to_uint8_keeps_coords_and_attrs(grid):
    mask = xr.DataArray(
        np.eye(8, dtype=bool), dims=("y", "x"), coords=grid.coords, name="m",
        attrs={"long_name": "x"},
    )
    out = mask_to_uint8(mask)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 8
    assert out.attrs["long_name"] == "x"
    assert out.attrs["flag_values"] == "0,1"
    assert out.name == "m"


def test_mask_to_uint8_drops_non_dimension_coords(grid):
    mask = xr.DataArray(
        np.eye(8, dtype=bool), dims=("y", "x"),
        coords={"y": grid["y"].values, "x": grid["x"].values, "spatial_ref": 0},
        name="m",
    )
    assert "spatial_ref" in mask.coords
    out = mask_to_uint8(mask)
    assert "spatial_ref" not in out.coords
    np.testing.assert_array_equal(out["x"].values, grid["x"].values)


from tests.conftest import binary_canopy_raster
from spires_postprocess.masks import canopy_cover_fraction, tree_mask


def test_canopy_cover_fraction_averages_binary_raster(grid):
    raster = binary_canopy_raster(fill_columns=slice(0, 2), striped_columns=slice(2, 3))
    frac = canopy_cover_fraction(raster, grid)
    assert frac.dtype == np.float32
    assert frac.dims == ("y", "x")
    assert frac.name == "canopy_fraction"
    np.testing.assert_allclose(frac.isel(x=slice(0, 2)).values, 1.0, atol=1e-6)
    np.testing.assert_allclose(frac.isel(x=2).values, 0.5, atol=1e-6)
    np.testing.assert_allclose(frac.isel(x=slice(3, None)).values, 0.0, atol=1e-6)
    np.testing.assert_array_equal(frac["x"].values, grid["x"].values)
    np.testing.assert_array_equal(frac["y"].values, grid["y"].values)
    assert frac.attrs["units"] == "1"


def test_canopy_cover_fraction_accepts_percent_raster(grid):
    raster = binary_canopy_raster(fill_columns=slice(0, 8))
    percent = (raster.astype("uint8") * 40).astype("uint8")
    percent.values[0, 0] = 255
    percent = percent.rio.write_nodata(255)
    frac = canopy_cover_fraction(percent, grid, percent=True)
    assert float(np.nanmax(frac.values)) == pytest.approx(0.4, abs=1e-6)
    assert float(np.nanmin(frac.values)) == pytest.approx(0.4, abs=1e-6)
    assert frac.attrs["source_units"] == "percent"


def test_canopy_cover_fraction_rejects_nonbinary_without_percent(grid):
    raster = binary_canopy_raster(fill_columns=slice(0, 8))
    forty = raster.copy(data=(raster.values * 40).astype("uint8"))
    with pytest.raises(ValueError, match="binary"):
        canopy_cover_fraction(forty, grid)


def test_canopy_cover_fraction_requires_overlap(grid):
    raster = binary_canopy_raster(fill_columns=slice(0, 8))
    far = raster.assign_coords(x=raster["x"] + 10.0).rio.write_crs("EPSG:4326")
    with pytest.raises(ValueError, match="overlap"):
        canopy_cover_fraction(far, grid)


def test_canopy_cover_fraction_requires_crs(grid):
    raster = binary_canopy_raster(fill_columns=slice(0, 8))
    bare = xr.DataArray(grid.values, dims=grid.dims, coords={"y": grid["y"], "x": grid["x"]})
    with pytest.raises(ValueError, match="CRS"):
        canopy_cover_fraction(raster, bare)


def test_tree_mask_threshold_and_dilation(grid):
    cover = grid.copy(data=np.zeros((8, 8), dtype=np.float32))
    cover.values[4, 4] = 0.5
    cover.values[0, 0] = 0.10  # exactly at threshold: not a tree
    mask = tree_mask(cover)
    assert mask.dtype == bool
    assert mask.name == "tree_mask"
    assert int(mask.sum()) == 9
    assert bool(mask.isel(y=slice(3, 6), x=slice(3, 6)).all())
    assert not bool(mask[0, 0])
    assert mask.attrs["cover_threshold"] == 0.10
    assert mask.attrs["dilate_cells"] == 1


def test_tree_mask_without_dilation_and_nan(grid):
    cover = grid.copy(data=np.full((8, 8), np.nan, dtype=np.float32))
    cover.values[2, 2] = 0.9
    mask = tree_mask(cover, threshold=0.5, dilate_cells=0)
    assert int(mask.sum()) == 1
    assert bool(mask[2, 2])


from spires_postprocess.masks import land_mask_from_tiles, land_water_tile_path


def test_land_water_tile_path_names_sw_corner(tmp_path):
    (tmp_path / "landWater2020_-120_030.tif").touch()
    path = land_water_tile_path((-119.1, 37.5, -118.9, 37.7), tmp_path)
    assert path.name == "landWater2020_-120_030.tif"


def test_land_water_tile_path_missing_tile_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="landWater2020_-120_030.tif"):
        land_water_tile_path((-119.1, 37.5, -118.9, 37.7), tmp_path)


def test_land_water_tile_path_spanning_tiles_raises(tmp_path):
    with pytest.raises(ValueError, match="span"):
        land_water_tile_path((-120.5, 37.5, -119.5, 37.7), tmp_path)


def test_land_mask_from_tiles_marks_land_true(grid, water_tile_dir):
    mask = land_mask_from_tiles(grid, water_tile_dir)
    assert mask.dtype == bool
    assert mask.name == "land_mask"
    assert bool(mask.isel(x=slice(0, 4)).all())
    assert not bool(mask.isel(x=slice(4, None)).any())
    np.testing.assert_array_equal(mask["x"].values, grid["x"].values)
    assert mask.attrs["tile"] == "landWater2020_-120_030.tif"


def test_land_mask_requires_crs(water_tile_dir, grid):
    bare = xr.DataArray(grid.values, dims=grid.dims, coords={"y": grid["y"], "x": grid["x"]})
    with pytest.raises(ValueError, match="CRS"):
        land_mask_from_tiles(bare, water_tile_dir)
