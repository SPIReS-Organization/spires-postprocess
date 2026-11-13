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
