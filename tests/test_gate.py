import numpy as np
import pytest
import xarray as xr

from spires_postprocess.gate import PRODUCT_MIN_FSNOW, gate_products


def bool_plane(results, where=None):
    mask = xr.zeros_like(results["fsnow"], dtype=bool)
    if where is not None:
        mask.values[where] = True
    return mask


@pytest.fixture
def masks(results):
    land = ~bool_plane(results, where=(slice(None), slice(6, 8)))      # 2 water columns
    cloud = bool_plane(results, where=(0, slice(None)))                # row 0
    tree = bool_plane(results, where=(slice(3, 5), slice(0, 2)))       # 4 pixels
    saturated = bool_plane(results, where=(7, 0))                      # 1 pixel
    return land.rename("land_mask"), cloud.rename("cloud_mask"), tree.rename("tree_mask"), saturated.rename("shade_saturated_mask")


def test_gate_each_mask_nans_only_its_pixels(results, masks):
    land, cloud, tree, saturated = masks
    gated = gate_products(
        results, fsnow_variable="fsnow", land_mask=land, cloud_mask=cloud,
        tree_mask=tree, shade_saturated_mask=saturated,
    )
    fsnow = gated["fsnow_product"]
    assert fsnow.dtype == np.float32
    excluded = (~land | cloud | tree | saturated).values
    assert np.all(np.isnan(fsnow.values[excluded]))
    np.testing.assert_array_equal(fsnow.values[~excluded], results["fsnow"].values[~excluded])
    for name in ("cloud_mask", "land_mask", "tree_mask", "shade_saturated_mask"):
        assert gated[name].dtype == np.uint8
    assert int(gated["tree_mask"].sum()) == 4
    assert gated["fsnow_product"].attrs["source_variable"] == "fsnow"
    assert gated["fsnow_product"].attrs["gates"] == "cloud_mask,land_mask,tree_mask,shade_saturated_mask"


def test_gate_applies_min_fsnow_to_grain_and_lap(results, masks):
    land, cloud, _, _ = masks
    fsnow = results["fsnow"].copy()
    fsnow.values[5, 5] = 0.05
    fsnow.values[5, 4] = 0.10
    results = results.assign(fsnow=fsnow)
    gated = gate_products(results, fsnow_variable="fsnow", land_mask=land, cloud_mask=cloud)
    assert np.isnan(gated["grain_radius_product"][5, 5].item())
    assert np.isnan(gated["lap_concentration_product"][5, 5].item())
    assert gated["grain_radius_product"][5, 4].item() == results["grain_radius"][5, 4].item()
    assert gated["fsnow_product"][5, 5].item() == pytest.approx(0.05)  # fsnow itself is not gated
    assert np.isnan(gated["grain_radius_product"][0, 3].item())  # cloud row
    assert gated["grain_radius_product"].attrs["min_fsnow"] == PRODUCT_MIN_FSNOW == 0.10
    assert gated["lap_concentration_product"].attrs["lap_type"] == "dust"
    assert gated["lap_concentration_product"].attrs["units"] == "ppm"
    assert "tree_mask" not in gated


def test_gate_reads_named_variables(results, masks):
    land, cloud, _, _ = masks
    results = results.assign(
        fsnow_treefilled=results["fsnow"] * np.float32(0.5),
        grain_radius_treefilled=results["grain_radius"] * np.float32(2.0),
        lap_concentration_treefilled=results["lap_concentration"] * np.float32(3.0),
    )
    gated = gate_products(
        results, fsnow_variable="fsnow_treefilled", grain_variable="grain_radius_treefilled",
        lap_variable="lap_concentration_treefilled", land_mask=land, cloud_mask=cloud,
    )
    assert gated["grain_radius_product"].attrs["source_variable"] == "grain_radius_treefilled"
    assert gated["grain_radius_product"][5, 5].item() == results["grain_radius"][5, 5].item() * 2


def test_gate_does_not_modify_results(results, masks):
    land, cloud, tree, sat = masks
    before = results.copy(deep=True)
    gate_products(results, fsnow_variable="fsnow", land_mask=land, cloud_mask=cloud, tree_mask=tree, shade_saturated_mask=sat)
    assert results.identical(before)


def test_gate_rejects_bad_masks(results, masks):
    land, cloud, _, _ = masks
    with pytest.raises(ValueError, match="boolean"):
        gate_products(results, fsnow_variable="fsnow", land_mask=land.astype("uint8"), cloud_mask=cloud)
    with pytest.raises(ValueError, match="does not match"):
        gate_products(results, fsnow_variable="fsnow", land_mask=land, cloud_mask=cloud.assign_coords(x=cloud["x"] + 1))
    with pytest.raises(ValueError, match="missing"):
        gate_products(results, fsnow_variable="absent", land_mask=land, cloud_mask=cloud)
    with pytest.raises(TypeError):
        gate_products(results, fsnow_variable="fsnow", land_mask=land)  # cloud_mask required
