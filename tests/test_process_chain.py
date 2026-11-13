import numpy as np
import pytest
import xarray as xr
from spires_contract import SpiresData, validate_results

from spires_postprocess import process
from spires_postprocess.masks import cloud_mask_from_valid
from spires_postprocess.process import _require_unchanged_base_results


@pytest.fixture
def data(results, scene, grid):
    canopy = grid.copy(data=np.zeros((8, 8), dtype=np.float32))
    canopy.values[3:5, 1:3] = 0.6
    canopy.attrs = {"long_name": "Canopy cover fraction", "units": "1"}
    return SpiresData(scene=scene, ancillary=xr.Dataset({"canopy_fraction": canopy}), results=results)


def constant_inpainter(field, mask, vmin, vmax):
    out = field.copy()
    out[mask] = np.float32(0.5 * (vmin + vmax))
    return out


def test_full_chain_with_nadir_policy(data, scene, water_tile_dir):
    cloud = cloud_mask_from_valid(scene["valid"], data.results["fsnow"])
    out = process(
        data,
        apply_canopy_correction=True,
        canopy_correction_policy="nadir_full_shade",
        sensor="hls_l30",
        apply_shade_adjustment=True,
        tree_cover_threshold=0.10,
        cloud_mask=cloud,
        land_water_directory=water_tile_dir,
        gate_products=True,
        product_fsnow_variable="canopy_adjusted_fsnow",
    )
    res = out.results
    validate_results(res, scene=scene)
    _require_unchanged_base_results(data.results, res)
    for name in (
        "canopy_adjusted_fsnow", "shade_adjusted_fsnow", "fsnow_product",
        "grain_radius_product", "lap_concentration_product",
        "cloud_mask", "land_mask", "tree_mask", "shade_saturated_mask",
    ):
        assert name in res, name
        assert res[name].dims == ("y", "x")
    assert res["cloud_mask"].dtype == np.uint8
    assert np.all(np.isnan(res["fsnow_product"].isel(y=0).values))             # cloud row
    assert np.all(np.isnan(res["fsnow_product"].isel(x=slice(4, 8)).values))   # water columns
    assert int(res["tree_mask"].sum()) == 16                                   # 2x2 dilated to 4x4
    assert np.all(np.isnan(res["fsnow_product"].values[3:5, 1:3]))
    assert res["fsnow_product"].attrs["source_variable"] == "canopy_adjusted_fsnow"


def test_chain_with_treefill_keeps_tree_mask_as_companion(data, scene, water_tile_dir):
    cloud = cloud_mask_from_valid(scene["valid"], data.results["fsnow"])
    out = process(
        data,
        apply_shade_adjustment=True,
        tree_cover_threshold=0.10,
        cloud_mask=cloud,
        land_water_directory=water_tile_dir,
        treefill_inpainter=constant_inpainter,
        gate_products=True,
        product_fsnow_variable="shade_adjusted_fsnow",
    )
    res = out.results
    validate_results(res, scene=scene)
    assert "fsnow_treefilled" in res and "grain_radius_treefilled" in res
    assert res["fsnow_product"].attrs["source_variable"] == "fsnow_treefilled"
    assert res["grain_radius_product"].attrs["source_variable"] == "grain_radius_treefilled"
    assert "tree_mask" not in res["fsnow_product"].attrs["gates"]
    assert int(res["tree_mask"].sum()) == 16
    tree = res["tree_mask"].values.astype(bool)
    land = res["land_mask"].values.astype(bool)
    inside = tree & land & ~res["cloud_mask"].values.astype(bool)
    assert inside.any()
    assert np.all(res["fsnow_product"].values[inside] == np.float32(0.5))


def test_tree_mask_and_shade_without_gate(data):
    out = process(data, apply_shade_adjustment=True, tree_cover_threshold=0.10)
    res = out.results
    assert res["tree_mask"].dtype == np.uint8
    assert res["shade_saturated_mask"].dtype == np.uint8
    assert "fsnow_product" not in res
    assert "shade_adjusted_fsnow" in res


def test_no_options_returns_input(data):
    assert process(data) is data


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"gate_products": True, "land_water_directory": "/x", "product_fsnow_variable": "fsnow"}, "cloud_mask"),
        ({"gate_products": True, "product_fsnow_variable": "fsnow"}, "land_water_directory"),
        ({"gate_products": True, "land_water_directory": "/x"}, "product_fsnow_variable"),
        ({"treefill_inpainter": constant_inpainter, "product_fsnow_variable": "fsnow"}, "tree_cover_threshold"),
        ({"treefill_inpainter": constant_inpainter, "tree_cover_threshold": 0.1}, "product_fsnow_variable"),
        ({"product_fsnow_variable": "fsnow"}, "gate_products"),
    ],
)
def test_chain_requires_explicit_inputs(data, scene, kwargs, match):
    if "gate_products" in kwargs and "cloud_mask" not in match:
        kwargs = {**kwargs, "cloud_mask": cloud_mask_from_valid(scene["valid"], data.results["fsnow"])}
    with pytest.raises(ValueError, match=match):
        process(data, **kwargs)


def test_gate_with_missing_product_variable_raises(data, scene, water_tile_dir):
    cloud = cloud_mask_from_valid(scene["valid"], data.results["fsnow"])
    with pytest.raises(ValueError, match="canopy_adjusted_fsnow"):
        process(
            data, cloud_mask=cloud, land_water_directory=water_tile_dir,
            gate_products=True, product_fsnow_variable="canopy_adjusted_fsnow",
        )


def test_tree_mask_rejects_canopy_on_shifted_grid(data):
    canopy = data.ancillary["canopy_fraction"]
    shifted = xr.Dataset({"canopy_fraction": canopy.assign_coords(x=canopy["x"] + 1.0)})
    bad = SpiresData(scene=data.scene, ancillary=shifted, results=data.results)
    with pytest.raises(ValueError, match="does not match"):
        process(bad, tree_cover_threshold=0.10)


def test_inplace_inpainter_cannot_mutate_base_results(data):
    def in_place(field, mask, vmin, vmax):
        field[mask] = np.float32(0.123)
        return field

    before = data.results.copy(deep=True)
    out = process(
        data, tree_cover_threshold=0.10, treefill_inpainter=in_place,
        product_fsnow_variable="fsnow",
    )
    for name in ("fsnow", "fshade", "grain_radius", "lap_concentration"):
        assert out.results[name].identical(before[name]), name


def test_base_guard_detects_in_place_write(data):
    snapshot = data.results.copy(deep=True)
    updated = data.results.copy(deep=False)
    updated["fsnow"].values[0, 0] = 0.42
    with pytest.raises(RuntimeError, match="fsnow"):
        _require_unchanged_base_results(snapshot, updated)
