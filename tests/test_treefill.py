import numpy as np
import pytest
import xarray as xr

from spires_postprocess.treefill import treefill_results


def constant_inpainter(value):
    def inpainter(field, mask, vmin, vmax):
        assert field.dtype == np.float32 and mask.dtype == bool
        assert field.shape == mask.shape
        out = field.copy()
        out[mask] = value
        return out

    inpainter.__name__ = f"constant_{value}"
    return inpainter


@pytest.fixture
def tree(results):
    mask = xr.zeros_like(results["fsnow"], dtype=bool).rename("tree_mask")
    mask.values[2:5, 2:5] = True
    mask.attrs = {"cover_threshold": 0.1, "dilate_cells": 1}
    return mask


def test_treefill_only_changes_masked_pixels(results, tree):
    out = treefill_results(
        results, fsnow_variable="fsnow", tree_mask=tree, inpainter=constant_inpainter(0.75)
    )
    for name in ("fsnow", "fshade", "grain_radius", "lap_concentration"):
        assert out[name].identical(results[name])
    filled = out["fsnow_treefilled"]
    assert filled.dtype == np.float32
    np.testing.assert_array_equal(filled.values[~tree.values], results["fsnow"].values[~tree.values])
    assert np.all(filled.values[tree.values] == np.float32(0.75))
    assert filled.attrs["inpainter"] == "constant_0.75"
    assert filled.attrs["treefill_vmin"] == 0.0
    assert filled.attrs["treefill_vmax"] == 1.0
    assert filled.attrs["cover_threshold"] == 0.1
    assert "under-dispersed" in filled.attrs["treefill_caveat"]


def test_treefill_scales_grain_and_lap_to_unmasked_max(results, tree):
    seen = {}

    def recording(field, mask, vmin, vmax):
        seen[len(seen)] = (vmin, vmax)
        return field

    treefill_results(results, fsnow_variable="fsnow", tree_mask=tree, inpainter=recording)
    unmasked_grain = float(results["grain_radius"].where(~tree).max())
    unmasked_lap = float(results["lap_concentration"].where(~tree).max())
    assert seen[0] == (0.0, 1.0)
    assert seen[1] == (0.0, pytest.approx(unmasked_grain))
    assert seen[2] == (0.0, pytest.approx(unmasked_lap))


def test_treefill_clips_inpainter_output_inside_mask(results, tree):
    out = treefill_results(
        results, fsnow_variable="fsnow", tree_mask=tree, inpainter=constant_inpainter(7.0)
    )
    assert np.all(out["fsnow_treefilled"].values[tree.values] == np.float32(1.0))


def test_treefill_uses_named_fsnow_variable(results, tree):
    results = results.assign(canopy_adjusted_fsnow=results["fsnow"] * np.float32(0.5))
    out = treefill_results(
        results, fsnow_variable="canopy_adjusted_fsnow", tree_mask=tree,
        inpainter=constant_inpainter(0.2),
    )
    assert out["fsnow_treefilled"].attrs["source_variable"] == "canopy_adjusted_fsnow"
    np.testing.assert_array_equal(
        out["fsnow_treefilled"].values[~tree.values],
        results["canopy_adjusted_fsnow"].values[~tree.values],
    )


def test_treefill_rejects_bad_inputs(results, tree):
    with pytest.raises(ValueError, match="missing"):
        treefill_results(results, fsnow_variable="nope", tree_mask=tree, inpainter=constant_inpainter(0))
    with pytest.raises(ValueError, match="boolean"):
        treefill_results(results, fsnow_variable="fsnow", tree_mask=tree.astype("uint8"), inpainter=constant_inpainter(0))

    def wrong_shape(field, mask, vmin, vmax):
        return field[:-1]

    with pytest.raises(ValueError, match="shape"):
        treefill_results(results, fsnow_variable="fsnow", tree_mask=tree, inpainter=wrong_shape)
