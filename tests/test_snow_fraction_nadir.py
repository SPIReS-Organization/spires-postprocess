import numpy as np
import pytest
import xarray as xr

from spires_postprocess.snow_fraction import (
    CANOPY_CORRECTION_POLICIES,
    CANOPY_POLICY_BY_SENSOR,
    apply_snow_fraction_adjustments,
    validate_canopy_correction_policy,
)

NADIR_SENSORS = ("landsat", "sentinel2", "hls_l30", "hls_s30")


@pytest.mark.parametrize("sensor", NADIR_SENSORS)
def test_nadir_policy_accepted_for_near_nadir_sensors(sensor):
    assert CANOPY_POLICY_BY_SENSOR[sensor] == "nadir_full_shade"
    validate_canopy_correction_policy("nadir_full_shade", sensor)


@pytest.mark.parametrize("sensor", ("modis", "viirs"))
def test_nadir_policy_rejected_for_modis_viirs(sensor):
    with pytest.raises(ValueError, match="requires"):
        validate_canopy_correction_policy("nadir_full_shade", sensor)


def test_policies_tuple_is_deduplicated():
    assert CANOPY_CORRECTION_POLICIES == (
        "modis_full_shade", "viirs_canopy_weighted_shade", "nadir_full_shade",
    )


def test_nadir_adjustment_needs_no_geometry(results, grid):
    canopy = grid.copy(data=np.full((8, 8), 0.3, dtype=np.float32))
    out = apply_snow_fraction_adjustments(
        results, canopy_fraction=canopy, canopy_correction_policy="nadir_full_shade", sensor="landsat",
    )
    adjusted = out["canopy_adjusted_fsnow"]
    assert adjusted.dtype == np.float32
    expected = (results["fsnow"] / (1 - (results["fshade"] + 0.3).clip(0, 0.99))).clip(0, 1)
    np.testing.assert_allclose(adjusted.values, expected.values.astype(np.float32), atol=1e-6)
    assert adjusted.attrs["canopy_correction_policy"] == "nadir_full_shade"
    assert adjusted.attrs["sensor"] == "landsat"
    assert adjusted.attrs["terrain_geometry_used"] == 0
    assert adjusted.attrs["shade_treatment"] == "full"
    assert adjusted.attrs["viewable_canopy_model"] == "nadir_canopy_equals_cover"
    assert out["snow_detected"].attrs["raw_fsnow_threshold"] == pytest.approx(0.05)
    assert out["fsnow"].identical(results["fsnow"])


def test_nadir_equals_modis_branch_at_zero_geometry(results, grid):
    canopy = grid.copy(data=np.linspace(0, 0.6, 64, dtype=np.float32).reshape(8, 8))
    zero = grid.copy(data=np.zeros((8, 8), dtype=np.float32))
    nadir = apply_snow_fraction_adjustments(
        results, canopy_fraction=canopy, canopy_correction_policy="nadir_full_shade", sensor="hls_l30",
    )["canopy_adjusted_fsnow"]
    modis = apply_snow_fraction_adjustments(
        results, canopy_fraction=canopy, sensor_zenith=zero, sensor_azimuth=zero, slope=zero, aspect=zero,
        canopy_correction_policy="modis_full_shade", sensor="modis",
    )["canopy_adjusted_fsnow"]
    np.testing.assert_allclose(nadir.values, modis.values, atol=1e-6)


def test_nadir_rejects_supplied_geometry(results, grid):
    canopy = grid.copy(data=np.full((8, 8), 0.3, dtype=np.float32))
    zero = grid.copy(data=np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="geometry"):
        apply_snow_fraction_adjustments(
            results, canopy_fraction=canopy, sensor_zenith=zero,
            canopy_correction_policy="nadir_full_shade", sensor="landsat",
        )


def test_modis_still_requires_geometry(results, grid):
    canopy = grid.copy(data=np.full((8, 8), 0.3, dtype=np.float32))
    with pytest.raises(ValueError, match="slope"):
        apply_snow_fraction_adjustments(
            results, canopy_fraction=canopy, canopy_correction_policy="modis_full_shade", sensor="modis",
        )


def test_nadir_ignores_crown_radii(results, grid):
    canopy = grid.copy(data=np.full((8, 8), 0.3, dtype=np.float32))
    default = apply_snow_fraction_adjustments(
        results, canopy_fraction=canopy, canopy_correction_policy="nadir_full_shade", sensor="landsat",
    )["canopy_adjusted_fsnow"]
    custom = apply_snow_fraction_adjustments(
        results, canopy_fraction=canopy, canopy_correction_policy="nadir_full_shade", sensor="landsat",
        average_vertical_crown_radius=9.0, average_horizontal_crown_radius=0.5,
    )["canopy_adjusted_fsnow"]
    np.testing.assert_array_equal(custom.values, default.values)
    assert "average_vertical_crown_radius" not in custom.attrs
    assert "average_horizontal_crown_radius" not in custom.attrs


def test_sensor_specific_policy_still_requires_evaluated_crown_radii(results, grid):
    canopy = grid.copy(data=np.full((8, 8), 0.3, dtype=np.float32))
    zero = grid.copy(data=np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="crown radii"):
        apply_snow_fraction_adjustments(
            results, canopy_fraction=canopy, sensor_zenith=zero, sensor_azimuth=zero, slope=zero, aspect=zero,
            canopy_correction_policy="modis_full_shade", sensor="modis",
            average_vertical_crown_radius=9.0,
        )
