# tests/test_cloudmask.py
"""OmniCloudMask provider: band stacking, classification wrapper, collapse to cloud_mask."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.conftest import LAT, LON, N


def reflectance(bands=("blue", "green", "red", "nir", "swir1")) -> xr.DataArray:
    """(y, x, band) float32 scene like spires_io.open_raster_scene returns."""
    values = np.stack(
        [np.full((N, N), 0.1 * (i + 1), dtype=np.float32) for i in range(len(bands))], axis=-1
    )
    return xr.DataArray(
        values,
        dims=("y", "x", "band"),
        coords={"y": LAT, "x": LON, "band": list(bands)},
        name="reflectance",
    )


def test_rgnir_stack_orders_scales_and_casts():
    from spires_postprocess.cloudmask import rgnir_stack

    stack = rgnir_stack(reflectance())
    assert stack.shape == (3, N, N)
    assert stack.dtype == np.uint16
    # bands = blue .1, green .2, red .3, nir .4 -> red, green, nir
    assert stack[0, 0, 0] == 3000
    assert stack[1, 0, 0] == 2000
    assert stack[2, 0, 0] == 4000


def test_rgnir_stack_accepts_band_first_layout():
    from spires_postprocess.cloudmask import rgnir_stack

    stack = rgnir_stack(reflectance().transpose("band", "y", "x"))
    assert stack.shape == (3, N, N)
    assert stack[2, 0, 0] == 4000


def test_rgnir_stack_nodata_zeroes_all_three_bands():
    from spires_postprocess.cloudmask import rgnir_stack

    refl = reflectance()
    refl.loc[dict(band="nir")][0, 0] = np.nan  # one band missing -> whole pixel nodata
    stack = rgnir_stack(refl)
    assert (stack[:, 0, 0] == 0).all()
    assert (stack[:, 1, 1] != 0).all()


def test_rgnir_stack_never_maps_an_observation_to_nodata():
    from spires_postprocess.cloudmask import rgnir_stack

    refl = reflectance()
    refl.loc[dict(band="red")][0, 0] = 0.0
    refl.loc[dict(band="green")][0, 0] = -0.05  # Landsat C2 offset can go negative
    refl.loc[dict(band="nir")][0, 0] = 9.0  # absurdly bright: clip, do not wrap
    stack = rgnir_stack(refl)
    assert stack[0, 0, 0] == 1
    assert stack[1, 0, 0] == 1
    assert stack[2, 0, 0] == 65535


def test_rgnir_stack_requires_red_green_nir():
    from spires_postprocess.cloudmask import rgnir_stack

    with pytest.raises(ValueError, match="nir"):
        rgnir_stack(reflectance(bands=("blue", "green", "red")))
    with pytest.raises(ValueError, match="band"):
        rgnir_stack(reflectance().isel(band=0, drop=True))


class StubClassifier:
    """Returns a fixed class plane; records what it was called with."""

    provenance = {"cloud_model": "stub", "cloud_backend": "test"}

    def __init__(self, classes: np.ndarray):
        self.classes = classes
        self.received = None

    def __call__(self, rgnir: np.ndarray) -> np.ndarray:
        self.received = rgnir
        return self.classes


def checker_classes() -> np.ndarray:
    classes = np.zeros((N, N), dtype=np.uint8)
    classes[1, :] = 1
    classes[2, :] = 2
    classes[3, :] = 3
    return classes


def test_classify_clouds_returns_uint8_plane_with_provenance():
    from spires_postprocess.cloudmask import classify_clouds

    stub = StubClassifier(checker_classes())
    out = classify_clouds(reflectance(), classifier=stub)
    assert out.dtype == np.uint8
    assert out.dims == ("y", "x")
    assert out.name == "cloud_classes"
    np.testing.assert_array_equal(out["y"].values, LAT)
    np.testing.assert_array_equal(out["x"].values, LON)
    assert out.attrs["flag_values"] == "0,1,2,3,255"
    assert out.attrs["cloud_model"] == "stub"
    assert out.attrs["cloud_backend"] == "test"
    assert out.attrs["input_bands"] == "red,green,nir"
    assert stub.received.dtype == np.uint16 and stub.received.shape == (3, N, N)


def test_classify_clouds_flags_nodata_as_255_not_clear():
    from spires_postprocess.cloudmask import classify_clouds

    refl = reflectance()
    refl.loc[dict(band="red")][0, :] = np.nan
    out = classify_clouds(refl, classifier=StubClassifier(checker_classes()))
    assert (out.values[0, :] == 255).all()
    assert (out.values[1, :] == 1).all()


def test_classify_clouds_rejects_bad_classifier_output():
    from spires_postprocess.cloudmask import classify_clouds

    with pytest.raises(ValueError, match="shape"):
        classify_clouds(reflectance(), classifier=StubClassifier(np.zeros((N, N + 1), np.uint8)))
    with pytest.raises(ValueError, match="uint8"):
        classify_clouds(reflectance(), classifier=StubClassifier(np.zeros((N, N), np.int32)))
    bad = checker_classes()
    bad[0, 0] = 7
    with pytest.raises(ValueError, match="0..3"):
        classify_clouds(reflectance(), classifier=StubClassifier(bad))


def test_classify_clouds_works_without_provenance_attribute():
    from spires_postprocess.cloudmask import classify_clouds

    out = classify_clouds(reflectance(), classifier=lambda rgnir: checker_classes())
    assert "cloud_model" not in out.attrs
    assert out.attrs["flag_meanings"].startswith("clear thick_cloud")
