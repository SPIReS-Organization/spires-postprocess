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


def test_cloud_mask_from_classes_excludes_cloud_shadow_and_nodata(grid):
    from spires_postprocess.cloudmask import classify_clouds, cloud_mask_from_classes
    from spires_postprocess.masks import mask_to_uint8

    refl = reflectance()
    refl.loc[dict(band="green")][4, :] = np.nan
    classes = classify_clouds(refl, classifier=StubClassifier(checker_classes()))
    mask = cloud_mask_from_classes(classes, grid)
    assert mask.dtype == np.dtype(bool)
    assert mask.name == "cloud_mask"
    assert mask.dims == ("y", "x")
    assert not mask.values[0].any()  # clear
    assert mask.values[1].all() and mask.values[2].all() and mask.values[3].all()
    assert mask.values[4].all()  # no observation
    assert not mask.values[5:].any()
    assert "True = excluded" in mask.attrs["long_name"]
    assert mask.attrs["source"] == "spires_postprocess.cloudmask"
    assert mask_to_uint8(mask).dtype == np.uint8  # the companion-layer path accepts it


def test_cloud_mask_from_classes_can_keep_nodata_and_narrow_classes(grid):
    from spires_postprocess.cloudmask import cloud_mask_from_classes

    classes = xr.DataArray(checker_classes(), dims=("y", "x"), coords={"y": LAT, "x": LON})
    classes.values[4, :] = 255
    mask = cloud_mask_from_classes(classes, grid, cloud_classes=(1,), exclude_no_observation=False)
    assert mask.values[1].all()
    assert not mask.values[2].any() and not mask.values[3].any() and not mask.values[4].any()


def test_cloud_mask_from_classes_validates_inputs(grid):
    from spires_postprocess.cloudmask import cloud_mask_from_classes

    with pytest.raises(ValueError, match="uint8"):
        cloud_mask_from_classes(xr.DataArray(np.zeros((N, N), np.int16), dims=("y", "x"), coords={"y": LAT, "x": LON}), grid)
    shifted = xr.DataArray(checker_classes(), dims=("y", "x"), coords={"y": LAT + 1.0, "x": LON})
    with pytest.raises(ValueError):
        cloud_mask_from_classes(shifted, grid)


def test_omnicloudmask_classifier_import_error_names_the_extra(monkeypatch):
    import sys

    from spires_postprocess.cloudmask import omnicloudmask_classifier

    monkeypatch.setitem(sys.modules, "omnicloudmask", None)  # makes `import omnicloudmask` raise ImportError
    with pytest.raises(ImportError, match=r"spires-postprocess\[cloudmask\]"):
        omnicloudmask_classifier()


def test_classify_clouds_default_backend_needs_the_extra(monkeypatch):
    import sys

    from spires_postprocess.cloudmask import classify_clouds

    monkeypatch.setitem(sys.modules, "omnicloudmask", None)
    with pytest.raises(ImportError, match="cloudmask"):
        classify_clouds(reflectance())


def test_omnicloudmask_classifier_calls_predict_with_pinned_defaults(monkeypatch):
    import sys
    import types

    from spires_postprocess.cloudmask import OMNICLOUDMASK_VERSION, omnicloudmask_classifier

    calls = {}

    def fake_predict(input_array, **kwargs):
        calls["array"] = input_array
        calls.update(kwargs)
        return np.zeros((1,) + input_array.shape[1:], dtype=np.uint8)

    fake = types.SimpleNamespace(__version__=OMNICLOUDMASK_VERSION, predict_from_array=fake_predict)
    monkeypatch.setitem(sys.modules, "omnicloudmask", fake)

    classifier = omnicloudmask_classifier()
    out = classifier(np.ones((3, 40, 40), np.uint16))
    assert out.shape == (40, 40) and out.dtype == np.uint8
    assert calls["inference_device"] == "cpu"
    assert calls["mosaic_device"] == "cpu"
    assert calls["no_data_value"] == 0
    assert calls["model_version"] == 4.0
    assert calls["patch_size"] == 1000 and calls["patch_overlap"] == 300
    assert calls["batch_size"] == 1
    assert classifier.provenance == {
        "cloud_model": "OmniCloudMask",
        "cloud_model_version": "4.0",
        "omnicloudmask_version": OMNICLOUDMASK_VERSION,
        "cloud_backend": "omnicloudmask.predict_from_array[cpu]",
    }


def test_omnicloudmask_classifier_warns_on_version_drift(monkeypatch):
    import sys
    import types

    from spires_postprocess.cloudmask import omnicloudmask_classifier

    fake = types.SimpleNamespace(__version__="9.9.9", predict_from_array=lambda a, **k: None)
    monkeypatch.setitem(sys.modules, "omnicloudmask", fake)
    with pytest.warns(UserWarning, match="9.9.9"):
        omnicloudmask_classifier()


def test_public_exports():
    import spires_postprocess

    for name in ("classify_clouds", "cloud_mask_from_classes", "omnicloudmask_classifier", "Classifier", "rgnir_stack"):
        assert name in spires_postprocess.__all__
        assert hasattr(spires_postprocess, name)


def test_omnicloudmask_reference_backend_is_deterministic_on_cpu():
    pytest.importorskip("omnicloudmask")
    from spires_postprocess.cloudmask import OMNICLOUDMASK_VERSION, classify_clouds, omnicloudmask_classifier

    n = 64
    rng = np.random.default_rng(0)
    base = rng.uniform(0.05, 0.25, size=(n, n)).astype(np.float32)
    bands = {"red": base, "green": base * 0.9, "nir": base * 1.4}
    for b in bands.values():
        b[16:32, 16:48] = 0.85  # bright block: cloud-like
    stack = np.stack([bands["red"], bands["green"], bands["nir"]], axis=-1)
    stack[:4, :, :] = np.nan  # nodata rows
    refl = xr.DataArray(
        stack,
        dims=("y", "x", "band"),
        coords={"y": np.arange(n, dtype=float), "x": np.arange(n, dtype=float), "band": ["red", "green", "nir"]},
    )
    classifier = omnicloudmask_classifier()
    first = classify_clouds(refl, classifier=classifier)
    second = classify_clouds(refl, classifier=classifier)
    np.testing.assert_array_equal(first.values, second.values)
    assert set(np.unique(first.values)) <= {0, 1, 2, 3, 255}
    assert (first.values[:4] == 255).all()
    assert (first.values[4:] != 255).all()
    assert first.attrs["omnicloudmask_version"] == OMNICLOUDMASK_VERSION
    assert first.attrs["cloud_model_version"] == "4.0"
