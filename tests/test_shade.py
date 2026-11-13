import numpy as np
import pytest

from spires_postprocess.shade import (
    SHADE_SATURATION,
    shade_adjusted_fsnow,
    shade_saturated_mask,
)


def test_saturation_constant_matches_upstream():
    from spires_postprocess.snow_fraction import _OBSCURATION_LIMIT

    assert SHADE_SATURATION == _OBSCURATION_LIMIT == 0.99


def test_shade_saturated_mask(results):
    fshade = results["fshade"].copy()
    fshade.values[0, 0] = 0.99
    fshade.values[0, 1] = 1.0
    fshade.values[0, 2] = np.nan
    fshade.values[0, 3] = 0.98
    mask = shade_saturated_mask(fshade)
    assert mask.dtype == bool
    assert mask.name == "shade_saturated_mask"
    assert bool(mask[0, 0]) and bool(mask[0, 1])
    assert not bool(mask[0, 2]) and not bool(mask[0, 3])
    assert int(mask.sum()) == 2


def test_shade_adjusted_fsnow_rescales_and_clips(results):
    fsnow = results["fsnow"].copy()
    fshade = results["fshade"].copy()
    fsnow.values[0, 0], fshade.values[0, 0] = 0.4, 0.5      # -> 0.8
    fsnow.values[0, 1], fshade.values[0, 1] = 0.9, 0.5      # -> clip 1.0
    fsnow.values[0, 2], fshade.values[0, 2] = 0.3, 0.99     # saturated -> NaN
    fsnow.values[0, 3], fshade.values[0, 3] = np.nan, 0.1   # NaN propagates
    fsnow.values[0, 4], fshade.values[0, 4] = 0.0, 0.0      # stays 0
    out = shade_adjusted_fsnow(fsnow, fshade)
    assert out.dtype == np.float32
    assert out.name == "shade_adjusted_fsnow"
    assert out[0, 0].item() == pytest.approx(0.8, abs=1e-6)
    assert out[0, 1].item() == 1.0
    assert np.isnan(out[0, 2].item())
    assert np.isnan(out[0, 3].item())
    assert out[0, 4].item() == 0.0
    assert out.attrs["shade_saturation"] == 0.99
    assert out.attrs["canopy_included"] == 0


def test_shade_adjusted_fsnow_rejects_float64(results):
    with pytest.raises(ValueError, match="float32"):
        shade_adjusted_fsnow(results["fsnow"].astype("float64"), results["fshade"])
