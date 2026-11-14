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
