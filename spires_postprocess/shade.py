"""Shade-only snow-fraction rescale for near-nadir sensors without canopy data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    require_float32,
    require_matching_coords,
    require_values_in_range,
    validate_target_layout,
)

SHADE_SATURATION = 0.99  # == snow_fraction._OBSCURATION_LIMIT


def shade_saturated_mask(fshade: xr.DataArray) -> xr.DataArray:
    """True where shade is at or above saturation; NaN shade is not saturated."""
    validate_target_layout(fshade, "fshade")
    mask = (fshade >= np.float32(SHADE_SATURATION)).rename("shade_saturated_mask")
    mask.attrs = {
        "long_name": "Shade-saturated pixel (snow fraction unknown, True = excluded)",
        "shade_saturation": SHADE_SATURATION,
        "comparison": "greater_or_equal",
    }
    return mask


def shade_adjusted_fsnow(fsnow: xr.DataArray, fshade: xr.DataArray) -> xr.DataArray:
    """``fsnow / (1 - fshade)`` clipped to [0, 1]; NaN where shade is saturated.

    Saturated cells are unknown, not zero. Inputs must be float32 fractions.
    """
    dims = validate_target_layout(fsnow, "fsnow")
    require_float32(fsnow, "fsnow")
    require_float32(fshade, "fshade")
    require_values_in_range(fsnow, "fsnow", minimum=0.0, maximum=1.0)
    require_values_in_range(fshade, "fshade", minimum=0.0, maximum=1.0)
    if fshade.dims != dims:
        raise ValueError("fshade must have the same dimensions as fsnow")
    require_matching_coords(fshade, fsnow, dims, label="fshade", target_label="fsnow")

    saturated = fshade >= np.float32(SHADE_SATURATION)
    denominator = np.float32(1.0) - fshade.clip(max=np.float32(SHADE_SATURATION))
    adjusted = (fsnow / denominator).clip(0.0, 1.0).where(~saturated)
    adjusted = adjusted.astype("float32").rename("shade_adjusted_fsnow")
    adjusted.attrs = {
        "long_name": "Snow fraction adjusted for shade only",
        "units": "1",
        "model": "flat_shade_rescale",
        "shade_included": 1,
        "canopy_included": 0,
        "ice_included": 0,
        "shade_saturation": SHADE_SATURATION,
        "saturated_treatment": "nan",
    }
    return adjusted
