"""Companion masks for the Landsat / Sentinel-2 / HLS product chain.

Every function returns a boolean ``(y, x)`` DataArray on the results grid. Masks
are True where the pixel is *excluded*, except :func:`land_mask_from_tiles`,
which is True for land because the product semantics are "keep land".
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    require_matching_coords,
    validate_target_layout,
)


def mask_to_uint8(mask: xr.DataArray) -> xr.DataArray:
    """Store a boolean mask as a uint8 companion layer (1 = True)."""
    _require_boolean(mask, mask.name or "mask")
    out = mask.astype("uint8")
    out.attrs = dict(mask.attrs)
    out.attrs.update(flag_values="0,1", flag_meanings="false true")
    return out


def cloud_mask_from_valid(valid: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Invert a spires-io ``valid`` layer into a cloud / invalid-observation mask."""
    validate_target_layout(target, "target")
    _require_boolean(valid, "valid")
    _require_plane(valid, target, "valid")
    mask = (~valid).rename("cloud_mask")
    mask.attrs = {
        "long_name": "Cloud or invalid observation mask (True = excluded)",
        "source": str(valid.attrs.get("source", "spires_io.valid")),
    }
    return mask


def _require_boolean(layer: xr.DataArray, label: str) -> None:
    if not isinstance(layer, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    if layer.dtype != np.dtype(bool):
        raise ValueError(f"{label} must be a boolean DataArray; got {layer.dtype}")


def _require_plane(layer: xr.DataArray, target: xr.DataArray, label: str) -> None:
    if layer.dims != ("y", "x"):
        raise ValueError(f"{label} must have dimensions ('y', 'x'); got {layer.dims}")
    require_matching_coords(
        layer, target, ("y", "x"), label=label, target_label="results"
    )
