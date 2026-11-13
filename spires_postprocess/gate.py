"""Product gate: apply companion masks and the minimum-fsnow rule to retrievals."""

from __future__ import annotations

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    prepare_aligned_layer,
    validate_target_layout,
)
from spires_postprocess.masks import mask_to_uint8

PRODUCT_MIN_FSNOW = 0.10


def gate_products(
    results: xr.Dataset,
    *,
    fsnow_variable: str,
    land_mask: xr.DataArray,
    cloud_mask: xr.DataArray,
    tree_mask: xr.DataArray | None = None,
    shade_saturated_mask: xr.DataArray | None = None,
    min_fsnow: float = PRODUCT_MIN_FSNOW,
    grain_variable: str = "grain_radius",
    lap_variable: str = "lap_concentration",
) -> xr.Dataset:
    """Return a new dataset of gated products plus uint8 companion masks.

    ``fsnow_product`` is ``results[fsnow_variable]`` with NaN where cloud, water,
    tree, or shade-saturated. ``grain_radius_product`` and
    ``lap_concentration_product`` additionally NaN where ``fsnow_product <
    min_fsnow``. The layer names are explicit: the gate never guesses which
    fsnow layer is the product. ``results`` is not modified.
    """
    if not isinstance(results, xr.Dataset):
        raise TypeError("results must be an xarray.Dataset")
    missing = [n for n in (fsnow_variable, grain_variable, lap_variable) if n not in results]
    if missing:
        raise ValueError(f"results is missing required variable(s): {missing}")
    fsnow = results[fsnow_variable]
    validate_target_layout(fsnow, f"results[{fsnow_variable!r}]")

    applied = {"cloud_mask": cloud_mask, "land_mask": land_mask}
    if tree_mask is not None:
        applied["tree_mask"] = tree_mask
    if shade_saturated_mask is not None:
        applied["shade_saturated_mask"] = shade_saturated_mask
    aligned = {}
    for name, mask in applied.items():
        if not isinstance(mask, xr.DataArray) or mask.dtype != np.dtype(bool):
            raise ValueError(f"{name} must be a boolean DataArray")
        aligned[name] = prepare_aligned_layer(mask, fsnow, label=name, target_label="results")

    excluded = aligned["cloud_mask"] | ~aligned["land_mask"]
    for name in ("tree_mask", "shade_saturated_mask"):
        if name in aligned:
            excluded = excluded | aligned[name]
    gates = ",".join(applied)

    fsnow_product = fsnow.where(~excluded).astype("float32").rename("fsnow_product")
    fsnow_product.attrs = dict(fsnow.attrs)
    fsnow_product.attrs.update(
        long_name="Gated fractional snow product",
        units="1",
        source_variable=fsnow_variable,
        gates=gates,
    )
    keep = ~excluded & (fsnow_product >= np.float32(min_fsnow))

    out = xr.Dataset({"fsnow_product": fsnow_product})
    for source_name, target_name in ((grain_variable, "grain_radius_product"), (lap_variable, "lap_concentration_product")):
        field = results[source_name]
        prepared = prepare_aligned_layer(field, fsnow, label=f"results[{source_name!r}]", target_label="results")
        product = prepared.where(keep).astype("float32").rename(target_name)
        product.attrs = dict(field.attrs)
        product.attrs.update(
            long_name=f"Gated {field.attrs.get('long_name', source_name)}",
            source_variable=source_name,
            gates=gates,
            min_fsnow=float(min_fsnow),
            min_fsnow_variable="fsnow_product",
        )
        out[target_name] = product
    for name, mask in applied.items():
        out[name] = mask_to_uint8(mask.rename(name))
    return out
