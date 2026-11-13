"""Canopy treefill: inpaint retrievals under the tree mask via an injected callable.

The core package never imports torch. ``Inpainter`` is any callable
``(field, mask, vmin, vmax) -> field`` on float32 ``(y, x)`` arrays where
``mask`` is boolean (True = hole). :mod:`spires_postprocess.treefill_client`
provides one backed by the spipyservice LaMa HTTP service.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    require_matching_coords,
    validate_target_layout,
)

Inpainter = Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]

TREEFILL_CAVEAT = (
    "LaMa inpainting is unbiased in the median but under-dispersed: filled "
    "pixels are smoother than the surrounding retrieval"
)
_FILLED_LAYERS = (
    ("grain_radius", "grain_radius_treefilled"),
    ("lap_concentration", "lap_concentration_treefilled"),
)


def treefill_results(
    results: xr.Dataset,
    *,
    fsnow_variable: str,
    tree_mask: xr.DataArray,
    inpainter: Inpainter,
) -> xr.Dataset:
    """Add ``fsnow_treefilled``, ``grain_radius_treefilled``, ``lap_concentration_treefilled``.

    Only pixels where ``tree_mask`` is True change; all others are bit-identical
    to the source layer. ``fsnow`` is scaled to [0, 1]; grain radius and LAP are
    scaled to [0, max of the un-masked field]. The input dataset is not modified.
    """
    if not isinstance(results, xr.Dataset):
        raise TypeError("results must be an xarray.Dataset")
    missing = [n for n in (fsnow_variable, *(s for s, _ in _FILLED_LAYERS)) if n not in results]
    if missing:
        raise ValueError(f"results is missing required variable(s): {missing}")
    fsnow = results[fsnow_variable]
    if validate_target_layout(fsnow, f"results[{fsnow_variable!r}]") != ("y", "x"):
        raise ValueError("treefill supports ('y', 'x') results only")
    if not isinstance(tree_mask, xr.DataArray) or tree_mask.dtype != np.dtype(bool):
        raise ValueError("tree_mask must be a boolean DataArray")
    if tree_mask.dims != ("y", "x"):
        raise ValueError("tree_mask must have dimensions ('y', 'x')")
    require_matching_coords(tree_mask, fsnow, ("y", "x"), label="tree_mask", target_label="results")
    hole = np.asarray(tree_mask.values, dtype=bool)

    common_attrs = {
        "inpainter": getattr(inpainter, "__name__", repr(inpainter)),
        "cover_threshold": tree_mask.attrs.get("cover_threshold", "unknown"),
        "dilate_cells": tree_mask.attrs.get("dilate_cells", "unknown"),
        "treefill_caveat": TREEFILL_CAVEAT,
    }
    out = results.copy()
    out["fsnow_treefilled"] = _fill_layer(
        fsnow, hole, inpainter, 0.0, 1.0, name="fsnow_treefilled",
        long_name="Snow fraction with canopy pixels inpainted", attrs=common_attrs,
    )
    for source_name, target_name in _FILLED_LAYERS:
        field = results[source_name]
        vmax = float(np.nanmax(np.where(hole, np.nan, field.values))) if (~hole).any() else np.nan
        if not np.isfinite(vmax):
            raise ValueError(f"{source_name} has no finite un-masked pixel to scale the inpainter")
        if vmax <= 0.0:
            vmax = 1.0
        out[target_name] = _fill_layer(
            field, hole, inpainter, 0.0, vmax, name=target_name,
            long_name=f"{field.attrs.get('long_name', source_name)} with canopy pixels inpainted",
            attrs=common_attrs,
        )
    return out


def _fill_layer(
    field: xr.DataArray,
    hole: np.ndarray,
    inpainter: Inpainter,
    vmin: float,
    vmax: float,
    *,
    name: str,
    long_name: str,
    attrs: dict,
) -> xr.DataArray:
    source = np.ascontiguousarray(field.values, dtype=np.float32)
    filled = np.asarray(inpainter(source, hole.copy(), float(vmin), float(vmax)), dtype=np.float32)
    if filled.shape != source.shape:
        raise ValueError(
            f"inpainter returned shape {filled.shape}, expected {source.shape}"
        )
    merged = np.where(hole, np.clip(filled, vmin, vmax), source).astype(np.float32)
    layer = xr.DataArray(
        merged, dims=("y", "x"), coords={"y": field["y"], "x": field["x"]}, name=name
    )
    layer.attrs = dict(field.attrs)
    layer.attrs.update(attrs)
    layer.attrs.update(
        long_name=long_name,
        source_variable=str(field.name),
        treefill_vmin=float(vmin),
        treefill_vmax=float(vmax),
        treefill_applied=1,
    )
    return layer
