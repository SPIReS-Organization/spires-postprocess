"""Companion masks for the Landsat / Sentinel-2 / HLS product chain.

Every function returns a boolean ``(y, x)`` DataArray on the results grid. Masks
are True where the pixel is *excluded*, except :func:`land_mask_from_tiles`,
which is True for land because the product semantics are "keep land".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from spires_postprocess._xarray_validation import (
    require_matching_coords,
    validate_target_layout,
)

TREE_COVER_THRESHOLD = 0.10
TREE_DILATE_CELLS = 1
CANOPY_EDGE_BUFFER_M = 100.0
LAND_WATER_TILE_DEG = 30
LAND_WATER_EDGE_BUFFER_DEG = 0.01
LAND_WATER_LAND_VALUE = 1
_METRES_PER_DEGREE = 111_320.0


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


def canopy_cover_fraction(
    canopy_raster: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray:
    """Resample a binary or percent canopy raster to a cover fraction on ``target``.

    The raster is clipped to the target bounds plus a 100 m buffer before
    average-resampling so edge pixels get full averaging support. Percent
    rasters (finite maximum above 1) are divided by 100; the raster's declared
    nodata becomes NaN and is excluded from the average.
    """
    import rioxarray  # noqa: F401  registers the .rio accessor
    from rasterio.enums import Resampling
    from rioxarray.exceptions import NoDataInBounds

    validate_target_layout(target, "target")
    if target.dims != ("y", "x"):
        raise ValueError("target must have dimensions ('y', 'x')")
    if target.rio.crs is None:
        raise ValueError("target must carry a CRS (use .rio.write_crs)")
    if not isinstance(canopy_raster, xr.DataArray):
        raise TypeError("canopy_raster must be an xarray.DataArray")
    raster = canopy_raster.squeeze(drop=True)
    if raster.dims != ("y", "x"):
        raise ValueError(
            f"canopy_raster must reduce to dimensions ('y', 'x'); got {raster.dims}"
        )
    if raster.rio.crs is None:
        raise ValueError("canopy_raster must carry a CRS")

    minx, miny, maxx, maxy = target.rio.transform_bounds(raster.rio.crs)
    buffer = (
        CANOPY_EDGE_BUFFER_M / _METRES_PER_DEGREE
        if raster.rio.crs.is_geographic
        else CANOPY_EDGE_BUFFER_M
    )
    try:
        window = raster.rio.clip_box(
            minx - buffer, miny - buffer, maxx + buffer, maxy + buffer
        )
    except NoDataInBounds as error:
        raise ValueError("canopy_raster does not overlap the target grid") from error

    fraction = _canopy_as_fraction(window)
    matched = fraction.rio.reproject_match(
        target, resampling=Resampling.average, nodata=np.nan
    )
    out = matched.clip(0.0, 1.0).astype("float32")
    out = xr.DataArray(
        out.values,
        dims=("y", "x"),
        coords={"y": target["y"], "x": target["x"]},
        name="canopy_fraction",
    )
    out.attrs = {
        "long_name": "Canopy cover fraction",
        "units": "1",
        "source": str(canopy_raster.attrs.get("source", "canopy_raster")),
        "resampling": "average",
        "edge_buffer_m": CANOPY_EDGE_BUFFER_M,
    }
    return out


def _canopy_as_fraction(window: xr.DataArray) -> xr.DataArray:
    values = window.astype("float32")
    nodata = window.rio.nodata
    if nodata is not None:
        values = values.where(window != nodata)
    finite_max = float(values.max(skipna=True))
    if np.isfinite(finite_max) and finite_max > 1.0:
        values = values.where(values <= 100.0) / np.float32(100.0)
    values = values.rio.write_crs(window.rio.crs)
    return values.rio.write_nodata(np.nan, encoded=False)


def tree_mask(
    cover_fraction: xr.DataArray,
    *,
    threshold: float = TREE_COVER_THRESHOLD,
    dilate_cells: int = TREE_DILATE_CELLS,
) -> xr.DataArray:
    """Pixels with cover above ``threshold``, dilated by ``dilate_cells``.

    NaN cover is treated as no tree. Dilation uses a full ``(2k+1)**2`` square.
    """
    validate_target_layout(cover_fraction, "cover_fraction")
    if cover_fraction.dims != ("y", "x"):
        raise ValueError("cover_fraction must have dimensions ('y', 'x')")
    if dilate_cells < 0:
        raise ValueError("dilate_cells must be >= 0")
    core = np.asarray(cover_fraction.values > threshold, dtype=bool)
    if dilate_cells > 0:
        size = 2 * dilate_cells + 1
        core = ndimage.binary_dilation(core, structure=np.ones((size, size), dtype=bool))
    mask = xr.DataArray(
        core,
        dims=("y", "x"),
        coords={"y": cover_fraction["y"], "x": cover_fraction["x"]},
        name="tree_mask",
    )
    mask.attrs = {
        "long_name": "Tree mask from canopy cover (True = excluded)",
        "cover_threshold": float(threshold),
        "comparison": "strictly_greater_than",
        "dilate_cells": int(dilate_cells),
        "source": str(cover_fraction.attrs.get("source", "canopy_fraction")),
    }
    return mask


def land_water_tile_path(
    bounds_4326: tuple[float, float, float, float],
    directory: str | Path,
) -> Path:
    """Return the single ``landWater2020_{lon:04d}_{lat:03d}.tif`` tile covering bounds.

    Tiles are 30-degree EPSG:4326 squares named by their south-west corner.
    """
    minx, miny, maxx, maxy = bounds_4326
    lons = _tile_origins(minx, maxx)
    lats = _tile_origins(miny, maxy)
    if len(lons) != 1 or len(lats) != 1:
        raise ValueError(
            f"target bounds {bounds_4326} span multiple land/water tiles "
            f"(lon origins {lons}, lat origins {lats}); split the target"
        )
    path = Path(directory) / f"landWater2020_{lons[0]:04d}_{lats[0]:03d}.tif"
    if not path.is_file():
        raise FileNotFoundError(f"land/water tile is not staged: {path}")
    return path


def _tile_origins(low: float, high: float) -> list[int]:
    first = math.floor(low / LAND_WATER_TILE_DEG) * LAND_WATER_TILE_DEG
    last = math.floor(high / LAND_WATER_TILE_DEG) * LAND_WATER_TILE_DEG
    return list(range(first, last + 1, LAND_WATER_TILE_DEG))


def land_mask_from_tiles(target: xr.DataArray, directory: str | Path) -> xr.DataArray:
    """True where the land/water tile says LAND, nearest-resampled onto ``target``.

    ``directory`` has no default and no environment fallback on purpose: an
    unnoticed absent mask is the failure this function exists to prevent.
    """
    import rioxarray  # noqa: F401

    validate_target_layout(target, "target")
    if target.dims != ("y", "x"):
        raise ValueError("target must have dimensions ('y', 'x')")
    if target.rio.crs is None:
        raise ValueError("target must carry a CRS (use .rio.write_crs)")
    bounds = tuple(float(b) for b in target.rio.transform_bounds("EPSG:4326"))
    path = land_water_tile_path(bounds, directory)
    minx, miny, maxx, maxy = bounds
    with rioxarray.open_rasterio(path) as tile:
        tile = tile.squeeze(drop=True)
        window = tile.rio.clip_box(
            minx - LAND_WATER_EDGE_BUFFER_DEG,
            miny - LAND_WATER_EDGE_BUFFER_DEG,
            maxx + LAND_WATER_EDGE_BUFFER_DEG,
            maxy + LAND_WATER_EDGE_BUFFER_DEG,
        ).load()
        crs = tile.rio.crs
    window = window.rio.write_crs(crs)
    # Nearest (rioxarray default) is the only correct choice for a categorical mask.
    matched = window.rio.reproject_match(target)
    mask = xr.DataArray(
        np.asarray(matched.values == LAND_WATER_LAND_VALUE, dtype=bool),
        dims=("y", "x"),
        coords={"y": target["y"], "x": target["x"]},
        name="land_mask",
    )
    mask.attrs = {
        "long_name": "Land mask (True = land, False = water)",
        "source": "landWater2020",
        "tile": path.name,
        "resampling": "nearest",
    }
    return mask
