"""Canopy cover from the Meta canopy-height model (CHM v2) via icechunk.

The CHM is a global ~1.19 m canopy-height raster in EPSG:3857, published as a
virtual icechunk store on Source Cooperative. :func:`open_meta_chm` returns it
lazily; :func:`canopy_raster_from_chm` cuts a window around a target grid and
thresholds height to a binary tree raster that
:func:`spires_postprocess.masks.canopy_cover_fraction` averages onto the grid
exactly as it does for a local binary raster.

Needs the ``chm`` extra: ``pip install 'spires-postprocess[chm]'``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import icechunk
except ImportError as exc:  # pragma: no cover - exercised by the import-guard test
    raise ImportError(
        "spires_postprocess.canopy_chm needs icechunk; install the extra: "
        "pip install 'spires-postprocess[chm]'"
    ) from exc

from spires_postprocess._xarray_validation import validate_target_layout
from spires_postprocess.masks import CANOPY_EDGE_BUFFER_M

CHM_CONTAINER_PREFIX = "s3://us-west-2.opendata.source.coop/"
CHM_BUCKET = "us-west-2.opendata.source.coop"
CHM_REPO_PREFIX = "tge-labs/meta-chm-v2/zarr/chm.zarr.icechunk"
CHM_REGION = "us-west-2"
CHM_GROUP = "1x"
CHM_VARIABLE = "chm"
CHM_CRS = "EPSG:3857"

TREE_HEIGHT_THRESHOLD_M = 1.0

# The published store's config disables retries (max_tries == 1). A tile-scale
# window is tens of thousands of chunk fetches, so a dropped connection is a
# certainty, not a risk; these are set explicitly on every open.
CHM_MAX_TRIES = 10
CHM_INITIAL_BACKOFF_MS = 500
CHM_MAX_BACKOFF_MS = 30_000

# Refuse windows larger than this (pixels) rather than start a multi-hour fetch.
# ~2 Gpx is roughly a 20 km x 20 km ROI at 1.19 m; a full MGRS tile is ~13.5 Gpx.
MAX_WINDOW_PIXELS = 2_000_000_000


def canopy_raster_from_chm(
    chm: xr.DataArray,
    target: xr.DataArray,
    *,
    height_threshold_m: float = TREE_HEIGHT_THRESHOLD_M,
) -> xr.DataArray:
    """Window ``chm`` around ``target`` and threshold height into a binary tree raster.

    ``chm`` is the (possibly lazy) native-resolution canopy-height layer from
    :func:`open_meta_chm`, or any georeferenced ``(y, x)`` height raster. The
    window is the target's bounds in the CHM's CRS plus
    :data:`~spires_postprocess.masks.CANOPY_EDGE_BUFFER_M`, so the subsequent
    average has full support at the edges. Pixels strictly higher than
    ``height_threshold_m`` are 1, all others 0. The result is what
    :func:`spires_postprocess.masks.canopy_cover_fraction` expects with
    ``percent=False``.

    Raises ``ValueError`` if the window would exceed :data:`MAX_WINDOW_PIXELS`
    (a whole MGRS tile is ~13.5 Gpx of CHM and tens of thousands of chunk
    fetches; raise the limit deliberately rather than discover it hours in).
    """
    import rioxarray  # noqa: F401  registers the .rio accessor
    from rioxarray.exceptions import NoDataInBounds

    validate_target_layout(target, "target")
    if target.dims != ("y", "x"):
        raise ValueError("target must have dimensions ('y', 'x')")
    if target.rio.crs is None:
        raise ValueError("target must carry a CRS (use .rio.write_crs)")
    if not isinstance(chm, xr.DataArray) or chm.dims != ("y", "x"):
        raise ValueError("chm must be a (y, x) xarray.DataArray")
    if chm.rio.crs is None:
        raise ValueError("chm must carry a CRS")

    minx, miny, maxx, maxy = target.rio.transform_bounds(chm.rio.crs)
    buffer = CANOPY_EDGE_BUFFER_M
    try:
        window = chm.rio.clip_box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    except NoDataInBounds as error:
        raise ValueError("target does not overlap the canopy-height raster") from error

    pixels = int(window.sizes["y"]) * int(window.sizes["x"])
    if pixels > MAX_WINDOW_PIXELS:
        raise ValueError(
            f"CHM window is {pixels:,} pixels ({window.sizes['y']} x {window.sizes['x']}); "
            f"the limit is {MAX_WINDOW_PIXELS:,}. Raise canopy_chm.MAX_WINDOW_PIXELS "
            "deliberately for tile-scale runs."
        )

    heights = window.compute()
    is_tree = (heights > height_threshold_m).astype("uint8")
    out = xr.DataArray(
        is_tree.values,
        dims=("y", "x"),
        coords={"y": heights["y"], "x": heights["x"]},
        name="canopy_binary",
    )
    out = out.rio.write_crs(heights.rio.crs)
    out.attrs = {
        "long_name": "Tree presence from canopy height",
        "source": "meta-chm-v2",
        "height_threshold_m": float(height_threshold_m),
    }
    return out
