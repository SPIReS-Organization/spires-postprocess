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
