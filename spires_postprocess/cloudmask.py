# spires_postprocess/cloudmask.py
"""Cloud and cloud-shadow classification with OmniCloudMask (Wright et al. 2025).

OmniCloudMask (OCM) is a sensor-agnostic CNN that labels each pixel of a Red,
Green, NIR image at 10-50 m as 0 clear, 1 thick cloud, 2 thin cloud or 3 cloud
shadow. It applies to Landsat, HLS and Sentinel-2 scenes, not to MODIS/VIIRS.

The core package never imports torch. ``Classifier`` is any callable
``(rgnir) -> classes`` taking a ``(3, y, x)`` uint16 array (reflectance x 10000,
0 = nodata) and returning ``(y, x)`` uint8 classes 0..3.
``omnicloudmask_classifier`` builds the in-process reference implementation
behind the ``cloudmask`` extra (``pip install 'spires-postprocess[cloudmask]'``);
a caller may substitute any other backend honouring the same contract (for
example a GPU service) and the rest of this module is unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import validate_target_layout
from spires_postprocess.masks import _require_plane

#: Common band names every spires-io 10-50 m spec exposes. On disk these are
#: Landsat SR_B4/B3/B5, HLS L30 B04/B03/B05, HLS S30 and Sentinel-2 B04/B03/B08.
RGNIR_BANDS: tuple[str, str, str] = ("red", "green", "nir")
#: OCM's own loaders feed integer DN at reflectance x 10000 and treat 0 as nodata.
REFLECTANCE_SCALE = 10000
NODATA_DN = 0
#: OCM zeroes its prediction where all bands are nodata, which is indistinguishable
#: from "clear"; we re-flag those pixels with this value.
NO_OBSERVATION = 255
CLASS_FLAG_VALUES = "0,1,2,3,255"
CLASS_FLAG_MEANINGS = "clear thick_cloud thin_cloud cloud_shadow no_observation"
#: Classes a cloud mask excludes.
CLOUD_CLASSES: tuple[int, ...] = (1, 2, 3)

#: Pinned reference implementation. The model version is OCM's own numbering.
OMNICLOUDMASK_VERSION = "1.7.1"
OMNICLOUDMASK_MODEL_VERSION = 4.0

#: ``(3, y, x)`` uint16 Red/Green/NIR -> ``(y, x)`` uint8 classes in 0..3.
Classifier = Callable[[np.ndarray], np.ndarray]


def rgnir_stack(reflectance: xr.DataArray, bands: tuple[str, str, str] = RGNIR_BANDS) -> np.ndarray:
    """``(3, y, x)`` uint16 Red, Green, NIR at reflectance x 10000 for OCM.

    A pixel with a non-finite value in any of the three bands becomes nodata (0)
    in all three, so OCM's all-bands-zero nodata rule and ours agree. Finite
    values are rounded and clipped to [1, 65535] so a genuine observation (even
    0.0 or a negative Landsat reflectance) is never read as nodata.
    """
    if not isinstance(reflectance, xr.DataArray) or "band" not in reflectance.dims:
        raise ValueError("reflectance must be a DataArray with a 'band' dimension")
    have = [str(b) for b in reflectance["band"].values]
    missing = [b for b in bands if b not in have]
    if missing:
        raise ValueError(f"reflectance lacks band(s) {missing}; has {have}")
    values = reflectance.sel(band=list(bands)).transpose("band", "y", "x").values.astype(np.float64)
    observed = np.isfinite(values).all(axis=0)
    dn = np.clip(np.rint(np.nan_to_num(values) * REFLECTANCE_SCALE), 1, np.iinfo(np.uint16).max)
    return np.where(observed[None, :, :], dn, NODATA_DN).astype(np.uint16)
