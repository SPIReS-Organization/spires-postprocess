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


def classify_clouds(reflectance: xr.DataArray, *, classifier: Classifier | None = None) -> xr.DataArray:
    """Per-pixel OCM classes on the reflectance grid as uint8 ``(y, x)``.

    Values: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow, 255 no
    observation (a band was non-finite). ``classifier`` defaults to the
    in-process OmniCloudMask backend, which needs the ``cloudmask`` extra.
    Provenance attrs are copied from ``classifier.provenance`` when present.
    """
    if classifier is None:
        classifier = omnicloudmask_classifier()
    rgnir = rgnir_stack(reflectance)
    classes = np.asarray(classifier(rgnir))
    if classes.shape != rgnir.shape[1:]:
        raise ValueError(f"classifier returned shape {classes.shape}; expected {rgnir.shape[1:]}")
    if classes.dtype != np.uint8:
        raise ValueError(f"classifier must return uint8 classes; got {classes.dtype}")
    if not np.isin(classes, (0, 1, 2, 3)).all():
        raise ValueError("classifier must return classes in 0..3")
    classes = classes.copy()
    classes[(rgnir == NODATA_DN).all(axis=0)] = NO_OBSERVATION

    coords = {k: v for k, v in reflectance.coords.items() if set(v.dims) <= {"y", "x"}}
    out = xr.DataArray(classes, dims=("y", "x"), coords=coords, name="cloud_classes")
    out.attrs = {
        "long_name": "OmniCloudMask cloud and cloud shadow classes",
        "flag_values": CLASS_FLAG_VALUES,
        "flag_meanings": CLASS_FLAG_MEANINGS,
        "input_bands": ",".join(RGNIR_BANDS),
        "input_scaling": f"reflectance x {REFLECTANCE_SCALE} as uint16, {NODATA_DN} = nodata",
    }
    out.attrs.update({str(k): str(v) for k, v in dict(getattr(classifier, "provenance", {})).items()})
    return out


def cloud_mask_from_classes(
    classes: xr.DataArray,
    target: xr.DataArray,
    *,
    cloud_classes: tuple[int, ...] = CLOUD_CLASSES,
    exclude_no_observation: bool = True,
) -> xr.DataArray:
    """Collapse OCM classes to the boolean ``cloud_mask`` ``process()`` accepts (True = excluded)."""
    validate_target_layout(target, "target")
    if not isinstance(classes, xr.DataArray) or classes.dtype != np.dtype(np.uint8):
        raise ValueError("classes must be a uint8 DataArray from classify_clouds")
    _require_plane(classes, target, "classes")
    excluded = np.isin(classes.values, cloud_classes)
    if exclude_no_observation:
        excluded |= classes.values == NO_OBSERVATION
    mask = xr.DataArray(excluded, dims=("y", "x"), coords=classes.coords, name="cloud_mask")
    mask.attrs = {
        "long_name": "Cloud or invalid observation mask (True = excluded)",
        "source": "spires_postprocess.cloudmask",
        "excluded_classes": ",".join(map(str, cloud_classes)) + (",255" if exclude_no_observation else ""),
    }
    for key in ("cloud_model", "cloud_model_version", "omnicloudmask_version", "cloud_backend"):
        if key in classes.attrs:
            mask.attrs[key] = classes.attrs[key]
    return mask


@dataclass(frozen=True)
class OmniCloudMaskClassifier:
    """Reference backend: ``omnicloudmask.predict_from_array`` with upstream defaults.

    ``inference_device`` is a concrete torch device string; use
    ``omnicloudmask_classifier()`` to pick the GPU automatically when one is
    present. The resolved device is recorded in ``provenance``.
    """

    inference_device: str
    patch_size: int = 1000
    patch_overlap: int = 300
    batch_size: int = 1
    model_version: float = OMNICLOUDMASK_MODEL_VERSION

    @property
    def provenance(self) -> dict[str, str]:
        import omnicloudmask

        return {
            "cloud_model": "OmniCloudMask",
            "cloud_model_version": str(self.model_version),
            "omnicloudmask_version": str(omnicloudmask.__version__),
            "cloud_backend": f"omnicloudmask.predict_from_array[{self.inference_device}]",
        }

    def __call__(self, rgnir: np.ndarray) -> np.ndarray:
        import omnicloudmask

        pred = omnicloudmask.predict_from_array(
            np.ascontiguousarray(rgnir),
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            batch_size=self.batch_size,
            inference_device=self.inference_device,
            mosaic_device=self.inference_device,
            no_data_value=NODATA_DN,
            apply_no_data_mask=True,
            model_version=self.model_version,
        )
        return np.asarray(pred, dtype=np.uint8)[0]


def default_inference_device() -> str:
    """``"cuda"`` when a GPU is visible to torch, else ``"cpu"``."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def omnicloudmask_classifier(inference_device: str | None = None, **options) -> OmniCloudMaskClassifier:
    """Build the in-process OmniCloudMask backend; needs ``pip install 'spires-postprocess[cloudmask]'``.

    Runs on the GPU when torch sees one, otherwise on the CPU; pass
    ``inference_device`` to force either.
    """
    try:
        import omnicloudmask
    except ImportError as exc:
        raise ImportError(
            "the OmniCloudMask backend needs the optional 'cloudmask' extra: "
            "pip install 'spires-postprocess[cloudmask]'"
        ) from exc
    installed = str(getattr(omnicloudmask, "__version__", "unknown"))
    if installed != OMNICLOUDMASK_VERSION:
        warnings.warn(
            f"omnicloudmask {installed} installed; spires-postprocess pins {OMNICLOUDMASK_VERSION} "
            "for reproducible masks",
            UserWarning,
            stacklevel=2,
        )
    if inference_device is None:
        inference_device = default_inference_device()
    return OmniCloudMaskClassifier(inference_device=inference_device, **options)
