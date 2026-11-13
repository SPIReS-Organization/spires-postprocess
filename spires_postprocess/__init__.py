"""Postprocessing for SPIReS inversion results."""

__version__ = "0.1.0"

from spires_postprocess.gate import PRODUCT_MIN_FSNOW, gate_products
from spires_postprocess.lookup import (
    load_albedo_lookup,
    load_forcing_lookup,
)
from spires_postprocess.masks import (
    canopy_cover_fraction,
    cloud_mask_from_valid,
    land_mask_from_tiles,
    tree_mask,
)
from spires_postprocess.process import process
from spires_postprocess.shade import shade_adjusted_fsnow, shade_saturated_mask
from spires_postprocess.snow_fraction import (
    apply_snow_fraction_adjustments,
    calculate_viewable_canopy_fraction,
)
from spires_postprocess.snow_radiative import (
    compute_delta_vis,
    compute_radiative_forcing,
    compute_snow_albedo,
)
from spires_postprocess.treefill import Inpainter, treefill_results

__all__ = [
    "__version__",
    "apply_snow_fraction_adjustments",
    "calculate_viewable_canopy_fraction",
    "canopy_cover_fraction",
    "cloud_mask_from_valid",
    "compute_delta_vis",
    "compute_radiative_forcing",
    "compute_snow_albedo",
    "gate_products",
    "Inpainter",
    "land_mask_from_tiles",
    "load_albedo_lookup",
    "load_forcing_lookup",
    "process",
    "PRODUCT_MIN_FSNOW",
    "shade_adjusted_fsnow",
    "shade_saturated_mask",
    "tree_mask",
    "treefill_results",
]
