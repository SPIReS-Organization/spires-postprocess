"""Postprocessing for SPIReS inversion results."""

__version__ = "0.1.0"

from spires_postprocess.snow_fraction import (
    apply_snow_fraction_adjustments,
    calculate_viewable_canopy_fraction,
)

__all__ = [
    "__version__",
    "apply_snow_fraction_adjustments",
    "calculate_viewable_canopy_fraction",
]
