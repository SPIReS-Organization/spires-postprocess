"""Postprocessing for SPIReS inversion results."""

__version__ = "0.1.0"

from spires_postprocess.lookup import (
    load_albedo_lookup,
    load_forcing_lookup,
)
from spires_postprocess.process import process
from spires_postprocess.snow_fraction import (
    apply_snow_fraction_adjustments,
    calculate_viewable_canopy_fraction,
)
from spires_postprocess.snow_radiative import (
    compute_delta_vis,
    compute_radiative_forcing,
    compute_snow_albedo,
)
from spires_postprocess.temporal import DEFAULT_MAX_GAP, fill_time_gaps

__all__ = [
    "__version__",
    "DEFAULT_MAX_GAP",
    "apply_snow_fraction_adjustments",
    "calculate_viewable_canopy_fraction",
    "compute_delta_vis",
    "compute_radiative_forcing",
    "compute_snow_albedo",
    "fill_time_gaps",
    "load_albedo_lookup",
    "load_forcing_lookup",
    "process",
]
