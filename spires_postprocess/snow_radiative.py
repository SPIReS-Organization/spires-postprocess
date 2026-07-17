"""LUT-based snow albedo, delta-VIS, and radiative-forcing products."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    prepare_aligned_layer,
    require_matching_coords,
    require_real_numeric,
    validate_target_layout,
)
from spires_postprocess.lookup import (
    COSINE_ILLUMINATION,
    COSINE_SOLAR_ZENITH,
    DUST_MASS_FRACTION,
    SOOT_MASS_FRACTION,
    SQRT_GRAIN_RADIUS_UM,
    LookupTable,
    interpolate_lookup,
    lookup_axis_ranges,
    validate_lookup_table,
)


_GRAIN_RADIUS_UNITS = {"um", "µm", "μm", "micrometer", "micrometers"}
_DUST_PPM_UNITS = {"ppm", "parts per million"}


def compute_snow_albedo(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    cosine_illumination: xr.DataArray,
    lookup: xr.Dataset,
) -> xr.Dataset:
    """Add four clean/dirty flat/terrain-corrected snow albedo products.

    ``results['grain_size']`` is effective snow grain radius in micrometers.
    ``results['dust_concentration']`` is parts per million and is divided by
    1,000,000 for lookup evaluation. Soot mass fraction is fixed to zero.
    Missing or out-of-domain pixels return NaN; lookup extrapolation is never
    performed. The caller's dataset is not modified.
    """
    grain_radius, dust_ppm = _validated_inversion_inputs(results)
    mu0 = _prepare_geometry(
        cosine_solar_zenith,
        grain_radius,
        name=COSINE_SOLAR_ZENITH,
    )
    mu_i = _prepare_geometry(
        cosine_illumination,
        grain_radius,
        name=COSINE_ILLUMINATION,
    )
    clean_table = validate_lookup_table(lookup, "clean_albedo")
    dirty_table = validate_lookup_table(lookup, "dirty_albedo")

    transformed = _transformed_inputs(grain_radius, dust_ppm)
    clean_flat = _evaluate(
        clean_table,
        {
            COSINE_SOLAR_ZENITH: mu0,
            COSINE_ILLUMINATION: mu0,
            SQRT_GRAIN_RADIUS_UM: transformed[SQRT_GRAIN_RADIUS_UM],
        },
        grain_radius,
    ).clip(min=0.0, max=1.0)
    dirty_flat = _evaluate(
        dirty_table,
        {
            COSINE_SOLAR_ZENITH: mu0,
            COSINE_ILLUMINATION: mu0,
            **transformed,
        },
        grain_radius,
    ).clip(min=0.0, max=1.0)
    clean_terrain = _evaluate(
        clean_table,
        {
            COSINE_SOLAR_ZENITH: mu0,
            COSINE_ILLUMINATION: mu_i,
            SQRT_GRAIN_RADIUS_UM: transformed[SQRT_GRAIN_RADIUS_UM],
        },
        grain_radius,
    ).clip(min=0.0, max=1.0)
    dirty_terrain = _evaluate(
        dirty_table,
        {
            COSINE_SOLAR_ZENITH: mu0,
            COSINE_ILLUMINATION: mu_i,
            **transformed,
        },
        grain_radius,
    ).clip(min=0.0, max=1.0)

    updated = results.copy()
    products = (
        (
            "albedo_clean_flat",
            clean_flat,
            clean_table,
            "Clean-snow albedo for flat geometry",
            "flat",
        ),
        (
            "albedo_dirty_flat",
            dirty_flat,
            dirty_table,
            "Dust-affected snow albedo for flat geometry",
            "flat",
        ),
        (
            "albedo_clean_terrain_corrected",
            clean_terrain,
            clean_table,
            "Clean-snow albedo corrected for local terrain illumination",
            "terrain_corrected",
        ),
        (
            "albedo_dirty_terrain_corrected",
            dirty_terrain,
            dirty_table,
            "Dust-affected snow albedo corrected for local terrain illumination",
            "terrain_corrected",
        ),
    )
    for name, values, table, long_name, geometry in products:
        updated[name] = _with_product_metadata(
            values,
            name=name,
            long_name=long_name,
            units="1",
            geometry=geometry,
            table=table,
        )
    return updated


def compute_delta_vis(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    lookup: xr.Dataset,
) -> xr.Dataset:
    """Add flat-surface dimensionless delta-VIS from its independent LUT.

    Grain size is effective snow grain radius in micrometers; dust
    concentration is ppm and is divided by 1,000,000. Soot is fixed to zero.
    Missing or out-of-domain required inputs produce NaN pixels.
    """
    return _compute_flat_dirty_product(
        results,
        cosine_solar_zenith=cosine_solar_zenith,
        lookup=lookup,
        table_name="delta_vis",
        product_name="delta_vis",
        long_name="Visible albedo reduction due to snow impurities",
        units="1",
    )


def compute_radiative_forcing(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    lookup: xr.Dataset,
) -> xr.Dataset:
    """Add flat-surface snow radiative forcing in W m-2 from its LUT.

    Grain size is effective snow grain radius in micrometers; dust
    concentration is ppm and is divided by 1,000,000. Soot is fixed to zero.
    Missing or out-of-domain required inputs produce NaN pixels.
    """
    return _compute_flat_dirty_product(
        results,
        cosine_solar_zenith=cosine_solar_zenith,
        lookup=lookup,
        table_name="radiative_forcing",
        product_name="radiative_forcing",
        long_name="Snow radiative forcing due to impurities",
        units="W m-2",
    )


def _compute_flat_dirty_product(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    lookup: xr.Dataset,
    table_name: str,
    product_name: str,
    long_name: str,
    units: str,
) -> xr.Dataset:
    grain_radius, dust_ppm = _validated_inversion_inputs(results)
    mu0 = _prepare_geometry(
        cosine_solar_zenith,
        grain_radius,
        name=COSINE_SOLAR_ZENITH,
    )
    table = validate_lookup_table(lookup, table_name)
    transformed = _transformed_inputs(grain_radius, dust_ppm)
    values = _evaluate(
        table,
        {
            COSINE_SOLAR_ZENITH: mu0,
            COSINE_ILLUMINATION: mu0,
            **transformed,
        },
        grain_radius,
    )
    updated = results.copy()
    updated[product_name] = _with_product_metadata(
        values,
        name=product_name,
        long_name=long_name,
        units=units,
        geometry="flat",
        table=table,
    )
    return updated


def _validated_inversion_inputs(
    results: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    if not isinstance(results, xr.Dataset):
        raise TypeError("results must be an xarray.Dataset")
    missing = [
        name for name in ("grain_size", "dust_concentration") if name not in results
    ]
    if missing:
        raise ValueError(f"results is missing required variable(s): {missing}")

    grain_radius = results["grain_size"]
    dust_ppm = results["dust_concentration"]
    validate_target_layout(grain_radius, "results['grain_size']")
    require_real_numeric(grain_radius, "results['grain_size']")
    require_real_numeric(dust_ppm, "results['dust_concentration']")
    if dust_ppm.dims != grain_radius.dims:
        raise ValueError(
            "results['dust_concentration'] must have the same dimensions as "
            "results['grain_size']"
        )
    require_matching_coords(
        dust_ppm,
        grain_radius,
        grain_radius.dims,
        label="results['dust_concentration']",
        target_label="results['grain_size']",
    )
    _validate_units(
        grain_radius,
        allowed=_GRAIN_RADIUS_UNITS,
        label="results['grain_size']",
        convention="effective snow grain radius in micrometers",
    )
    _validate_units(
        dust_ppm,
        allowed=_DUST_PPM_UNITS,
        label="results['dust_concentration']",
        convention="parts per million",
    )
    return grain_radius, dust_ppm


def _prepare_geometry(
    geometry: xr.DataArray,
    target: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    return prepare_aligned_layer(
        geometry,
        target,
        label=name,
        target_label="results['grain_size']",
        require_numeric=True,
    )


def _transformed_inputs(
    grain_radius: xr.DataArray,
    dust_ppm: xr.DataArray,
) -> dict[str, xr.DataArray]:
    sqrt_grain_radius = np.sqrt(grain_radius.where(grain_radius >= 0.0))
    dust_mass_fraction = dust_ppm / 1_000_000.0
    return {
        SQRT_GRAIN_RADIUS_UM: sqrt_grain_radius,
        DUST_MASS_FRACTION: dust_mass_fraction,
        SOOT_MASS_FRACTION: xr.zeros_like(grain_radius, dtype=np.float64),
    }


def _evaluate(
    table: LookupTable,
    inputs: Mapping[str, xr.DataArray],
    target: xr.DataArray,
) -> xr.DataArray:
    return interpolate_lookup(table, inputs).transpose(*target.dims).astype("float32")


def _with_product_metadata(
    values: xr.DataArray,
    *,
    name: str,
    long_name: str,
    units: str,
    geometry: str,
    table: LookupTable,
) -> xr.DataArray:
    product = values.astype("float32").rename(name)
    attrs: dict[str, object] = {
        "long_name": long_name,
        "units": units,
        "model": "SPIReS normalized LUT linear interpolation",
        "lookup_variable": table.name,
        "lookup_axis_ranges": lookup_axis_ranges(table),
        "grain_size_interpretation": "effective snow grain radius",
        "grain_size_units": "um",
        "grain_size_transform": "sqrt(radius_um)",
        "dust_input_units": "ppm",
        "dust_transform": "dust_ppm / 1000000",
        "soot_mass_fraction": 0.0,
        "geometry": geometry,
    }
    if "source" in table.provenance:
        attrs["lookup_source"] = table.provenance["source"]
    if "source_checksum_sha256" in table.provenance:
        attrs["lookup_source_checksum_sha256"] = table.provenance[
            "source_checksum_sha256"
        ]
    for name in ("atmosphere_model", "dust_model"):
        if name in table.provenance:
            attrs[name] = table.provenance[name]
    product.attrs = attrs
    return product


def _validate_units(
    data: xr.DataArray,
    *,
    allowed: set[str],
    label: str,
    convention: str,
) -> None:
    declared = data.attrs.get("units")
    if declared is None or declared == "":
        return
    if not isinstance(declared, str) or declared.strip().casefold() not in allowed:
        raise ValueError(
            f"{label} units must identify {convention}; got {declared!r}"
        )
