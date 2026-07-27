"""LUT-based snow albedo, delta-VIS, and radiative-forcing products."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    prepare_aligned_layer,
    require_float32,
    require_matching_coords,
    require_values_in_range,
    validate_target_layout,
)
from spires_postprocess.lookup import (
    ALTITUDE,
    ILLUMINATION_ANGLE,
    LAP_CONCENTRATION,
    SKYVIEW,
    SOLAR_ZENITH,
    SQRT_GRAIN_RADIUS,
    LookupTable,
    interpolate_lookup,
    lookup_axis_ranges,
    validate_lookup_table,
)


_GRAIN_RADIUS_UNITS = {"um", "µm", "μm", "micrometer", "micrometers"}
_LAP_PPM_UNITS = {"ppm", "parts per million"}
_ALTITUDE_KM_UNITS = {"km", "kilometer", "kilometers", "kilometre", "kilometres"}


def compute_snow_albedo(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    cosine_illumination: xr.DataArray,
    lookup: xr.Dataset,
    skyview: xr.DataArray | None = None,
    altitude: xr.DataArray | None = None,
) -> xr.Dataset:
    """Add clean/dirty flat/terrain-corrected snow albedo products.

    The canonical lookup contains one ``albedo`` variable. Clean products are
    evaluated at zero LAP concentration, while dirty products use the retrieved
    ``lap_concentration``. Geometry cosines are explicitly converted to the
    lookup's degree-valued axes. Optional sky-view and altitude inputs are
    required only when those axes are present in the lookup.
    """
    grain_radius, lap_ppm = _validated_inversion_inputs(results)
    solar_zenith = _angle_from_cosine(
        cosine_solar_zenith,
        grain_radius,
        name="cosine_solar_zenith",
    )
    illumination_angle = _angle_from_cosine(
        cosine_illumination,
        grain_radius,
        name="cosine_illumination",
    )
    table = validate_lookup_table(lookup, "albedo")
    _require_zero_lap(table)

    clean_lap = xr.zeros_like(lap_ppm, dtype=np.float32)
    clean_flat = _evaluate(
        table,
        _lookup_inputs(
            table,
            grain_radius=grain_radius,
            lap_concentration=clean_lap,
            solar_zenith=solar_zenith,
            illumination_angle=solar_zenith,
            skyview=skyview,
            altitude=altitude,
        ),
        grain_radius,
    ).clip(min=0.0, max=1.0)
    dirty_flat = _evaluate(
        table,
        _lookup_inputs(
            table,
            grain_radius=grain_radius,
            lap_concentration=lap_ppm,
            solar_zenith=solar_zenith,
            illumination_angle=solar_zenith,
            skyview=skyview,
            altitude=altitude,
        ),
        grain_radius,
    ).clip(min=0.0, max=1.0)
    clean_terrain = _evaluate(
        table,
        _lookup_inputs(
            table,
            grain_radius=grain_radius,
            lap_concentration=clean_lap,
            solar_zenith=solar_zenith,
            illumination_angle=illumination_angle,
            skyview=skyview,
            altitude=altitude,
        ),
        grain_radius,
    ).clip(min=0.0, max=1.0)
    dirty_terrain = _evaluate(
        table,
        _lookup_inputs(
            table,
            grain_radius=grain_radius,
            lap_concentration=lap_ppm,
            solar_zenith=solar_zenith,
            illumination_angle=illumination_angle,
            skyview=skyview,
            altitude=altitude,
        ),
        grain_radius,
    ).clip(min=0.0, max=1.0)

    updated = results.copy()
    products = (
        (
            "albedo_clean_flat",
            clean_flat,
            "Clean-snow albedo for flat geometry",
            "flat",
            "zero",
        ),
        (
            "albedo_dirty_flat",
            dirty_flat,
            "Dust-affected snow albedo for flat geometry",
            "flat",
            "retrieved",
        ),
        (
            "albedo_clean_terrain_corrected",
            clean_terrain,
            "Clean-snow albedo corrected for local terrain illumination",
            "terrain_corrected",
            "zero",
        ),
        (
            "albedo_dirty_terrain_corrected",
            dirty_terrain,
            "Dust-affected snow albedo corrected for local terrain illumination",
            "terrain_corrected",
            "retrieved",
        ),
    )
    for name, values, long_name, geometry, lap_evaluation in products:
        updated[name] = _with_product_metadata(
            values,
            name=name,
            long_name=long_name,
            units="1",
            geometry=geometry,
            lap_evaluation=lap_evaluation,
            table=table,
        )
    return updated


def compute_delta_vis(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    lookup: xr.Dataset,
) -> xr.Dataset:
    """Add flat-surface visible albedo reduction from its canonical LUT."""
    return _compute_flat_dust_product(
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
    """Add flat-surface snow radiative forcing from its canonical LUT."""
    return _compute_flat_dust_product(
        results,
        cosine_solar_zenith=cosine_solar_zenith,
        lookup=lookup,
        table_name="radiative_forcing",
        product_name="radiative_forcing",
        long_name="Snow radiative forcing due to impurities",
        units="W m-2",
    )


def _compute_flat_dust_product(
    results: xr.Dataset,
    *,
    cosine_solar_zenith: xr.DataArray,
    lookup: xr.Dataset,
    table_name: str,
    product_name: str,
    long_name: str,
    units: str,
) -> xr.Dataset:
    grain_radius, lap_ppm = _validated_inversion_inputs(results)
    solar_zenith = _angle_from_cosine(
        cosine_solar_zenith,
        grain_radius,
        name="cosine_solar_zenith",
    )
    table = validate_lookup_table(lookup, table_name)
    values = _evaluate(
        table,
        _lookup_inputs(
            table,
            grain_radius=grain_radius,
            lap_concentration=lap_ppm,
            solar_zenith=solar_zenith,
            illumination_angle=solar_zenith,
        ),
        grain_radius,
    )
    updated = results.copy()
    updated[product_name] = _with_product_metadata(
        values,
        name=product_name,
        long_name=long_name,
        units=units,
        geometry="flat",
        lap_evaluation="retrieved",
        table=table,
    )
    return updated


def _validated_inversion_inputs(
    results: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    if not isinstance(results, xr.Dataset):
        raise TypeError("results must be an xarray.Dataset")
    missing = [
        name for name in ("grain_radius", "lap_concentration") if name not in results
    ]
    if missing:
        raise ValueError(f"results is missing required variable(s): {missing}")

    grain_radius = results["grain_radius"]
    lap_ppm = results["lap_concentration"]
    validate_target_layout(grain_radius, "results['grain_radius']")
    require_float32(grain_radius, "results['grain_radius']")
    require_float32(lap_ppm, "results['lap_concentration']")
    require_values_in_range(
        grain_radius,
        "results['grain_radius']",
        minimum=0.0,
    )
    require_values_in_range(
        lap_ppm,
        "results['lap_concentration']",
        minimum=0.0,
    )
    if lap_ppm.dims != grain_radius.dims:
        raise ValueError(
            "results['lap_concentration'] must have the same dimensions as "
            "results['grain_radius']"
        )
    require_matching_coords(
        lap_ppm,
        grain_radius,
        grain_radius.dims,
        label="results['lap_concentration']",
        target_label="results['grain_radius']",
    )
    _require_units(
        grain_radius,
        allowed=_GRAIN_RADIUS_UNITS,
        label="results['grain_radius']",
        convention="effective snow grain radius in micrometers",
    )
    _require_units(
        lap_ppm,
        allowed=_LAP_PPM_UNITS,
        label="results['lap_concentration']",
        convention="parts per million",
    )
    if lap_ppm.attrs.get("lap_type") != "dust":
        raise ValueError(
            "results['lap_concentration'] must declare lap_type='dust' "
            "for dust-specific postprocessing"
        )
    return grain_radius, lap_ppm


def _angle_from_cosine(
    cosine: xr.DataArray,
    target: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    prepared = prepare_aligned_layer(
        cosine,
        target,
        label=name,
        target_label="results['grain_radius']",
    )
    require_float32(prepared, name)
    require_values_in_range(prepared, name, minimum=0.0, maximum=1.0)
    angle = np.rad2deg(np.arccos(prepared))
    return angle.transpose(*target.dims).astype("float32")


def _lookup_inputs(
    table: LookupTable,
    *,
    grain_radius: xr.DataArray,
    lap_concentration: xr.DataArray,
    solar_zenith: xr.DataArray,
    illumination_angle: xr.DataArray,
    skyview: xr.DataArray | None = None,
    altitude: xr.DataArray | None = None,
) -> dict[str, xr.DataArray]:
    inputs = {
        SOLAR_ZENITH: solar_zenith,
        ILLUMINATION_ANGLE: illumination_angle,
        LAP_CONCENTRATION: lap_concentration,
        SQRT_GRAIN_RADIUS: np.sqrt(grain_radius).astype("float32"),
    }
    if SKYVIEW in table.dimensions:
        if skyview is None:
            raise ValueError(
                "lookup includes a 'skyview' axis, so skyview must be provided"
            )
        prepared_skyview = prepare_aligned_layer(
            skyview,
            grain_radius,
            label="skyview",
            target_label="results['grain_radius']",
        )
        require_float32(prepared_skyview, "skyview")
        require_values_in_range(
            prepared_skyview,
            "skyview",
            minimum=0.0,
            maximum=1.0,
        )
        inputs[SKYVIEW] = prepared_skyview

    if ALTITUDE in table.dimensions:
        if altitude is None:
            raise ValueError(
                "lookup includes an 'altitude' axis, so altitude must be provided"
            )
        prepared_altitude = prepare_aligned_layer(
            altitude,
            grain_radius,
            label="altitude",
            target_label="results['grain_radius']",
        )
        require_float32(prepared_altitude, "altitude")
        require_values_in_range(prepared_altitude, "altitude")
        _require_units(
            prepared_altitude,
            allowed=_ALTITUDE_KM_UNITS,
            label="altitude",
            convention="physical altitude in kilometres",
        )
        inputs[ALTITUDE] = prepared_altitude
    return inputs


def _evaluate(
    table: LookupTable,
    inputs: Mapping[str, xr.DataArray],
    target: xr.DataArray,
) -> xr.DataArray:
    return interpolate_lookup(table, inputs).transpose(*target.dims).astype("float32")


def _require_zero_lap(table: LookupTable) -> None:
    axis = table.axes[table.dimensions.index(LAP_CONCENTRATION)]
    if not (axis[0] <= 0.0 <= axis[-1]):
        raise ValueError(
            "albedo lookup lap_concentration axis must include zero for "
            "clean-snow products"
        )


def _with_product_metadata(
    values: xr.DataArray,
    *,
    name: str,
    long_name: str,
    units: str,
    geometry: str,
    lap_evaluation: str,
    table: LookupTable,
) -> xr.DataArray:
    product = values.astype("float32").rename(name)
    attrs: dict[str, object] = {
        "long_name": long_name,
        "units": units,
        "model": "SPIReS canonical LUT linear interpolation",
        "lookup_variable": table.name,
        "lookup_axis_ranges": lookup_axis_ranges(table),
        "grain_radius_interpretation": "effective snow grain radius",
        "grain_radius_units": "um",
        "grain_radius_transform": "sqrt(radius_um)",
        "lap_input_units": "ppm",
        "lap_type": "dust",
        "lap_evaluation": lap_evaluation,
        "geometry": geometry,
    }
    for key, value in table.provenance.items():
        attrs[f"lookup_{key}"] = value
    product.attrs = attrs
    return product


def _require_units(
    data: xr.DataArray,
    *,
    allowed: set[str],
    label: str,
    convention: str,
) -> None:
    declared = data.attrs.get("units")
    if not isinstance(declared, str) or declared.strip().casefold() not in allowed:
        raise ValueError(
            f"{label} units must identify {convention}; got {declared!r}"
        )
