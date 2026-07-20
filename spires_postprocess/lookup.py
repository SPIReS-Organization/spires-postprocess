"""Normalized lookup-table validation and xarray-aware interpolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from spires_postprocess._xarray_validation import (
    require_float32,
    require_real_numeric_dtype,
)


COSINE_SOLAR_ZENITH = "cosine_solar_zenith"
COSINE_ILLUMINATION = "cosine_illumination"
SQRT_GRAIN_RADIUS_UM = "sqrt_grain_radius_um"
DUST_MASS_FRACTION = "dust_mass_fraction"
SOOT_MASS_FRACTION = "soot_mass_fraction"

ALBEDO_ROUNDOFF_TOLERANCE = 1.0e-6
_PROVENANCE_ATTRS = (
    "source",
    "source_checksum_sha256",
    "atmosphere_model",
    "dust_model",
)


@dataclass(frozen=True)
class LookupSchema:
    """Required normalized layout and units for one lookup variable."""

    dimensions: tuple[str, ...]
    units: str
    albedo: bool = False


@dataclass(frozen=True)
class LookupTable:
    """Validated lookup data with explicit axis order."""

    name: str
    dimensions: tuple[str, ...]
    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    units: str
    provenance: Mapping[str, object]


_ALBEDO_DIMS = (
    COSINE_SOLAR_ZENITH,
    COSINE_ILLUMINATION,
    SQRT_GRAIN_RADIUS_UM,
)
_DIRTY_DIMS = (
    *_ALBEDO_DIMS,
    DUST_MASS_FRACTION,
    SOOT_MASS_FRACTION,
)
LOOKUP_SCHEMAS: Mapping[str, LookupSchema] = {
    "clean_albedo": LookupSchema(_ALBEDO_DIMS, "1", albedo=True),
    "dirty_albedo": LookupSchema(_DIRTY_DIMS, "1", albedo=True),
    "delta_vis": LookupSchema(_DIRTY_DIMS, "1"),
    "radiative_forcing": LookupSchema(_DIRTY_DIMS, "W m-2"),
}


def validate_lookup_table(lookup: xr.Dataset, variable: str) -> LookupTable:
    """Validate and normalize one required variable from a lookup dataset.

    Only ``variable`` and its coordinates are inspected. Malformed unrelated
    variables in a combined dataset do not affect this validation call. Lookup
    data variables must be float32; coordinate dtypes remain unconstrained apart
    from being real numeric.
    """
    if not isinstance(lookup, xr.Dataset):
        raise TypeError("lookup must be an xarray.Dataset")
    if variable not in LOOKUP_SCHEMAS:
        raise ValueError(f"unsupported lookup variable {variable!r}")
    if variable not in lookup.data_vars:
        raise ValueError(f"lookup is missing required variable {variable!r}")

    schema = LOOKUP_SCHEMAS[variable]
    data = lookup[variable]
    if data.dims != schema.dimensions:
        raise ValueError(
            f"lookup variable {variable!r} must have dimensions "
            f"{schema.dimensions} in that order; got {data.dims}"
        )
    if data.attrs.get("units") != schema.units:
        raise ValueError(
            f"lookup variable {variable!r} units must be {schema.units!r}; "
            f"got {data.attrs.get('units')!r}"
        )

    axes = tuple(
        _validated_coordinate(lookup, dimension, variable=variable)
        for dimension in schema.dimensions
    )
    values = _validated_table_values(data, variable=variable)
    expected_shape = tuple(axis.size for axis in axes)
    if values.shape != expected_shape:
        raise ValueError(
            f"lookup variable {variable!r} shape must match coordinate lengths "
            f"{expected_shape}; got {values.shape}"
        )

    if schema.albedo and (
        np.any(values < -ALBEDO_ROUNDOFF_TOLERANCE)
        or np.any(values > 1.0 + ALBEDO_ROUNDOFF_TOLERANCE)
    ):
        raise ValueError(
            f"lookup variable {variable!r} contains albedo outside "
            f"[-{ALBEDO_ROUNDOFF_TOLERANCE:g}, "
            f"{1.0 + ALBEDO_ROUNDOFF_TOLERANCE:g}]"
        )

    if SOOT_MASS_FRACTION in schema.dimensions:
        soot_axis = axes[schema.dimensions.index(SOOT_MASS_FRACTION)]
        if not (soot_axis[0] <= 0.0 <= soot_axis[-1]):
            raise ValueError(
                f"lookup variable {variable!r} soot_mass_fraction coordinate "
                "must contain zero within its domain"
            )

    provenance = {
        name: lookup.attrs[name]
        for name in _PROVENANCE_ATTRS
        if name in lookup.attrs
    }
    return LookupTable(
        name=variable,
        dimensions=schema.dimensions,
        axes=axes,
        values=values,
        units=schema.units,
        provenance=provenance,
    )


def interpolate_lookup(
    table: LookupTable,
    inputs: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    """Linearly interpolate a validated table over broadcast scene arrays."""
    missing = [dimension for dimension in table.dimensions if dimension not in inputs]
    if missing:
        raise ValueError(
            f"lookup interpolation for {table.name!r} is missing input(s): {missing}"
        )
    ordered = []
    for dimension in table.dimensions:
        values = inputs[dimension]
        if not isinstance(values, xr.DataArray):
            raise TypeError(f"interpolation input {dimension!r} must be a DataArray")
        ordered.append(values)

    try:
        broadcast = xr.broadcast(*ordered)
    except ValueError as exc:
        raise ValueError(
            f"inputs for lookup variable {table.name!r} cannot be broadcast"
        ) from exc

    interpolator = RegularGridInterpolator(
        table.axes,
        table.values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    def _interpolate_block(*blocks: np.ndarray) -> np.ndarray:
        points = np.stack(blocks, axis=-1)
        return interpolator(points).astype(np.float32, copy=False)

    return xr.apply_ufunc(
        _interpolate_block,
        *broadcast,
        dask="parallelized",
        output_dtypes=[np.float32],
        keep_attrs=False,
    ).astype("float32")


def lookup_axis_ranges(table: LookupTable) -> str:
    """Return NetCDF-serializable lookup axis ranges as compact JSON."""
    ranges = {
        dimension: [float(axis[0]), float(axis[-1])]
        for dimension, axis in zip(table.dimensions, table.axes, strict=True)
    }
    return json.dumps(ranges, separators=(",", ":"), sort_keys=True)


def _validated_coordinate(
    lookup: xr.Dataset,
    dimension: str,
    *,
    variable: str,
) -> np.ndarray:
    if dimension not in lookup.coords:
        raise ValueError(
            f"lookup variable {variable!r} is missing coordinate {dimension!r}"
        )
    coordinate = lookup.coords[dimension]
    if coordinate.dims != (dimension,):
        raise ValueError(
            f"lookup coordinate {dimension!r} must be one-dimensional with "
            f"dimensions ({dimension!r},); got {coordinate.dims}"
        )
    require_real_numeric_dtype(
        coordinate.dtype, f"lookup coordinate {dimension!r}"
    )
    values = np.asarray(coordinate.data, dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"lookup coordinate {dimension!r} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"lookup coordinate {dimension!r} must be entirely finite")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError(
            f"lookup coordinate {dimension!r} must be unique and strictly increasing"
        )
    return np.ascontiguousarray(values)


def _validated_table_values(data: xr.DataArray, *, variable: str) -> np.ndarray:
    require_float32(data, f"lookup variable {variable!r}")
    # Retain canonical float32 storage. RegularGridInterpolator promotes its
    # coordinate/weight arithmetic and returned values to float64 as needed;
    # eagerly doubling the complete LUT here adds memory without recovering any
    # precision that was not present in the validated float32 source values.
    values = np.ascontiguousarray(np.asarray(data.data))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"lookup variable {variable!r} must be entirely finite")
    return values
