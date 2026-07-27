"""Canonical lookup-table loading, validation, and interpolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from spires_contract import (
    ContractError,
    canonical_lut_unit,
    validate_albedo_lut,
)
import xarray as xr

from spires_postprocess._xarray_validation import (
    require_float32,
    require_real_numeric_dtype,
)


SOLAR_ZENITH = "solar_zenith"
ILLUMINATION_ANGLE = "illumination_angle"
LAP_CONCENTRATION = "lap_concentration"
SQRT_GRAIN_RADIUS = "sqrt_grain_radius"
SKYVIEW = "skyview"
ALTITUDE = "altitude"

ALBEDO_ROUNDOFF_TOLERANCE = 1.0e-6
CANONICAL_LOOKUP_DIMS = (
    SOLAR_ZENITH,
    ILLUMINATION_ANGLE,
    LAP_CONCENTRATION,
    SQRT_GRAIN_RADIUS,
)
_NETCDF_SUFFIXES = {".nc", ".cdf", ".netcdf"}
_VARIABLE_UNITS = {
    "albedo": "1",
    "delta_vis": "1",
    "radiative_forcing": "W m-2",
}
_PROVENANCE_ATTRS = (
    "source",
    "source_filename",
    "source_sha256",
    "source_checksum_sha256",
    "status",
    "atmosphere_model",
    "dust_model",
    "lap_type",
)


@dataclass(frozen=True)
class LookupTable:
    """One validated canonical lookup variable with explicit axis order."""

    name: str
    dimensions: tuple[str, ...]
    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    units: str
    provenance: Mapping[str, object]


def load_albedo_lookup(source: xr.Dataset | str | Path) -> xr.Dataset:
    """Load and validate a canonical NetCDF albedo lookup dataset."""
    lookup = _load_netcdf_lookup(source, label="albedo lookup")
    validate_lookup_table(lookup, "albedo")
    return lookup


def load_forcing_lookup(source: xr.Dataset | str | Path) -> xr.Dataset:
    """Load and validate a canonical NetCDF delta-VIS/RF lookup dataset."""
    lookup = _load_netcdf_lookup(source, label="delta-VIS/RF lookup")
    present = [
        name for name in ("delta_vis", "radiative_forcing") if name in lookup
    ]
    if not present:
        raise ValueError(
            "delta-VIS/RF lookup must contain 'delta_vis', "
            "'radiative_forcing', or both"
        )
    for variable in present:
        validate_lookup_table(lookup, variable)
    return lookup


def validate_lookup_table(lookup: xr.Dataset, variable: str) -> LookupTable:
    """Validate one canonical albedo, delta-VIS, or RF lookup variable."""
    if not isinstance(lookup, xr.Dataset):
        raise TypeError("lookup must be an xarray.Dataset")
    if variable not in _VARIABLE_UNITS:
        raise ValueError(f"unsupported lookup variable {variable!r}")
    if variable not in lookup.data_vars:
        raise ValueError(f"lookup is missing required variable {variable!r}")

    data = lookup[variable]
    if variable == "albedo":
        validate_albedo_lut(lookup, expected_lap_type="dust")
        dimensions = tuple(data.dims)
    else:
        dimensions = CANONICAL_LOOKUP_DIMS
        if data.dims != dimensions:
            raise ValueError(
                f"lookup variable {variable!r} must have dimensions "
                f"{dimensions} in that order; got {data.dims}"
            )
        _validate_canonical_coordinates(lookup, dimensions)

    units = _VARIABLE_UNITS[variable]
    if data.attrs.get("units") != units:
        raise ValueError(
            f"lookup variable {variable!r} units must be {units!r}; "
            f"got {data.attrs.get('units')!r}"
        )
    require_float32(data, f"lookup variable {variable!r}")

    axes = tuple(
        _validated_coordinate(lookup, dimension, variable=variable)
        for dimension in dimensions
    )
    values = np.ascontiguousarray(np.asarray(data.data))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"lookup variable {variable!r} must be entirely finite")
    expected_shape = tuple(axis.size for axis in axes)
    if values.shape != expected_shape:
        raise ValueError(
            f"lookup variable {variable!r} shape must match coordinate lengths "
            f"{expected_shape}; got {values.shape}"
        )

    if variable in {"albedo", "delta_vis"} and (
        np.any(values < -ALBEDO_ROUNDOFF_TOLERANCE)
        or np.any(values > 1.0 + ALBEDO_ROUNDOFF_TOLERANCE)
    ):
        raise ValueError(
            f"lookup variable {variable!r} contains values outside "
            f"[-{ALBEDO_ROUNDOFF_TOLERANCE:g}, "
            f"{1.0 + ALBEDO_ROUNDOFF_TOLERANCE:g}]"
        )

    provenance = {
        name: lookup.attrs[name]
        for name in _PROVENANCE_ATTRS
        if name in lookup.attrs
    }
    return LookupTable(
        name=variable,
        dimensions=dimensions,
        axes=axes,
        values=values,
        units=units,
        provenance=provenance,
    )


def interpolate_lookup(
    table: LookupTable,
    inputs: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    """Linearly interpolate a validated table over broadcast scene arrays."""
    missing = [
        dimension for dimension in table.dimensions if dimension not in inputs
    ]
    if missing:
        raise ValueError(
            f"inputs for lookup variable {table.name!r} are missing {missing}"
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


def _load_netcdf_lookup(
    source: xr.Dataset | str | Path,
    *,
    label: str,
) -> xr.Dataset:
    if isinstance(source, xr.Dataset):
        return source
    if not isinstance(source, (str, Path)):
        raise TypeError(f"{label} must be an xarray.Dataset or NetCDF path")

    path = Path(source).expanduser()
    if path.suffix.lower() not in _NETCDF_SUFFIXES:
        raise ValueError(
            f"{label} must use NetCDF; runtime MATLAB LUT support is disabled"
        )
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    with xr.open_dataset(path) as opened:
        lookup = opened.load()
    attrs = dict(lookup.attrs)
    attrs.setdefault("source_filename", path.name)
    attrs.setdefault("source_sha256", _sha256(path))
    lookup.attrs = attrs
    return lookup


def _validate_canonical_coordinates(
    lookup: xr.Dataset,
    dimensions: tuple[str, ...],
) -> None:
    for dimension in dimensions:
        if dimension not in lookup.coords:
            raise ValueError(f"lookup is missing coordinate {dimension!r}")
        coordinate = lookup.coords[dimension]
        try:
            canonical_lut_unit(dimension, coordinate.attrs.get("units"))
        except ContractError as exc:
            raise ValueError(str(exc)) from exc

    lap_type = lookup.coords[LAP_CONCENTRATION].attrs.get("lap_type")
    if lap_type != "dust":
        raise ValueError(
            "lookup coordinate 'lap_concentration' must declare "
            "lap_type='dust'"
        )


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
