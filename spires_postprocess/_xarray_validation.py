"""Shared validation and alignment helpers for xarray product calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


SUPPORTED_RESULT_DIMS = frozenset({("y", "x"), ("time", "y", "x")})


def validate_target_layout(data: xr.DataArray, label: str) -> tuple[str, ...]:
    """Validate a supported result layout with explicit dimension coordinates."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    if data.dims not in SUPPORTED_RESULT_DIMS:
        raise ValueError(
            f"{label} must have dimensions ('y', 'x') or ('time', 'y', 'x'); "
            f"got {data.dims}"
        )
    for dimension in data.dims:
        if dimension not in data.coords:
            raise ValueError(f"{label} is missing coordinate {dimension!r}")
    return data.dims


def prepare_aligned_layer(
    layer: xr.DataArray,
    target: xr.DataArray,
    *,
    label: str,
    target_label: str = "target",
    require_numeric: bool = False,
) -> xr.DataArray:
    """Validate a target-shaped or static layer and broadcast it to the target."""
    if not isinstance(layer, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    allowed_dims = {target.dims}
    if target.dims == ("time", "y", "x"):
        allowed_dims.add(("y", "x"))
    if layer.dims not in allowed_dims:
        allowed = f"{target.dims}"
        if target.dims == ("time", "y", "x"):
            allowed += " or ('y', 'x')"
        raise ValueError(f"{label} must have dimensions {allowed}; got {layer.dims}")
    if require_numeric:
        require_real_numeric(layer, label)
    require_matching_coords(
        layer,
        target,
        layer.dims,
        label=label,
        target_label=target_label,
    )
    if layer.dims == target.dims:
        return layer
    return layer.broadcast_like(target).transpose(*target.dims)


def require_matching_coords(
    data: xr.DataArray,
    target: xr.DataArray,
    dimensions: Sequence[str],
    *,
    label: str,
    target_label: str = "target",
) -> None:
    """Require exact coordinate values on every named dimension."""
    for dimension in dimensions:
        if dimension not in data.coords:
            raise ValueError(f"{label} is missing coordinate {dimension!r}")
        if dimension not in target.coords:
            raise ValueError(f"{target_label} is missing coordinate {dimension!r}")
        if not np.array_equal(
            data.coords[dimension].values,
            target.coords[dimension].values,
        ):
            raise ValueError(
                f"{label} coordinate {dimension!r} does not match {target_label}"
            )


def require_real_numeric(data: xr.DataArray, label: str) -> None:
    """Require a DataArray with a non-boolean, non-complex numeric dtype."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    require_real_numeric_dtype(data.dtype, label)


def require_float32(data: xr.DataArray, label: str) -> None:
    """Require the canonical scientific-data boundary dtype without casting."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    normalized = np.dtype(data.dtype)
    if normalized != np.dtype(np.float32):
        raise ValueError(
            f"{label} must have dtype float32 at the package boundary; "
            f"got {normalized}"
        )


def require_values_in_range(
    data: xr.DataArray,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """Reject infinite or out-of-range finite values while allowing NaNs.

    Dask-backed inputs remain Dask-backed, although validating their values
    necessarily evaluates the Boolean reduction used by this check.
    """
    invalid = np.isinf(data)
    constraints = []
    if minimum is not None:
        invalid = invalid | (data < minimum)
        constraints.append(f">= {minimum:g}")
    if maximum is not None:
        invalid = invalid | (data > maximum)
        constraints.append(f"<= {maximum:g}")

    reduced = invalid.any()
    if hasattr(reduced.data, "compute"):
        reduced = reduced.compute()
    if bool(reduced.item()):
        expected = " and ".join(constraints) if constraints else "finite or NaN"
        raise ValueError(
            f"{label} contains infinite or out-of-range values; expected "
            f"finite values {expected}, with NaN allowed"
        )


def require_real_numeric_dtype(dtype: np.dtype, label: str) -> None:
    """Require a non-boolean, non-complex NumPy numeric dtype."""
    normalized = np.dtype(dtype)
    if (
        not np.issubdtype(normalized, np.number)
        or np.issubdtype(normalized, np.bool_)
        or np.issubdtype(normalized, np.complexfloating)
    ):
        raise ValueError(
            f"{label} must contain real numeric values; got {normalized}"
        )
