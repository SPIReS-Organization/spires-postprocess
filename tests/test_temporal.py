"""Temporal gap fill: differential tests against the xarray expression it replaces."""

import datetime

import numpy as np
import pytest
import xarray as xr

from spires_postprocess import temporal

MAX_GAP = datetime.timedelta(days=14)
# Irregular spacing on purpose: index-linear and time-linear fills differ here.
DAY_SPACING = (0, 5, 10, 15, 20, 32, 45)


def dates(spacing=DAY_SPACING):
    start = np.datetime64("2025-01-01")
    return np.array([start + np.timedelta64(d, "D") for d in spacing])


def cube(values, spacing=DAY_SPACING):
    """(time, y, x) float32 DataArray with a datetime64 time coordinate."""
    values = np.asarray(values, dtype="float32")
    if values.ndim == 1:
        values = values.reshape(-1, 1, 1)
    ny, nx = values.shape[1:]
    return xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": dates(spacing), "y": np.arange(ny), "x": np.arange(nx)},
        name="fsnow",
    )


def oracle(da, max_gap=MAX_GAP, extrapolate=False):
    """The xarray expression the vectorized kernel must reproduce."""
    if extrapolate:
        filled = da.interpolate_na(
            dim="time", method="linear", max_gap=max_gap, fill_value="extrapolate"
        )
    else:
        filled = da.interpolate_na(dim="time", method="linear", max_gap=max_gap)
    return filled.ffill(dim="time").bfill(dim="time")


def assert_matches_oracle(da, max_gap=MAX_GAP, extrapolate=False):
    want = oracle(da, max_gap, extrapolate)
    got = temporal.fill_time_gaps(da, max_gap, extrapolate=extrapolate)
    np.testing.assert_array_equal(np.isnan(want.values), np.isnan(got.values))
    finite = ~np.isnan(want.values)
    np.testing.assert_allclose(
        got.values[finite], want.values[finite], rtol=0, atol=np.finfo("float32").eps * 8
    )
    return got.values


def test_matches_xarray_on_random_nan_patterns():
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 1.0, (7, 40, 50)).astype("float32")
    values[rng.uniform(size=values.shape) < 0.45] = np.nan
    assert_matches_oracle(cube(values))
