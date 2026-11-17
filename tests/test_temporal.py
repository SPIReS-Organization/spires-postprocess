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


def test_a_numeric_time_axis_is_rejected():
    da = cube([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0]).assign_coords(time=np.arange(7) * 5)
    with pytest.raises(TypeError, match="datetime-like"):
        temporal.fill_time_gaps(da, MAX_GAP)


def test_matches_xarray_on_hand_picked_series():
    nan = np.nan
    series = [
        [1.0, nan, 3.0, nan, nan, 6.0, 7.0],
        [nan, nan, 3.0, 4.0, nan, nan, nan],
        [1.0, nan, nan, nan, nan, nan, 7.0],
        [nan, nan, nan, nan, nan, nan, nan],
        [nan, 2.0, nan, nan, nan, nan, nan],
    ]
    values = np.array(series, dtype="float32").T.reshape(7, 5, 1)
    assert_matches_oracle(cube(values))


def test_only_an_all_nan_pixel_stays_nan():
    rng = np.random.default_rng(1)
    values = rng.uniform(0.0, 1.0, (7, 30, 30)).astype("float32")
    values[rng.uniform(size=values.shape) < 0.6] = np.nan
    values[:, 0, 0] = np.nan
    out = temporal.fill_time_gaps(cube(values), MAX_GAP)
    all_nan = np.isnan(values).all(axis=0)
    assert all_nan.any() and not all_nan.all(), "fixture did not cover both cases"
    np.testing.assert_array_equal(np.isnan(out.values).all(axis=0), all_nan)
    assert not np.isnan(out.values[:, ~all_nan]).any()


def test_a_gap_wider_than_max_gap_is_persistence_not_a_ramp():
    nan = np.nan
    # Valid at day 0 and day 45; 45 > 14 so no ramp, but the trailing
    # ffill/bfill still fills every slot.
    got = temporal.fill_time_gaps(cube([1.0, nan, nan, nan, nan, nan, 7.0]), MAX_GAP)
    np.testing.assert_allclose(got.values.ravel(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])


def test_max_gap_is_inclusive():
    nan = np.nan
    # Days 0 and 15 bracket day 5 and day 10: span exactly 15 days.
    series = cube([1.0, nan, nan, 4.0, nan, nan, nan])
    exactly = temporal.fill_time_gaps(series, datetime.timedelta(days=15))
    assert float(exactly.values.ravel()[1]) == pytest.approx(2.0)
    under = temporal.fill_time_gaps(series, datetime.timedelta(days=14))
    assert float(under.values.ravel()[1]) == pytest.approx(1.0)  # persistence


def test_interpolation_is_linear_in_time_not_in_index():
    nan = np.nan
    # Days 20 (index 4) and 45 (index 6) bracket day 32 (index 5); the
    # index-linear answer would be the midpoint 0.5.
    got = temporal.fill_time_gaps(
        cube([nan, nan, nan, nan, 0.0, nan, 1.0]), datetime.timedelta(days=30)
    ).values.ravel()
    want = (32 - 20) / (45 - 20)
    assert got[5] == pytest.approx(want, abs=1e-3)
    assert got[5] != pytest.approx(0.5, abs=1e-3)


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(2)
    values = rng.uniform(0.0, 1.0, (7, 33, 17)).astype("float32")
    values[rng.uniform(size=values.shape) < 0.5] = np.nan
    da = cube(values)
    whole = temporal.fill_time_gaps(da, MAX_GAP).values
    per_row = temporal.fill_time_gaps(da, MAX_GAP, chunk_bytes=7 * 17 * 4).values
    assert (np.isnan(whole) == np.isnan(per_row)).all()
    np.testing.assert_array_equal(whole[~np.isnan(whole)], per_row[~np.isnan(per_row)])
