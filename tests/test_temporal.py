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


def test_time_coord_is_handed_back_unconverted():
    da = cube([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    da = da.assign_coords(
        time=("time", np.array([datetime.date(2025, 1, 1 + d) for d in (0, 1, 2, 3, 4, 5, 6)], dtype=object))
    )
    out = temporal.fill_time_gaps(da, MAX_GAP)
    assert out.time.dtype == object
    assert float(out.values.ravel()[1]) == pytest.approx(2.0)


def test_dtype_dims_name_and_attrs_survive():
    da = cube([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    da.attrs = {"long_name": "Fractional snow endmember", "units": "1"}
    out = temporal.fill_time_gaps(da, MAX_GAP)
    assert out.dtype == np.float32
    assert out.dims == ("time", "y", "x")
    assert out.name == "fsnow"
    assert out.attrs == da.attrs


def test_time_need_not_be_the_first_dim():
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, (7, 6, 5)).astype("float32")
    values[rng.uniform(size=values.shape) < 0.5] = np.nan
    da = cube(values)
    straight = temporal.fill_time_gaps(da, MAX_GAP)
    transposed = temporal.fill_time_gaps(da.transpose("y", "x", "time"), MAX_GAP)
    xr.testing.assert_allclose(straight, transposed.transpose("time", "y", "x"))


def test_a_dataset_is_filled_variable_by_variable():
    rng = np.random.default_rng(4)
    layers = {}
    for name in ("fsnow", "grain_radius"):
        v = rng.uniform(0.0, 1.0, (7, 4, 4)).astype("float32")
        v[rng.uniform(size=v.shape) < 0.5] = np.nan
        layers[name] = cube(v).rename(name)
    ds = xr.Dataset(layers)
    got = temporal.fill_time_gaps(ds, MAX_GAP)
    for name in layers:
        want = oracle(ds[name])
        assert (np.isnan(want.values) == np.isnan(got[name].values)).all(), name


def test_a_timeless_variable_in_a_dataset_passes_through():
    da = cube([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    static = xr.DataArray(np.float32([[2900.0]]), dims=("y", "x"), coords={"y": [0], "x": [0]})
    ds = xr.Dataset({"fsnow": da, "elevation": static})
    out = temporal.fill_time_gaps(ds, MAX_GAP)
    xr.testing.assert_equal(out["elevation"], static)
    assert not np.isnan(out["fsnow"].values).any()


def test_the_input_is_not_mutated():
    values = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0], dtype="float32").reshape(7, 1, 1)
    da = cube(values.copy())
    before = da.values.copy()
    temporal.fill_time_gaps(da, MAX_GAP)
    np.testing.assert_array_equal(np.isnan(da.values), np.isnan(before))


def test_max_gap_accepts_the_obvious_spellings():
    da = cube([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    for gap in (datetime.timedelta(days=14), np.timedelta64(14, "D"), 14, 14.0):
        out = temporal.fill_time_gaps(da, gap)
        assert float(out.values[1, 0, 0]) == pytest.approx(2.0)


def test_an_integer_array_is_rejected():
    da = cube([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype("int32")
    with pytest.raises(ValueError, match="float dtype"):
        temporal.fill_time_gaps(da, MAX_GAP)


def test_a_missing_time_dim_is_rejected():
    da = xr.DataArray(np.float32([[1.0]]), dims=("y", "x"), coords={"y": [0], "x": [0]})
    with pytest.raises(ValueError, match="no 'time' dim"):
        temporal.fill_time_gaps(da, MAX_GAP)


def test_a_time_axis_longer_than_the_index_dtype_is_rejected():
    n = 128
    da = xr.DataArray(
        np.ones((n, 1, 1), dtype="float32"),
        dims=("time", "y", "x"),
        coords={"time": dates(range(n)), "y": [0], "x": [0]},
    )
    with pytest.raises(ValueError, match="127"):
        temporal.fill_time_gaps(da, MAX_GAP)


def test_extrapolate_matches_xarray_on_random_nan_patterns():
    rng = np.random.default_rng(5)
    values = rng.uniform(0.0, 1.0, (7, 40, 50)).astype("float32")
    values[rng.uniform(size=values.shape) < 0.45] = np.nan
    assert_matches_oracle(cube(values), extrapolate=True)


def test_extrapolate_fills_an_edge_run_within_max_gap_linearly():
    nan = np.nan
    # Valid at days 10 and 15 (values 2, 3): slope 0.2/day; leading run is
    # 10 days wide (<= 14), so day 0 -> 0.0 and day 5 -> 1.0.
    out = assert_matches_oracle(cube([nan, nan, 2.0, 3.0, 4.0, 5.0, 6.0]), extrapolate=True).ravel()
    assert out[0] == pytest.approx(0.0, abs=1e-5)
    assert out[1] == pytest.approx(1.0, abs=1e-5)


def test_extrapolate_honours_max_gap_on_the_edges_too():
    nan = np.nan
    # Last valid at day 20; trailing run spans 25 days > 14 -> persistence.
    out = assert_matches_oracle(cube([1.0, 2.0, 3.0, 4.0, 5.0, nan, nan]), extrapolate=True).ravel()
    assert out[5] == pytest.approx(5.0)
    assert out[6] == pytest.approx(5.0)


def test_extrapolate_edge_limit_is_inclusive():
    nan = np.nan
    # Leading run: first valid at day 15, so the run is exactly 15 days.
    da = cube([nan, nan, nan, 4.0, 5.0, 6.0, 7.0])
    at_15 = assert_matches_oracle(da, datetime.timedelta(days=15), extrapolate=True).ravel()
    at_14 = assert_matches_oracle(da, datetime.timedelta(days=14), extrapolate=True).ravel()
    assert at_15[0] == pytest.approx(1.0, abs=1e-5)  # line through (15,4),(20,5)
    assert at_14[0] == pytest.approx(4.0)  # persistence


def test_extrapolate_falls_back_to_persistence_without_two_samples():
    nan = np.nan
    out = temporal.fill_time_gaps(cube([nan, nan, 3.0, nan, nan, nan, nan]), MAX_GAP, extrapolate=True)
    np.testing.assert_allclose(out.values.ravel(), 3.0)
    empty = temporal.fill_time_gaps(cube([nan] * 7), MAX_GAP, extrapolate=True)
    assert np.isnan(empty.values).all(), f"invented data from nothing: {empty.values}"


def test_extrapolate_leaves_a_full_series_alone():
    values = np.linspace(0.0, 1.0, 7, dtype="float32")
    out = temporal.fill_time_gaps(cube(values), MAX_GAP, extrapolate=True)
    np.testing.assert_allclose(out.values.ravel(), values)
