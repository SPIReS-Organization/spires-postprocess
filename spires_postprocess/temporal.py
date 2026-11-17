"""Temporal gap fill for ``(time, y, x)`` result stacks, vectorized over time.

This reproduces, to float32 epsilon and with an identical NaN pattern,

    obj.interpolate_na(dim="time", method="linear", max_gap=max_gap)
       .ffill(dim="time").bfill(dim="time")

(and with ``extrapolate=True`` the ``fill_value="extrapolate"`` variant for the
leading and trailing runs). xarray dispatches that expression to one SciPy
interpolator per pixel, which at tile scale (~120 Mpx x 4 variables over a
length-7 time axis) did not finish in 95 minutes on one core; the running
max/min formulation below does the same work in a few minutes.

Semantics that are easy to get backwards:

* ``max_gap`` selects the fill *method*, never whether a cell is filled. The
  trailing ``ffill``/``bfill`` is unconditional, so an interior gap whose
  bracketing samples are further apart than ``max_gap`` is filled by
  persistence instead of by a ramp. The only pixel that stays NaN is one with
  no valid sample on the whole axis.
* ``max_gap`` is inclusive (``<=``), matching xarray.
* Interpolation is linear in *time*, not in index; overpass spacing is
  irregular.
* Day offsets are float32 measured from the first sample. Realistic time axes
  are small integers of days, exact in float32, and a float64 axis doubles the
  size of every full-size temporary.
"""

from __future__ import annotations

import datetime

import numpy as np
import xarray as xr

DEFAULT_MAX_GAP = datetime.timedelta(days=14)

# Target bytes for one chunk's input slab; peak RSS runs roughly 8x this.
CHUNK_TARGET_BYTES = 512 * 1024 * 1024

# int8 index arrays: the time axis is a handful of overpasses.
_MAX_TIME_LENGTH = 127


def _day_offsets(values) -> np.ndarray:
    """Time coordinate (datetime64 or object ``datetime``) -> float32 days from the first sample."""
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = np.array([np.datetime64(v) for v in arr])
    seconds = arr.astype("datetime64[s]").astype("int64")
    return ((seconds - seconds[0]) / 86400.0).astype("float32")


def _max_gap_days(max_gap) -> float:
    """``max_gap`` -> float days. Accepts timedelta, timedelta64 or a bare number of days."""
    if isinstance(max_gap, datetime.timedelta):
        return max_gap.total_seconds() / 86400.0
    if isinstance(max_gap, np.timedelta64):
        return max_gap.astype("timedelta64[s]").astype("int64") / 86400.0
    return float(max_gap)


def fill_gaps_numpy(values, day_offsets, max_gap_days, extrapolate=False):
    """Fill NaNs along axis 0 of ``values``. Pure numpy; ``values`` is not mutated.

    Parameters
    ----------
    values : ndarray, float dtype, time on axis 0.
    day_offsets : (time,) float array of days from :func:`_day_offsets`.
    max_gap_days : float. Interior gaps whose bracketing samples are further
        apart than this are filled by persistence rather than by a ramp.
    extrapolate : bool. Fit a line through the first (last) two valid samples
        for a leading (trailing) NaN run that is no wider than ``max_gap_days``.
    """
    n = values.shape[0]
    t = np.asarray(day_offsets, dtype="float32")
    if t.shape != (n,):
        raise ValueError(f"day_offsets has shape {t.shape}, expected ({n},)")
    if n > _MAX_TIME_LENGTH:
        raise ValueError(
            f"time axis is {n} long; the int8 index arrays top out at {_MAX_TIME_LENGTH}"
        )
    idx = np.arange(n, dtype="int8").reshape((n,) + (1,) * (values.ndim - 1))

    nanmask = np.isnan(values)
    # Index of the nearest valid sample at or before each slot (-1 if none)
    # and at or after it (n if none), via running max/min over the index.
    prev = np.maximum.accumulate(np.where(nanmask, np.int8(-1), idx), axis=0)
    nxt = np.flip(
        np.minimum.accumulate(np.flip(np.where(nanmask, np.int8(n), idx), axis=0), axis=0),
        axis=0,
    )
    has_prev = prev >= 0
    has_next = nxt < n
    # Clip in place so the sentinels become safe gather indices; has_prev/has_next
    # already captured them, so this must stay after those two lines.
    np.clip(prev, 0, n - 1, out=prev)
    np.clip(nxt, 0, n - 1, out=nxt)

    vp = np.take_along_axis(values, prev, axis=0)
    vn = np.take_along_axis(values, nxt, axis=0)

    out = values.copy()

    # 1. Interior gaps bracketed within max_gap: linear in time.
    span = t[nxt] - t[prev]
    lin = nanmask & has_prev & has_next
    lin &= span <= np.float32(max_gap_days)
    weight = np.zeros(values.shape, dtype="float32")
    np.divide(t.reshape(idx.shape) - t[prev], span, out=weight, where=lin)
    del span
    ramp = vn - vp
    ramp *= weight
    ramp += vp
    np.copyto(out, ramp, where=lin)
    del ramp, weight, lin

    if extrapolate:
        _extrapolate_edges(out, values, nanmask, t, idx, n, max_gap_days)

    # 2. Whatever is still NaN sits at an edge with no bracketing neighbour
    # within max_gap (or could not be extrapolated): persistence.
    still = np.isnan(out)
    np.copyto(out, vp, where=still & has_prev)
    np.isnan(out, out=still)
    np.copyto(out, vn, where=still & has_next)
    return out


def _extrapolate_edges(out, values, nanmask, t, idx, n, max_gap_days):
    """Linearly extrapolate leading and trailing NaN runs, in place on ``out``.

    A pixel with fewer than two valid samples cannot define a line, and an edge
    run wider than ``max_gap_days`` (inclusive, like the interior test) is not
    drawn either; both are left for the persistence fallback. Interior cells are
    never written.
    """
    valid = ~nanmask
    gap = np.float32(max_gap_days)

    def _fit(first, second, region):
        v1 = np.take_along_axis(values, first, axis=0)
        v2 = np.take_along_axis(values, second, axis=0)
        dt = t[second] - t[first]
        slope = np.zeros(values.shape, dtype="float32")
        np.divide(v2 - v1, dt, out=slope, where=region)
        line = t.reshape(idx.shape) - t[first]
        line = line * slope
        line += v1
        np.copyto(out, line, where=region)

    # argmax on a boolean returns 0 when nothing is True, hence the explicit any().
    first = np.argmax(valid, axis=0).astype("int8")[np.newaxis]
    any_valid = valid.any(axis=0)[np.newaxis]
    rest = valid.copy()
    np.put_along_axis(rest, first, False, axis=0)
    second = np.argmax(rest, axis=0).astype("int8")[np.newaxis]
    has_second = rest.any(axis=0)[np.newaxis]
    del rest
    leading = nanmask & (idx < first) & any_valid & has_second
    leading &= (t[first] - t[0]) <= gap
    _fit(first, second, leading)
    del first, second, has_second, leading

    flipped = np.flip(valid, axis=0)
    last = (n - 1 - np.argmax(flipped, axis=0)).astype("int8")[np.newaxis]
    rest = valid.copy()
    np.put_along_axis(rest, last, False, axis=0)
    flipped = np.flip(rest, axis=0)
    second_last = (n - 1 - np.argmax(flipped, axis=0)).astype("int8")[np.newaxis]
    has_second_last = rest.any(axis=0)[np.newaxis]
    del rest, flipped
    trailing = nanmask & (idx > last) & any_valid & has_second_last
    trailing &= (t[-1] - t[last]) <= gap
    _fit(last, second_last, trailing)


def fill_time_gaps(obj, max_gap=DEFAULT_MAX_GAP, extrapolate=False, chunk_bytes=None):
    """Fill NaNs along ``time`` in a ``(time, y, x)`` DataArray or a Dataset of them.

    Dataset variables without a ``time`` dimension pass through untouched. The
    time coordinate is handed back unconverted. See the module docstring for the
    fill semantics; ``chunk_bytes`` bounds the input slab per chunk (default
    :data:`CHUNK_TARGET_BYTES`) and never changes the answer.
    """
    if isinstance(obj, xr.Dataset):
        filled = {
            name: (
                fill_time_gaps(da, max_gap, extrapolate, chunk_bytes)
                if "time" in da.dims
                else da
            )
            for name, da in obj.data_vars.items()
        }
        return xr.Dataset(filled, coords=obj.coords, attrs=obj.attrs)

    if not isinstance(obj, xr.DataArray):
        raise TypeError("fill_time_gaps expects an xarray.DataArray or Dataset")
    if "time" not in obj.dims:
        raise ValueError(f"no 'time' dim on {obj.name!r}; dims are {obj.dims}")
    values = obj.values
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(
            f"{obj.name!r} is {values.dtype}; a gap fill needs a float dtype to have NaNs to fill"
        )

    axis = obj.get_axis_num("time")
    t = _day_offsets(obj["time"].values)
    gap = _max_gap_days(max_gap)
    moved = np.moveaxis(values, axis, 0)
    n = moved.shape[0]
    rows = moved.shape[1] if moved.ndim > 1 else 1
    row_bytes = moved[:, :1].nbytes if moved.ndim > 1 else moved.nbytes
    target = CHUNK_TARGET_BYTES if chunk_bytes is None else int(chunk_bytes)
    step = max(1, min(rows, target // max(row_bytes, 1)))

    out = np.empty_like(moved)
    if moved.ndim == 1:
        out[:] = fill_gaps_numpy(np.ascontiguousarray(moved), t, gap, extrapolate)
    else:
        for start in range(0, rows, step):
            stop = min(rows, start + step)
            out[:, start:stop] = fill_gaps_numpy(
                np.ascontiguousarray(moved[:, start:stop]), t, gap, extrapolate
            )
    return obj.copy(data=np.moveaxis(out, 0, axis))
