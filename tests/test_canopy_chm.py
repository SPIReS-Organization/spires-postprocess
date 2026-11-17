"""canopy_raster_from_chm on a synthetic EPSG:3857 height raster; one live store test."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("icechunk")
from spires_postprocess import canopy_chm  # noqa: E402
from spires_postprocess.masks import canopy_cover_fraction  # noqa: E402

# conftest's grid is EPSG:4326 around (-119.0, 37.65); build a matching 3857 CHM.
WEB_MERCATOR_X0 = -13_247_000.0
WEB_MERCATOR_Y0 = 4_540_000.0


def height_raster(heights: np.ndarray, pixel_m: float = 1.19) -> xr.DataArray:
    """(y, x) uint8 canopy height in metres, EPSG:3857, covering the test grid plus margin."""
    import rioxarray  # noqa: F401

    ny, nx = heights.shape
    x = WEB_MERCATOR_X0 + pixel_m * (np.arange(nx) + 0.5)
    y = WEB_MERCATOR_Y0 - pixel_m * (np.arange(ny) + 0.5)
    da = xr.DataArray(heights.astype("uint8"), dims=("y", "x"), coords={"y": y, "x": x}, name="chm")
    return da.rio.write_crs(canopy_chm.CHM_CRS)


def mercator_grid(n: int = 8, pixel_m: float = 30.0) -> xr.DataArray:
    """A target grid in EPSG:3857 well inside the height raster."""
    import rioxarray  # noqa: F401

    x = WEB_MERCATOR_X0 + 300.0 + pixel_m * (np.arange(n) + 0.5)
    y = WEB_MERCATOR_Y0 - 300.0 - pixel_m * (np.arange(n) + 0.5)
    da = xr.DataArray(np.zeros((n, n), dtype="float32"), dims=("y", "x"), coords={"y": y, "x": x}, name="fsnow")
    return da.rio.write_crs(canopy_chm.CHM_CRS)


def test_threshold_is_strictly_greater_than_one_metre():
    heights = np.zeros((800, 800), dtype="uint8")
    heights[:, :400] = 1   # exactly 1 m: not a tree
    heights[:, 400:] = 2   # 2 m: tree
    raster = canopy_chm.canopy_raster_from_chm(height_raster(heights), mercator_grid())
    assert raster.dtype == np.uint8
    assert raster.dims == ("y", "x")
    assert set(np.unique(raster.values)) == {0, 1}
    assert raster.rio.crs.to_epsg() == 3857
    assert raster.attrs["height_threshold_m"] == 1.0
    assert raster.attrs["source"] == "meta-chm-v2"


def test_window_is_clipped_to_target_plus_buffer():
    heights = np.full((800, 800), 5, dtype="uint8")
    target = mercator_grid()
    raster = canopy_chm.canopy_raster_from_chm(height_raster(heights), target)
    minx, miny, maxx, maxy = target.rio.bounds()
    buffer = canopy_chm.CANOPY_EDGE_BUFFER_M
    assert raster.x.min() >= minx - buffer - 1.19 and raster.x.max() <= maxx + buffer + 1.19
    assert raster.y.min() >= miny - buffer - 1.19 and raster.y.max() <= maxy + buffer + 1.19
    assert raster.shape[0] < 800 and raster.shape[1] < 800


def test_output_feeds_canopy_cover_fraction():
    heights = np.zeros((800, 800), dtype="uint8")
    heights[:, 400:] = 10
    target = mercator_grid()
    raster = canopy_chm.canopy_raster_from_chm(height_raster(heights), target)
    frac = canopy_cover_fraction(raster, target)
    assert frac.dims == ("y", "x")
    assert frac.name == "canopy_fraction"
    assert 0.0 <= float(frac.min()) and float(frac.max()) <= 1.0
    # Trees fill the eastern half of the raster, so the eastern target columns
    # are fuller than the western ones.
    assert float(frac.isel(x=-1).mean()) > float(frac.isel(x=0).mean())


def test_custom_height_threshold():
    heights = np.full((800, 800), 3, dtype="uint8")
    raster = canopy_chm.canopy_raster_from_chm(
        height_raster(heights), mercator_grid(), height_threshold_m=5.0
    )
    assert int(raster.values.sum()) == 0
    assert raster.attrs["height_threshold_m"] == 5.0


def test_window_guard_refuses_oversized_requests(monkeypatch):
    monkeypatch.setattr(canopy_chm, "MAX_WINDOW_PIXELS", 1_000)
    heights = np.zeros((800, 800), dtype="uint8")
    with pytest.raises(ValueError, match="pixels"):
        canopy_chm.canopy_raster_from_chm(height_raster(heights), mercator_grid())


def test_target_outside_the_chm_is_an_error():
    heights = np.zeros((100, 100), dtype="uint8")
    far = mercator_grid().assign_coords(x=mercator_grid().x + 1.0e6)
    far = far.rio.write_crs(canopy_chm.CHM_CRS)
    with pytest.raises(ValueError, match="overlap"):
        canopy_chm.canopy_raster_from_chm(height_raster(heights), far)
