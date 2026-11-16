import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from spires_postprocess import treefill_lama  # noqa: E402


def _write(path, payload=b"weights"):
    path.write_bytes(payload)
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def test_resolve_weights_prefers_explicit_path(tmp_path, monkeypatch):
    p = tmp_path / "m.pt"
    digest = _write(p)
    monkeypatch.setenv("LAMA_MODEL", str(tmp_path / "ignored.pt"))
    path, got = treefill_lama.resolve_weights(p, expected_sha256=digest)
    assert path == p and got == digest


def test_resolve_weights_uses_env_when_no_path(tmp_path, monkeypatch):
    p = tmp_path / "env.pt"
    digest = _write(p)
    monkeypatch.setenv("LAMA_MODEL", str(p))
    path, got = treefill_lama.resolve_weights(None, expected_sha256=digest)
    assert path == p and got == digest


def test_resolve_weights_rejects_bad_hash(tmp_path):
    p = tmp_path / "m.pt"
    _write(p)
    with pytest.raises(ValueError, match="sha256"):
        treefill_lama.resolve_weights(p, expected_sha256="00" * 32)


def test_resolve_weights_downloads_into_hub_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("LAMA_MODEL", raising=False)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    calls = []
    hash_prefixes = []

    def fake_download(url, dst, hash_prefix=None, progress=True):
        calls.append(url)
        hash_prefixes.append(hash_prefix)
        Path(dst).write_bytes(b"weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
    import hashlib
    digest = hashlib.sha256(b"weights").hexdigest()
    path, got = treefill_lama.resolve_weights(None, expected_sha256=digest)
    assert path == tmp_path / "checkpoints" / "big-lama.pt"
    assert got == digest and calls == [treefill_lama.LAMA_MODEL_URL]
    # torch verifies the download against hash_prefix before moving it into the cache
    assert hash_prefixes == [digest]
    # second call: cached, no download
    treefill_lama.resolve_weights(None, expected_sha256=digest)
    assert len(calls) == 1


def _fake_inpainter(tile, overlap, fill=0.25):
    """LamaInpainter with the model replaced by 'set masked pixels to fill'."""
    obj = treefill_lama.LamaInpainter.__new__(treefill_lama.LamaInpainter)
    obj.tile = tile
    obj.overlap = overlap
    obj.name = "lama"
    obj.version = "fake"
    obj.calls = 0

    def run_tile(x, m):
        obj.calls += 1
        assert x.dtype == np.float32 and m.dtype == bool and x.shape == m.shape
        assert m.any(), "mask-free tile must not reach the model"
        out = x.copy()
        out[m] = fill
        return out

    obj._run_tile = run_tile
    return obj


def _field_and_mask(ny=300, nx=260, seed=0):
    rng = np.random.default_rng(seed)
    field = rng.uniform(0.0, 1.0, (ny, nx)).astype(np.float32)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[40:90, 50:120] = True        # hole inside the first tile
    mask[200:250, 180:240] = True     # hole spanning a tile boundary at tile=128
    field[10, 10] = np.nan            # NaN outside the mask
    field[60, 60] = np.nan            # NaN inside the mask
    return field, mask


def test_tiled_matches_single_tile_for_deterministic_model():
    field, mask = _field_and_mask()
    whole = _fake_inpainter(tile=4096, overlap=0)(field, mask, 0.0, 1.0)
    tiled = _fake_inpainter(tile=128, overlap=32)(field, mask, 0.0, 1.0)
    np.testing.assert_array_equal(np.isnan(whole), np.isnan(tiled))
    np.testing.assert_allclose(np.nan_to_num(whole), np.nan_to_num(tiled), rtol=0, atol=1e-6)


def test_outside_mask_returned_verbatim_including_nan():
    field, mask = _field_and_mask()
    out = _fake_inpainter(tile=128, overlap=32)(field, mask, 0.0, 1.0)
    assert out.dtype == np.float32 and out.shape == field.shape
    np.testing.assert_array_equal(out[~mask], field[~mask])
    assert np.isnan(out[10, 10])


def test_inside_mask_filled_and_unnormalised():
    field, mask = _field_and_mask()
    out = _fake_inpainter(tile=4096, overlap=0, fill=0.25)(field, mask, 10.0, 50.0)
    assert np.all(np.isfinite(out[mask]))
    np.testing.assert_allclose(out[mask], 10.0 + 0.25 * 40.0, atol=1e-5)


def test_mask_free_tiles_are_skipped():
    field, mask = _field_and_mask()
    inp = _fake_inpainter(tile=128, overlap=32)
    inp(field, mask, 0.0, 1.0)
    # 300x260 with tile 128 / stride 96 -> 3 x 3 = 9 windows; the 3 x 3 grid is
    # ys [0,96,172] x xs [0,96,132]. Holes (40:90,50:120) and (200:250,180:240) touch
    # exactly 6 of the 9 windows, so the other 3 are mask-free and must be skipped.
    assert inp.calls == 6


def test_non_integer_tile_params_raise_instead_of_hanging(monkeypatch):
    # Before int() coercion, tile=4.5 / overlap=4.2 passed the 0 <= overlap < tile check
    # and then gave _windows a fractional stride, which never terminates. The fixed
    # constructor raises at validation, so no windowing runs.
    monkeypatch.setattr(treefill_lama, "resolve_weights", lambda *a, **k: (Path("x"), "0" * 64))
    monkeypatch.setattr(treefill_lama.torch.jit, "load", lambda *a, **k: torch.nn.Identity())
    with pytest.raises(ValueError, match="overlap"):
        treefill_lama.LamaInpainter(device="cpu", tile=4.5, overlap=4.2)


def _identity_model(img, msk):
    """Stand-in for the TorchScript graph: checks the padded input contract, returns img."""
    assert img.shape[1] == 3 and img.shape[2] % 8 == 0 and img.shape[3] % 8 == 0
    assert msk.shape[1] == 1 and msk.shape[2:] == img.shape[2:]
    return img


def test_run_tile_pads_to_multiple_of_8_and_crops_back():
    # 37 x 29 is not a multiple of 8; _run_tile must pad for the model and crop the output.
    obj = treefill_lama.LamaInpainter.__new__(treefill_lama.LamaInpainter)
    obj.device = torch.device("cpu")
    obj.model = _identity_model
    x = np.random.default_rng(1).uniform(0.0, 1.0, (37, 29)).astype(np.float32)
    m = np.zeros((37, 29), dtype=bool)
    m[5:10, 5:10] = True
    out = obj._run_tile(x, m)
    assert out.shape == (37, 29) and out.dtype == np.float32
    np.testing.assert_array_equal(out, x)


def test_run_tile_pads_symmetrically_not_with_zeros():
    """The pad must mirror the field, as simple-lama-inpainting's pad_img_to_modulo did.

    Zero padding plants a hard dark band along the bottom/right of every tile whose
    size is not a multiple of 8; the model's receptive field is global, so that band
    biases fills tile-wide (measured RMSE 0.11 vs 0.005 against the original on a
    667 x 667 HLS tile). The mask pad is irrelevant to the model as long as it is
    boolean; the field pad must reproduce the edge.
    """
    seen = {}

    def capture(img, msk):
        seen["img"] = img.numpy()[0, 0].copy()
        seen["msk"] = msk.numpy()[0, 0].copy()
        return img

    obj = treefill_lama.LamaInpainter.__new__(treefill_lama.LamaInpainter)
    obj.device = torch.device("cpu")
    obj.model = capture
    x = np.arange(5 * 6, dtype=np.float32).reshape(5, 6) / 30
    m = np.zeros((5, 6), dtype=bool)
    m[1:3, 1:3] = True
    obj._run_tile(x, m)
    img = seen["img"]
    assert img.shape == (8, 8)
    np.testing.assert_array_equal(img[:5, :6], x)
    # rows 5..7 mirror rows 4..2 (symmetric, edge included); cols 6..7 mirror cols 5..4
    np.testing.assert_array_equal(img[5:, :6], x[4:1:-1, :])
    np.testing.assert_array_equal(img[:5, 6:], x[:, 5:3:-1])
    assert seen["msk"].shape == (8, 8)


def test_shape_mismatch_raises():
    field, mask = _field_and_mask()
    with pytest.raises(ValueError, match="shape"):
        _fake_inpainter(tile=128, overlap=32)(field, mask[:-1], 0.0, 1.0)


def test_bad_range_raises():
    field, mask = _field_and_mask()
    with pytest.raises(ValueError, match="vmax"):
        _fake_inpainter(tile=128, overlap=32)(field, mask, 1.0, 1.0)


def test_windows_cover_array_and_respect_tile():
    wins = list(treefill_lama._windows(300, tile=128, overlap=32))
    assert wins[0] == (0, 128) and wins[-1] == (172, 300)
    assert all(b - a == 128 for a, b in wins)
    assert list(treefill_lama._windows(100, tile=128, overlap=32)) == [(0, 100)]


@pytest.mark.slow
def test_real_model_fills_square_hole_cpu():
    inp = treefill_lama.LamaInpainter(device="cpu", tile=256, overlap=32)
    yy, xx = np.mgrid[0:256, 0:256]
    field = np.where(((yy // 32) + (xx // 32)) % 2 == 0, 0.8, 0.2).astype(np.float32)
    mask = np.zeros_like(field, dtype=bool)
    mask[104:152, 104:152] = True
    out = inp(field, mask, 0.0, 1.0)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out[mask]))
    assert out[mask].min() >= 0.0 and out[mask].max() <= 1.0
    np.testing.assert_array_equal(out[~mask], field[~mask])
    assert inp.name == "lama"
    assert inp.version.startswith(f"big-lama@{treefill_lama.LAMA_MODEL_SHA256[:12]};torch=")
    assert inp.version.endswith(";device=cpu")
    assert 0.2 < out[mask].mean() < 0.8

    # End-to-end provenance: the real inpainter's name and version reach the output attrs.
    import xarray as xr

    from spires_postprocess.treefill import treefill_results

    coords = {"y": np.arange(256), "x": np.arange(256)}
    ds = xr.Dataset(
        {
            "fsnow": xr.DataArray(field, dims=("y", "x"), coords=coords),
            "grain_radius": xr.DataArray(field * 500.0, dims=("y", "x"), coords=coords),
            "lap_concentration": xr.DataArray(field * 100.0, dims=("y", "x"), coords=coords),
        }
    )
    tree = xr.DataArray(mask, dims=("y", "x"), coords=coords, name="tree_mask")
    tree.attrs = {"cover_threshold": 0.1, "dilate_cells": 1}
    out = treefill_results(ds, fsnow_variable="fsnow", tree_mask=tree, inpainter=inp)
    assert out["fsnow_treefilled"].attrs["inpainter"] == "lama"
    assert out["fsnow_treefilled"].attrs["inpainter_version"] == inp.version
