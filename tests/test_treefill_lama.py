import builtins
import sys

import numpy as np
import pytest


def test_import_error_names_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "spires_postprocess.treefill_lama", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"spires-postprocess\[treefill\]"):
        import spires_postprocess.treefill_lama  # noqa: F401


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

    def fake_download(url, dst, hash_prefix=None, progress=True):
        calls.append(url)
        Path(dst).write_bytes(b"weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
    import hashlib
    digest = hashlib.sha256(b"weights").hexdigest()
    path, got = treefill_lama.resolve_weights(None, expected_sha256=digest)
    assert path == tmp_path / "checkpoints" / "big-lama.pt"
    assert got == digest and calls == [treefill_lama.LAMA_MODEL_URL]
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
    # 300x260 with tile 128 / stride 96 -> 3 x 3 = 9 windows; only those touching a hole run
    assert 0 < inp.calls < 9


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
