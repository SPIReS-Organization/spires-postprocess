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
