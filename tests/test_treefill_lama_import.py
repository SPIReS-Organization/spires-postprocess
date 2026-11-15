import builtins
import sys

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
