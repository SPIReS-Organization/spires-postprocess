import builtins
import sys

import pytest


def test_import_error_names_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "spires_postprocess.canopy_chm", raising=False)
    monkeypatch.delitem(sys.modules, "icechunk", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "icechunk" or name.startswith("icechunk."):
            raise ImportError("No module named 'icechunk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"spires-postprocess\[chm\]"):
        import spires_postprocess.canopy_chm  # noqa: F401
