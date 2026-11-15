"""In-process LaMa inpainter satisfying :data:`spires_postprocess.treefill.Inpainter`.

The model is the public ``big-lama`` TorchScript export (LaMa, Suvorov et al.
2022, Apache-2.0). It consumes a float32 ``(1, 3, H, W)`` image in [0, 1] and a
``(1, 1, H, W)`` mask in {0, 1} and returns a float image, so the field is fed
to it as a normalised float and never quantised to 8 bits.

Requires the ``treefill`` extra (``pip install 'spires-postprocess[treefill]'``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised by the import-guard test
    raise ImportError(
        "spires_postprocess.treefill_lama needs torch; install the extra: "
        "pip install 'spires-postprocess[treefill]'"
    ) from exc

LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)
LAMA_MODEL_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
_MODULO = 8  # the TorchScript graph needs H and W divisible by 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weights(
    model_path: str | os.PathLike | None,
    *,
    expected_sha256: str = LAMA_MODEL_SHA256,
) -> tuple[Path, str]:
    """Return ``(path, sha256)`` of the TorchScript weights, downloading if needed.

    Resolution order: ``model_path`` argument, then the ``LAMA_MODEL`` environment
    variable, then ``LAMA_MODEL_URL`` fetched into ``torch.hub.get_dir()/checkpoints``.
    Raises ``ValueError`` when the file's sha256 differs from ``expected_sha256``.
    """
    if model_path is not None:
        path = Path(model_path)
    elif os.environ.get("LAMA_MODEL"):
        path = Path(os.environ["LAMA_MODEL"])
    else:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(LAMA_MODEL_URL, str(path), progress=False)
    if not path.exists():
        raise FileNotFoundError(f"LaMa weights not found: {path}")
    got = _sha256(path)
    if got != expected_sha256:
        raise ValueError(
            f"LaMa weights sha256 mismatch for {path}: got {got}, expected {expected_sha256}"
        )
    return path, got
