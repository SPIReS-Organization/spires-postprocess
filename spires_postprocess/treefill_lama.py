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
