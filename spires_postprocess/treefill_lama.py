"""In-process LaMa inpainter satisfying :data:`spires_postprocess.treefill.Inpainter`.

The model is the public ``big-lama`` TorchScript export (LaMa, Suvorov et al.
2022, Apache-2.0). It consumes a float32 ``(1, 3, H, W)`` image in [0, 1] and a
``(1, 1, H, W)`` mask in {0, 1} and returns a float image, so the field is fed
to it as a normalised float and never quantised to 8 bits. The TorchScript export
is fetched from the ``enesmsahin/simple-lama-inpainting`` GitHub release mirror.

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
            # torch checks hash_prefix before moving the download into place, so a
            # corrupt or tampered file never lands in the cache.
            torch.hub.download_url_to_file(
                LAMA_MODEL_URL, str(path), hash_prefix=expected_sha256, progress=False
            )
    if not path.exists():
        raise FileNotFoundError(f"LaMa weights not found: {path}")
    got = _sha256(path)
    if got != expected_sha256:
        raise ValueError(
            f"LaMa weights sha256 mismatch for {path}: got {got}, expected {expected_sha256}"
        )
    return path, got


def _windows(n: int, *, tile: int, overlap: int):
    """Yield ``(start, stop)`` windows of width ``tile`` covering ``[0, n)``.

    Stride is ``tile - overlap``; the last window is shifted inward so every
    window is exactly ``tile`` wide. An array shorter than ``tile`` is one window.
    """
    if n <= tile:
        yield 0, n
        return
    stride = tile - overlap
    start = 0
    while True:
        stop = start + tile
        if stop >= n:
            yield n - tile, n
            return
        yield start, stop
        start += stride


class LamaInpainter:
    """Callable ``(field, mask, vmin, vmax) -> field`` backed by LaMa.

    ``field`` is a 2-D float array (NaN allowed), ``mask`` a 2-D bool array
    (True = fill). The field is normalised to [0, 1] over ``[vmin, vmax]``,
    processed in ``tile``-sized windows overlapping by ``overlap`` pixels
    (overlaps averaged), un-normalised, and composited so pixels outside the
    mask are returned verbatim. Windows with no masked pixel are not run.
    """

    name = "lama"

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        tile: int = 4096,
        overlap: int = 128,
        model_path: str | os.PathLike | None = None,
    ) -> None:
        # Coerce before validating: a fractional tile or overlap would pass the range
        # check and then give _windows a non-integer stride, which never terminates.
        tile = int(tile)
        overlap = int(overlap)
        if overlap < 0 or overlap >= tile:
            raise ValueError("overlap must satisfy 0 <= overlap < tile")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tile = tile
        self.overlap = overlap
        self.weights_path, sha = resolve_weights(model_path)
        self.model = torch.jit.load(str(self.weights_path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        self.version = f"big-lama@{sha[:12]};torch={torch.__version__};device={self.device.type}"

    def _run_tile(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Run the model on one normalised float32 tile; returns the same shape."""
        ny, nx = x.shape
        py = (-ny) % _MODULO
        px = (-nx) % _MODULO
        xp = np.pad(x, ((0, py), (0, px)))
        mp = np.pad(m, ((0, py), (0, px)))
        img = torch.from_numpy(np.repeat(xp[None, None], 3, axis=1)).to(self.device)
        msk = torch.from_numpy(mp[None, None].astype(np.float32)).to(self.device)
        with torch.inference_mode():
            out = self.model(img, msk)
        return out[0, 0].detach().cpu().numpy()[:ny, :nx].astype(np.float32)

    def __call__(self, field, mask, vmin: float, vmax: float) -> np.ndarray:
        field = np.asarray(field, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if field.ndim != 2 or mask.shape != field.shape:
            raise ValueError(f"field and mask must be 2-D with equal shape; got {field.shape} and {mask.shape}")
        vmin, vmax = float(vmin), float(vmax)
        if not vmax > vmin:
            raise ValueError(f"vmax must exceed vmin; got vmin={vmin}, vmax={vmax}")
        if not mask.any():
            return field.copy()

        x = np.clip((np.nan_to_num(field, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
        acc = np.zeros_like(x)
        counts = np.zeros(x.shape, dtype=np.float32)
        for y0, y1 in _windows(x.shape[0], tile=self.tile, overlap=self.overlap):
            for x0, x1 in _windows(x.shape[1], tile=self.tile, overlap=self.overlap):
                m = mask[y0:y1, x0:x1]
                if not m.any():
                    continue
                acc[y0:y1, x0:x1] += self._run_tile(x[y0:y1, x0:x1], m)
                counts[y0:y1, x0:x1] += 1
        hit = counts > 0
        np.divide(acc, counts, out=acc, where=hit)
        acc *= (vmax - vmin)
        acc += vmin
        out = field.copy()
        np.copyto(out, acc, where=mask & hit)
        return out
