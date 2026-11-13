"""HTTP ``Inpainter`` backed by the spipyservice LaMa treefill service.

Kept separate from :mod:`spires_postprocess.treefill` so the core package has
no ``requests`` import at module load. The service's ``/treefillnp`` route takes
multipart ``.npy`` uploads ``fsca`` and ``mask`` plus form fields ``vmin`` and
``vmax`` and returns ``.npy`` bytes; NaN inside the mask is encoded as ``vmin``
and pixels outside the mask are returned unchanged.
"""

from __future__ import annotations

import io

import numpy as np

from spires_postprocess.treefill import Inpainter

DEFAULT_BASE_URL = "http://localhost:18010"
DEFAULT_TIMEOUT_S = 600.0
_NPY_MIME = "application/octet-stream"


def http_inpainter(
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Inpainter:
    """Return an ``Inpainter`` that POSTs to ``{base_url}/treefillnp``."""
    base = base_url.rstrip("/")

    def inpainter(field: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        import requests

        reply = requests.post(
            f"{base}/treefillnp",
            files={
                "fsca": ("fsca.npy", _npy_bytes(np.asarray(field, dtype=np.float32)), _NPY_MIME),
                "mask": ("mask.npy", _npy_bytes(np.asarray(mask, dtype=bool)), _NPY_MIME),
            },
            data={"vmin": float(vmin), "vmax": float(vmax)},
            timeout=timeout,
        )
        reply.raise_for_status()
        return np.load(io.BytesIO(reply.content)).astype(np.float32)

    inpainter.__name__ = f"http_inpainter({base})"
    return inpainter


def service_available(base_url: str = DEFAULT_BASE_URL, *, timeout: float = 2.0) -> bool:
    """True when ``GET {base_url}/health`` answers 200."""
    import requests

    try:
        return requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _npy_bytes(array: np.ndarray) -> io.BytesIO:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    buffer.seek(0)
    return buffer
