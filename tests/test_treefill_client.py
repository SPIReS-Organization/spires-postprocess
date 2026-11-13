import io
import os

import numpy as np
import pytest

from spires_postprocess.treefill_client import (
    DEFAULT_BASE_URL,
    http_inpainter,
    service_available,
)


class FakeResponse:
    def __init__(self, array):
        buf = io.BytesIO()
        np.save(buf, array)
        self.content = buf.getvalue()
        self.status_code = 200

    def raise_for_status(self):
        return None


def test_default_base_url_is_live_service_port():
    assert DEFAULT_BASE_URL == "http://localhost:18010"


def test_http_inpainter_posts_npy_and_decodes_reply(monkeypatch):
    import requests

    captured = {}
    expected = np.full((4, 4), 0.5, dtype=np.float32)

    def fake_post(url, *, files, data, timeout):
        captured["url"] = url
        captured["fields"] = sorted(files)
        captured["fsca"] = np.load(io.BytesIO(files["fsca"][1].read()))
        captured["mask"] = np.load(io.BytesIO(files["mask"][1].read()))
        captured["data"] = data
        captured["timeout"] = timeout
        return FakeResponse(expected)

    monkeypatch.setattr(requests, "post", fake_post)
    field = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    inpainter = http_inpainter("http://example.test:18010/", timeout=12.5)
    out = inpainter(field, mask, 0.0, 1.0)

    assert captured["url"] == "http://example.test:18010/treefillnp"
    assert captured["fields"] == ["fsca", "mask"]
    np.testing.assert_array_equal(captured["fsca"], field)
    np.testing.assert_array_equal(captured["mask"], mask)
    assert captured["data"] == {"vmin": 0.0, "vmax": 1.0}
    assert captured["timeout"] == 12.5
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)
    assert inpainter.__name__ == "http_inpainter(http://example.test:18010)"


def test_http_inpainter_raises_on_http_error(monkeypatch):
    import requests

    class ErrorResponse:
        content = b""
        status_code = 500

        def raise_for_status(self):
            raise requests.HTTPError("500")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: ErrorResponse())
    inpainter = http_inpainter("http://example.test:18010")
    with pytest.raises(requests.HTTPError):
        inpainter(np.zeros((4, 4), np.float32), np.zeros((4, 4), bool), 0.0, 1.0)


def test_service_available_false_when_unreachable():
    assert service_available("http://127.0.0.1:9", timeout=0.5) is False


@pytest.mark.skipif(
    os.environ.get("SPIRES_TREEFILL_LIVE") != "1",
    reason="set SPIRES_TREEFILL_LIVE=1 to exercise the live LaMa service",
)
def test_live_service_round_trip():
    if not service_available(DEFAULT_BASE_URL):
        pytest.skip("treefill service not reachable")
    rng = np.random.default_rng(1)
    field = rng.uniform(0.2, 0.8, (64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    out = http_inpainter()(field, mask, 0.0, 1.0)
    assert out.shape == field.shape
    np.testing.assert_array_equal(out[~mask], field[~mask])
    assert np.all((out[mask] >= 0.0) & (out[mask] <= 1.0))
