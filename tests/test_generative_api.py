from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import threading

import numpy as np
from PIL import Image

from photoredactor.credential_store import EncryptedCredentialStore
from photoredactor.generative_api import (
    GenerativeAPIError,
    StabilityImageClient,
    inpaint_proxy,
    outpaint_proxy,
    strict_inpaint_result,
    strict_outpaint_result,
    validate_outpaint_dimensions,
    variant_seeds,
)


def rgba(width: int, height: int, color=(20, 40, 60, 255)) -> np.ndarray:
    return np.full((height, width, 4), color, dtype=np.uint8)


def test_variant_seeds_are_repeatable_and_bounded() -> None:
    assert variant_seeds(25, 4) == [25, 26, 27, 28]
    assert variant_seeds(4_294_967_294, 2) == [4_294_967_294, 1]
    random_values = variant_seeds(0, 4)
    assert len(random_values) == 4
    assert all(1 <= value <= 4_294_967_294 for value in random_values)


def test_inpaint_preserves_every_pixel_outside_mask() -> None:
    source = rgba(8, 6)
    generated = rgba(4, 3, (220, 10, 15, 255))
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    result = strict_inpaint_result(source, generated, mask)
    np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
    assert np.all(result[mask == 255, 0] == 220)


def test_outpaint_preserves_original_rectangle_exactly() -> None:
    source = rgba(7, 5)
    generated = rgba(3, 3, (200, 30, 10, 255))
    result = strict_outpaint_result(source, generated, (2, 3, 4, 1))
    assert result.shape == (9, 13, 4)
    np.testing.assert_array_equal(result[3:8, 2:9], source)


def test_proxy_limits_pixels_and_provider_margins() -> None:
    image = rgba(5000, 3000)
    mask = np.full((3000, 5000), 255, dtype=np.uint8)
    proxy, proxy_mask, scale = inpaint_proxy(image, mask)
    assert proxy.shape[:2] == proxy_mask.shape
    assert proxy.shape[0] * proxy.shape[1] <= 8_600_000
    assert 0 < scale < 1

    outpaint, margins, outpaint_scale = outpaint_proxy(image, (5000, 100, 3000, 20))
    assert max(margins) <= 2000
    assert outpaint_scale < 1
    assert outpaint.shape[0] * outpaint.shape[1] <= 8_600_000


def test_outpaint_dimensions_are_validated_before_request() -> None:
    image = rgba(100, 100)
    assert validate_outpaint_dimensions(image, (20, 10, 30, 40)) == (150, 150)
    for margins in ((0, 0, 0, 0), (1000, 0, 0, 0)):
        try:
            validate_outpaint_dimensions(image, margins)
        except GenerativeAPIError:
            pass
        else:
            raise AssertionError("Недопустимые размеры должны быть отклонены")


class _StabilityHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, bytes, str]] = []

    def log_message(self, _format: str, *_args) -> None:
        pass

    def do_GET(self) -> None:
        self.__class__.requests.append((self.path, b"", self.headers.get("Authorization", "")))
        payload = b'{"email":"test@example.com"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.requests.append((self.path, body, self.headers.get("Authorization", "")))
        buffer = io.BytesIO()
        Image.fromarray(rgba(96, 80, (190, 25, 35, 255)), "RGBA").save(buffer, "PNG")
        payload = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_stability_client_uses_documented_endpoints_and_fields() -> None:
    _StabilityHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StabilityHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = StabilityImageClient("real-shaped-test-key", f"http://127.0.0.1:{server.server_port}", timeout=10)
        assert client.account()["email"] == "test@example.com"
        image = rgba(8, 6)
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 2:7] = 255
        fill = client.inpaint(image, mask, "red cup", "letters", 123, "photographic")
        expanded = client.outpaint(image, (2, 1, 3, 2), "wooden table", 456, 0.4, "cinematic")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert fill.shape == image.shape
    assert np.all(fill[:, :, 0] == 190)
    assert expanded.shape == (9, 13, 4)
    np.testing.assert_array_equal(expanded[1:7, 2:10], image)
    assert [item[0] for item in _StabilityHandler.requests] == [
        "/v1/user/account",
        "/v2beta/stable-image/edit/inpaint",
        "/v2beta/stable-image/edit/outpaint",
    ]
    assert all(item[2] == "Bearer real-shaped-test-key" for item in _StabilityHandler.requests)
    inpaint_body = _StabilityHandler.requests[1][1]
    for value in (b'name="image"', b'name="mask"', b'name="prompt"', b"red cup", b'name="negative_prompt"', b"letters", b"123"):
        assert value in inpaint_body
    outpaint_body = _StabilityHandler.requests[2][1]
    for value in (b'name="left"', b'name="up"', b'name="right"', b'name="down"', b"456", b"wooden table"):
        assert value in outpaint_body


def test_api_error_keeps_status_and_request_id() -> None:
    error = StabilityImageClient._api_error(b'{"errors":["bad key"]}', 401, "request-17")
    assert isinstance(error, GenerativeAPIError)
    assert error.status == 401
    assert error.request_id == "request-17"
    assert "Неверный API-ключ" in str(error)


def test_windows_dpapi_credential_roundtrip(tmp_path) -> None:
    store = EncryptedCredentialStore(tmp_path / "credentials.bin")
    store.save("secret-value-123")
    assert store.path.read_bytes() != b"secret-value-123"
    assert store.load() == "secret-value-123"
    store.delete()
    assert store.load() is None
