import base64
import io

from PIL import Image

from delta_llm.runtime_worker import (
    ModelRuntime,
    image_from_data,
    image_to_data_url,
    parse_size,
)


def test_image_data_url_round_trip() -> None:
    original = Image.new("RGB", (32, 24), color="red")
    encoded = image_to_data_url(original)
    decoded = image_from_data(encoded)
    assert decoded is not None
    assert decoded.size == (32, 24)


def test_plain_base64_image_is_supported() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="blue").save(buffer, format="PNG")
    decoded = image_from_data(base64.b64encode(buffer.getvalue()).decode())
    assert decoded is not None
    assert decoded.size == (8, 8)


def test_parse_size_validates_limits_and_alignment() -> None:
    assert parse_size("512x768") == (768, 512)
    for value in ("bad", "128x128", "513x512", "2048x1024"):
        try:
            parse_size(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid size: {value}")


def test_runtime_rejects_unknown_task_before_inference() -> None:
    runtime = ModelRuntime.__new__(ModelRuntime)
    try:
        runtime.generate({"task": "not-a-task", "prompt": "test"})
    except ValueError as exc:
        assert "unsupported task" in str(exc)
    else:
        raise AssertionError("Expected an unsupported task error")
