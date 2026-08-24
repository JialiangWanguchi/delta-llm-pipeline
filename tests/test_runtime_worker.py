import base64
import io
import threading

from PIL import Image

from delta_llm.runtime_worker import (
    EFFECTIVE_CONTEXT_LIMIT,
    MAX_INPUT_IMAGES,
    ModelRuntime,
    build_input_terms,
    estimate_token_budget,
    image_from_data,
    image_to_data_url,
    images_from_payload,
    input_terms_from_payload,
    parse_size,
    vit_processed_size,
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


def encoded_image(color: str, size: tuple[int, int] = (8, 8)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_multiple_images_are_decoded_in_order() -> None:
    decoded = images_from_payload(
        {"images": [encoded_image("red", (8, 9)), encoded_image("blue", (10, 11))]}
    )
    assert [image.size for image in decoded] == [(8, 9), (10, 11)]
    terms = build_input_terms(decoded, "compare them")
    assert terms[0] == "Image 1:"
    assert isinstance(terms[1], Image.Image)
    assert terms[2] == "Image 2:"
    assert isinstance(terms[3], Image.Image)
    assert terms[4] == "compare them"


def test_legacy_single_image_has_no_extra_label() -> None:
    decoded = images_from_payload({"image": encoded_image("green")})
    terms = build_input_terms(decoded, "describe")
    assert len(terms) == 2
    assert isinstance(terms[0], Image.Image)
    assert terms[1] == "describe"


def test_multiple_image_validation() -> None:
    valid = encoded_image("black")
    invalid_payloads = [
        {"image": valid, "images": [valid]},
        {"images": valid},
        {"images": [valid] * (MAX_INPUT_IMAGES + 1)},
        {"images": [valid, ""]},
    ]
    for payload in invalid_payloads:
        try:
            images_from_payload(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid image payload: {payload.keys()}")


def test_legacy_images_accept_exactly_twenty_four() -> None:
    decoded = images_from_payload({"images": [encoded_image("black")] * 24})
    assert len(decoded) == MAX_INPUT_IMAGES == 24


def test_interleaved_content_preserves_exact_text_image_order() -> None:
    terms, images, types = input_terms_from_payload(
        {
            "content": [
                {"type": "text", "text": "before frame one"},
                {"type": "image", "image": encoded_image("red")},
                {"type": "text", "text": "between the frames"},
                {"type": "image", "image": encoded_image("blue")},
                {"type": "text", "text": "compare them"},
            ]
        }
    )
    assert types == ["text", "image", "text", "image", "text"]
    assert terms[0] == "before frame one"
    assert isinstance(terms[1], Image.Image)
    assert terms[2] == "between the frames"
    assert isinstance(terms[3], Image.Image)
    assert terms[4] == "compare them"
    assert len(images) == 2


def test_interleaved_content_accepts_twenty_four_images() -> None:
    content = []
    for index in range(24):
        content.extend(
            [
                {"type": "text", "text": f"Frame {index + 1}"},
                {"type": "image", "image": encoded_image("green")},
            ]
        )
    content.append({"type": "text", "text": "Summarize all frames"})
    terms, images, types = input_terms_from_payload({"content": content})
    assert len(images) == 24
    assert len(terms) == 49
    assert types == ["text", "image"] * 24 + ["text"]


def test_twenty_four_images_fit_the_default_visual_token_budget() -> None:
    terms: list[str | Image.Image] = []
    for index in range(24):
        terms.extend([f"Frame {index + 1}", Image.new("RGB", (336, 336), "green")])
    terms.append("Summarize all frames")
    budget = estimate_token_budget(terms, None, max_output_tokens=256)
    assert budget["visual_tokens"] == 24 * 576
    assert budget["required_tokens"] <= EFFECTIVE_CONTEXT_LIMIT
    assert budget["remaining_tokens"] >= 0


def test_vit_processed_size_is_patch_aligned_and_capped() -> None:
    for width, height in ((336, 336), (1920, 1080), (480, 270), (100, 1000)):
        processed_width, processed_height = vit_processed_size(width, height)
        assert processed_width % 14 == 0
        assert processed_height % 14 == 0
        assert max(processed_width, processed_height) <= 336


def test_interleaved_content_validation() -> None:
    valid = encoded_image("black")
    invalid_payloads = [
        {"prompt": "conflict", "content": [{"type": "text", "text": "x"}]},
        {"images": [valid], "content": [{"type": "text", "text": "x"}]},
        {"content": "not-an-array"},
        {"content": []},
        {"content": [{"type": "text", "text": ""}]},
        {"content": [{"type": "image", "image": valid}]},
        {"content": [{"type": "unknown", "text": "x"}]},
    ]
    for payload in invalid_payloads:
        try:
            input_terms_from_payload(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid content payload: {payload}")


def test_runtime_passes_multiple_images_to_official_interleave_api() -> None:
    class FakeInferencer:
        def __init__(self) -> None:
            self.terms = None

        def interleave_inference(self, terms, **kwargs):
            self.terms = terms
            assert kwargs["understanding_output"] is True
            return ["comparison complete"]

    runtime = ModelRuntime.__new__(ModelRuntime)
    runtime.model_name = "bagel-7b"
    runtime.lock = threading.Lock()
    runtime.metrics_lock = threading.Lock()
    runtime.queued_requests = 0
    runtime.active_requests = 0
    runtime.completed_requests = 0
    runtime.failed_requests = 0
    runtime.last_inference_seconds = None
    runtime.inferencer = FakeInferencer()

    result = runtime.generate(
        {
            "task": "image-understanding",
            "prompt": "compare",
            "images": [encoded_image("red"), encoded_image("blue")],
            "thinking": False,
            "max_output_tokens": 32,
        }
    )
    assert result["text"] == "comparison complete"
    assert result["input_image_count"] == 2
    assert result["input_content_types"] == ["text", "image", "text", "image", "text"]
    assert "images" not in result
    assert result["token_budget"]["required_tokens"] <= EFFECTIVE_CONTEXT_LIMIT
    assert runtime.inferencer.terms[0] == "Image 1:"
    assert runtime.inferencer.terms[2] == "Image 2:"


def test_runtime_passes_exact_interleaved_content_to_official_api() -> None:
    class FakeInferencer:
        def __init__(self) -> None:
            self.terms = None

        def interleave_inference(self, terms, **kwargs):
            self.terms = terms
            return ["ordered sequence received"]

    runtime = ModelRuntime.__new__(ModelRuntime)
    runtime.model_name = "thinkmorph-7b"
    runtime.lock = threading.Lock()
    runtime.metrics_lock = threading.Lock()
    runtime.queued_requests = 0
    runtime.active_requests = 0
    runtime.completed_requests = 0
    runtime.failed_requests = 0
    runtime.last_inference_seconds = None
    runtime.inferencer = FakeInferencer()
    result = runtime.generate(
        {
            "task": "image-understanding",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "image": encoded_image("red")},
                {"type": "text", "text": "second"},
                {"type": "image", "image": encoded_image("blue")},
                {"type": "text", "text": "compare"},
            ],
            "thinking": False,
            "max_output_tokens": 32,
        }
    )
    assert result["text"] == "ordered sequence received"
    assert result["input_image_count"] == 2
    assert result["input_content_types"] == ["text", "image", "text", "image", "text"]
    assert runtime.inferencer.terms[0] == "first"
    assert isinstance(runtime.inferencer.terms[1], Image.Image)
    assert runtime.inferencer.terms[2] == "second"
    assert isinstance(runtime.inferencer.terms[3], Image.Image)
    assert runtime.inferencer.terms[4] == "compare"


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


def test_runtime_accepts_text_only_input() -> None:
    class FakeInferencer:
        thinking_prompt = "think"

        def interleave_inference(self, terms, **kwargs):
            assert terms == ["Answer briefly"]
            assert kwargs["understanding_output"] is True
            return ["done"]

    runtime = ModelRuntime.__new__(ModelRuntime)
    runtime.model_name = "bagel-7b"
    runtime.lock = threading.Lock()
    runtime.metrics_lock = threading.Lock()
    runtime.queued_requests = 0
    runtime.active_requests = 0
    runtime.completed_requests = 0
    runtime.failed_requests = 0
    runtime.last_inference_seconds = None
    runtime.inferencer = FakeInferencer()
    result = runtime.generate(
        {
            "task": "image-understanding",
            "content": [{"type": "text", "text": "Answer briefly"}],
            "max_output_tokens": 8,
        }
    )
    assert result["text"] == "done"
    assert result["input_image_count"] == 0
