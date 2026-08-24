from pathlib import Path

from examples import verify_multi_image


def test_normalize_base_url() -> None:
    assert verify_multi_image.normalize_base_url("https://example.test") == (
        "https://example.test/v1"
    )


def test_webp_is_supported_when_windows_mimetypes_does_not_know_it(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "frame.webp"
    image.write_bytes(b"webp-test-data")
    monkeypatch.setattr(verify_multi_image.mimetypes, "guess_type", lambda _: (None, None))
    assert verify_multi_image.image_data_url(image).startswith("data:image/webp;base64,")
    assert verify_multi_image.normalize_base_url("https://example.test/v1/") == (
        "https://example.test/v1"
    )


def test_verifier_submits_ordered_images_to_both_models(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    submitted = []

    def fake_request(method, url, api_key, payload=None, timeout=30):
        assert api_key == "secret"
        if url.endswith("/models"):
            return {"data": [{"id": model} for model in verify_multi_image.MODELS]}
        if method == "POST":
            submitted.append(payload)
            return {"id": f"job-{payload['model']}", "queue_position": 1}
        if url.endswith("/result"):
            return {
                "text": "first and second are different",
                "input_image_count": 2,
                "elapsed_seconds": 1.25,
                "token_budget": {
                    "remaining_tokens": 100,
                    "visual_tokens_per_image": [256, 256],
                },
                "timings": {
                    "prefill_seconds": 0.5,
                    "decode_seconds": 0.5,
                    "end_to_end_seconds": 1.0,
                },
            }
        return {"status": "succeeded", "elapsed_seconds": 1.25}

    monkeypatch.setattr(verify_multi_image, "request_json", fake_request)
    result = verify_multi_image.main(
        [
            "--base-url",
            "https://example.test/v1",
            "--api-key",
            "secret",
            "--image",
            str(first),
            "--image",
            str(second),
            "--skip-chat-check",
        ]
    )

    assert result == 0
    assert [payload["model"] for payload in submitted] == list(
        verify_multi_image.MODELS
    )
    assert all(len(payload["images"]) == 2 for payload in submitted)
    assert all(payload["thinking"] is False for payload in submitted)
    assert capsys.readouterr().out.count("[PASS]") == 2


def test_verifier_submits_exact_interleaved_content(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    submitted = []
    expected_types = ["text", "image", "text", "image", "text"]

    def fake_request(method, url, api_key, payload=None, timeout=30):
        if url.endswith("/models"):
            return {"data": [{"id": model} for model in verify_multi_image.MODELS]}
        if method == "POST":
            submitted.append(payload)
            return {"id": f"job-{payload['model']}", "queue_position": 1}
        if url.endswith("/result"):
            return {
                "text": "ordered content received",
                "input_image_count": 2,
                "input_content_types": expected_types,
                "elapsed_seconds": 1.25,
                "token_budget": {
                    "remaining_tokens": 100,
                    "visual_tokens_per_image": [256, 256],
                },
                "timings": {
                    "prefill_seconds": 0.5,
                    "decode_seconds": 0.5,
                    "end_to_end_seconds": 1.0,
                },
            }
        return {"status": "succeeded", "elapsed_seconds": 1.25}

    monkeypatch.setattr(verify_multi_image, "request_json", fake_request)
    result = verify_multi_image.main(
        [
            "--base-url",
            "https://example.test/v1",
            "--api-key",
            "secret",
            "--image",
            str(first),
            "--image",
            str(second),
            "--interleaved",
            "--skip-chat-check",
        ]
    )
    assert result == 0
    for payload in submitted:
        assert "images" not in payload
        assert "prompt" not in payload
        assert [item["type"] for item in payload["content"]] == expected_types
    assert capsys.readouterr().out.count("[PASS]") == 2
