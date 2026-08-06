from delta_llm.catalog import GPU_SPECS, MODEL_SPECS, estimate_weighted_gpu_hours, validate_selection


def test_deepseek_requires_two_a40_cards() -> None:
    model = MODEL_SPECS["deepseek-r1-32b"]
    gpu = GPU_SPECS["a40"]
    assert validate_selection(model, gpu, 1)[0] is False
    assert validate_selection(model, gpu, 2)[0] is True


def test_one_h200_fits_deepseek() -> None:
    assert validate_selection(
        MODEL_SPECS["deepseek-r1-32b"], GPU_SPECS["h200"], 1
    )[0]


def test_a40_weighted_cost() -> None:
    assert estimate_weighted_gpu_hours(GPU_SPECS["a40"], 1, 47.5) == 23.75


def test_reject_too_many_cards() -> None:
    assert validate_selection(MODEL_SPECS["qwen3-8b"], GPU_SPECS["a40"], 5)[0] is False
