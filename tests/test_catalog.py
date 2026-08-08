from delta_llm.catalog import MODEL_SPECS, estimate_weighted_gpu_hours, validate_gpu_count


def test_pipeline_contains_only_requested_models() -> None:
    assert set(MODEL_SPECS) == {"bagel-7b", "thinkmorph-7b"}
    assert MODEL_SPECS["bagel-7b"].assigned_gpus == 1
    assert MODEL_SPECS["thinkmorph-7b"].assigned_gpus == 1


def test_dual_model_gpu_layout() -> None:
    assert validate_gpu_count(1)[0] is False
    assert validate_gpu_count(2)[0] is True
    assert validate_gpu_count(3)[0] is True
    assert validate_gpu_count(4)[0] is True
    assert validate_gpu_count(5)[0] is False


def test_a100_weighted_cost() -> None:
    assert estimate_weighted_gpu_hours(2, 47.5) == 95.0
