from __future__ import annotations

import pytest

from .conftest import (
    AIZYNTHFINDER_ASSETS,
    PROD_AIZYNTHFINDER_ASSETS,
    PROD_STOCKS,
    PROJECT_ROOT,
    RAW,
    RuntimeSmokeCase,
    run_runtime_smoke,
)

AIZYNTHFINDER_REQUIRED = (
    PROD_AIZYNTHFINDER_ASSETS / "config-mcts.yaml",
    PROD_AIZYNTHFINDER_ASSETS / "config-retrostar.yaml",
    PROD_AIZYNTHFINDER_ASSETS / "uspto_model.onnx",
    PROD_AIZYNTHFINDER_ASSETS / "uspto_templates.csv.gz",
    PROD_AIZYNTHFINDER_ASSETS / "uspto_ringbreaker_model.onnx",
    PROD_AIZYNTHFINDER_ASSETS / "uspto_ringbreaker_templates.csv.gz",
    PROD_AIZYNTHFINDER_ASSETS / "uspto_filter_model.onnx",
    PROD_AIZYNTHFINDER_ASSETS / "retrostar_value_model.pickle",
    PROD_STOCKS / "n5-stock.hdf5",
)

AIZYNTHFINDER_CASES = (
    RuntimeSmokeCase(
        name="aizynthfinder-mcts",
        runtime_dir=PROJECT_ROOT / "runtime" / "aizynthfinder",
        args=("2-run-aizyn-mcts.py", "--benchmark", "smoke-random-n5-3"),
        output_dir=RAW / "aizynthfinder-4.4.1-mcts-aizyn-iter100-depth6" / "smoke-random-n5-3",
        required_paths=AIZYNTHFINDER_REQUIRED + (AIZYNTHFINDER_ASSETS / "config-mcts.yaml",),
        expected_adapter="aizynthfinder",
    ),
    RuntimeSmokeCase(
        name="aizynthfinder-retrostar",
        runtime_dir=PROJECT_ROOT / "runtime" / "aizynthfinder",
        args=("3-run-aizyn-retro-star.py", "--benchmark", "smoke-random-n5-3", "--limit", "1"),
        output_dir=RAW / "aizynthfinder-4.4.1-retro-star-iter100-depth6" / "smoke-random-n5-3",
        required_paths=AIZYNTHFINDER_REQUIRED + (AIZYNTHFINDER_ASSETS / "config-retrostar.yaml",),
        expected_adapter="aizynthfinder",
        expected_targets=1,
    ),
)


@pytest.mark.smoke
@pytest.mark.parametrize("case", AIZYNTHFINDER_CASES, ids=[case.name for case in AIZYNTHFINDER_CASES])
def test_aizynthfinder_smoke(case: RuntimeSmokeCase) -> None:
    run_runtime_smoke(case)
