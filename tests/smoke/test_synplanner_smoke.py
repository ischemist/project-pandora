from __future__ import annotations

import pytest

from .conftest import (
    PROD_STOCKS,
    PROD_SYNPLANNER_ASSETS,
    PROJECT_ROOT,
    RAW,
    SYNPLANNER_ASSETS,
    RuntimeSmokeCase,
    run_runtime_smoke,
)

SYNPLANNER_COMMON_REQUIRED = (
    PROD_SYNPLANNER_ASSETS / "uspto" / "uspto_reaction_rules.pickle",
    PROD_SYNPLANNER_ASSETS / "uspto" / "weights" / "ranking_policy_network.ckpt",
    PROD_STOCKS / "n5-stock.csv.gz",
)

SYNPLANNER_CASES = (
    RuntimeSmokeCase(
        name="synplanner-mcts-val",
        runtime_dir=PROJECT_ROOT / "runtime" / "synplanner",
        args=("2-run-synp-val.py", "--benchmark", "smoke-random-n5-3", "--limit", "1"),
        output_dir=RAW / "synplanner-1.3.2-mcts-val-iter100" / "smoke-random-n5-3",
        required_paths=SYNPLANNER_COMMON_REQUIRED
        + (
            PROD_SYNPLANNER_ASSETS / "uspto" / "weights" / "value_network.ckpt",
            SYNPLANNER_ASSETS / "mcts-val-config.yaml",
        ),
        expected_adapter="synplanner",
        expected_targets=1,
        timeout_s=600,
    ),
    RuntimeSmokeCase(
        name="synplanner-mcts-rollout",
        runtime_dir=PROJECT_ROOT / "runtime" / "synplanner",
        args=("3-run-synp-rollout.py", "--benchmark", "smoke-random-n5-3", "--limit", "1"),
        output_dir=RAW / "synplanner-1.3.2-mcts-rollout-iter100" / "smoke-random-n5-3",
        required_paths=SYNPLANNER_COMMON_REQUIRED + (SYNPLANNER_ASSETS / "mcts-rollout-config.yaml",),
        expected_adapter="synplanner",
        expected_targets=1,
        timeout_s=600,
    ),
    RuntimeSmokeCase(
        name="synplanner-nmcs",
        runtime_dir=PROJECT_ROOT / "runtime" / "synplanner",
        args=("4-run-synp-nmcs.py", "--benchmark", "smoke-random-n5-3", "--limit", "1"),
        output_dir=RAW / "synplanner-1.3.2-nmcs-iter1-time60" / "smoke-random-n5-3",
        required_paths=SYNPLANNER_COMMON_REQUIRED + (SYNPLANNER_ASSETS / "nmcs-config.yaml",),
        expected_adapter="synplanner",
        expected_targets=1,
        timeout_s=600,
    ),
)


@pytest.mark.smoke
@pytest.mark.parametrize("case", SYNPLANNER_CASES, ids=[case.name for case in SYNPLANNER_CASES])
def test_synplanner_smoke(case: RuntimeSmokeCase) -> None:
    run_runtime_smoke(case)
