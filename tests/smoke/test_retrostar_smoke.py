from __future__ import annotations

import pytest

from .conftest import DATA_DIR, PROD_DATA_DIR, PROJECT_ROOT, RAW, RuntimeSmokeCase, run_runtime_smoke

PROD_RETROSTAR_ASSETS = PROD_DATA_DIR / "0-assets" / "model-configs" / "retro-star"
RETROSTAR_REQUIRED = (
    PROD_RETROSTAR_ASSETS / "one_step_model" / "template_rules_1.dat",
    PROD_RETROSTAR_ASSETS / "one_step_model" / "saved_rollout_state_1_2048.ckpt",
    PROD_RETROSTAR_ASSETS / "saved_models" / "best_epoch_final_4.pt",
    DATA_DIR / "1-benchmarks" / "stocks" / "n5-stock.csv.gz",
)


@pytest.mark.smoke
def test_retrostar_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROSTAR_ASSETS_DIR", str(PROD_RETROSTAR_ASSETS))
    case = RuntimeSmokeCase(
        name="retrostar",
        runtime_dir=PROJECT_ROOT / "runtime" / "retrostar",
        args=("2-run-retrostar.py", "--benchmark", "smoke-random-n5-3", "--limit", "1"),
        output_dir=RAW / "retro-star" / "smoke-random-n5-3",
        required_paths=RETROSTAR_REQUIRED,
        expected_adapter="retrostar",
        expected_targets=1,
        timeout_s=600,
    )
    run_runtime_smoke(case)
