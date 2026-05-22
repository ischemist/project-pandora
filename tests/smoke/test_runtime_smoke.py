from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "retrocast"
BENCHMARK_DEFINITIONS = DATA_DIR / "1-benchmarks" / "definitions"
AIZYNTHFINDER_ASSETS = DATA_DIR / "0-assets" / "model-configs" / "aizynthfinder"
STOCKS = DATA_DIR / "1-benchmarks" / "stocks"
RAW = DATA_DIR / "2-raw"


@dataclass(frozen=True)
class RuntimeSmokeCase:
    name: str
    runtime_dir: Path
    args: tuple[str, ...]
    output_dir: Path
    required_paths: tuple[Path, ...]
    timeout_s: int = 180


AIZYNTHFINDER_REQUIRED = (
    AIZYNTHFINDER_ASSETS / "config-mcts.yaml",
    AIZYNTHFINDER_ASSETS / "config-retrostar.yaml",
    AIZYNTHFINDER_ASSETS / "uspto_model.onnx",
    AIZYNTHFINDER_ASSETS / "uspto_templates.csv.gz",
    AIZYNTHFINDER_ASSETS / "uspto_ringbreaker_model.onnx",
    AIZYNTHFINDER_ASSETS / "uspto_ringbreaker_templates.csv.gz",
    AIZYNTHFINDER_ASSETS / "uspto_filter_model.onnx",
    AIZYNTHFINDER_ASSETS / "retrostar_value_model.pickle",
    STOCKS / "n5-stock.hdf5",
)


SMOKE_CASES = (
    RuntimeSmokeCase(
        name="aizynthfinder-mcts",
        runtime_dir=PROJECT_ROOT / "runtime" / "aizynthfinder",
        args=("2-run-aizyn-mcts.py", "--benchmark", "smoke-random-n5-3"),
        output_dir=RAW / "aizynthfinder-4.4.1-mcts-iter100-depth6" / "smoke-random-n5-3",
        required_paths=AIZYNTHFINDER_REQUIRED,
    ),
    RuntimeSmokeCase(
        name="aizynthfinder-retrostar",
        runtime_dir=PROJECT_ROOT / "runtime" / "aizynthfinder",
        args=("3-run-aizyn-retro-star.py", "--benchmark", "smoke-random-n5-3", "--limit", "1"),
        output_dir=RAW / "aizynthfinder-4.4.1-retro-star-iter100-depth6" / "smoke-random-n5-3",
        required_paths=AIZYNTHFINDER_REQUIRED,
    ),
)


@pytest.fixture(scope="session", autouse=True)
def smoke_benchmark() -> None:
    source = PROJECT_ROOT / "tests" / "fixtures" / "benchmarks" / "smoke-random-n5-3.json"
    target = BENCHMARK_DEFINITIONS / "smoke-random-n5-3.json.gz"
    BENCHMARK_DEFINITIONS.mkdir(parents=True, exist_ok=True)
    with open(source, encoding="utf-8") as src, gzip.open(target, "wt", encoding="utf-8") as dst:
        dst.write(src.read())


def missing_paths(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not path.exists()]


@pytest.mark.smoke
@pytest.mark.parametrize("case", SMOKE_CASES, ids=[case.name for case in SMOKE_CASES])
def test_runtime_smoke(case: RuntimeSmokeCase) -> None:
    missing = missing_paths(case.required_paths)
    if missing:
        pytest.skip("missing smoke assets: " + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing))

    shutil.rmtree(case.output_dir, ignore_errors=True)
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::SyntaxWarning")

    result = subprocess.run(
        ["uv", "run", "--directory", str(case.runtime_dir.relative_to(PROJECT_ROOT)), *case.args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=case.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with gzip.open(case.output_dir / "results.json.gz", "rt", encoding="utf-8") as fileobj:
        results = json.load(fileobj)
    with open(case.output_dir / "manifest.json", encoding="utf-8") as fileobj:
        manifest = json.load(fileobj)

    expected_targets = 1 if "--limit" in case.args else 3
    assert len(results) == expected_targets
    assert manifest["statistics"]["total_targets"] == expected_targets
    assert manifest["statistics"]["benchmark_total_targets"] == 3
