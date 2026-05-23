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
PROD_DATA_DIR = PROJECT_ROOT / "data" / "retrocast"
DATA_DIR = PROJECT_ROOT / "data" / "retrocast-tests"
BENCHMARK_DEFINITIONS = DATA_DIR / "1-benchmarks" / "definitions"
RAW = DATA_DIR / "2-raw"
STOCKS = DATA_DIR / "1-benchmarks" / "stocks"
PROD_STOCKS = PROD_DATA_DIR / "1-benchmarks" / "stocks"

AIZYNTHFINDER_ASSETS = DATA_DIR / "0-assets" / "model-configs" / "aizynthfinder"
PROD_AIZYNTHFINDER_ASSETS = PROD_DATA_DIR / "0-assets" / "model-configs" / "aizynthfinder"
SYNPLANNER_ASSETS = DATA_DIR / "0-assets" / "model-configs" / "synplanner"
PROD_SYNPLANNER_ASSETS = PROD_DATA_DIR / "0-assets" / "model-configs" / "synplanner"


@dataclass(frozen=True)
class RuntimeSmokeCase:
    name: str
    runtime_dir: Path
    args: tuple[str, ...]
    output_dir: Path
    required_paths: tuple[Path, ...]
    expected_targets: int = 3
    timeout_s: int = 180


@pytest.fixture(scope="session", autouse=True)
def smoke_inputs() -> None:
    source = PROJECT_ROOT / "tests" / "fixtures" / "benchmarks" / "smoke-random-n5-3.json"
    target = BENCHMARK_DEFINITIONS / "smoke-random-n5-3.json.gz"
    BENCHMARK_DEFINITIONS.mkdir(parents=True, exist_ok=True)
    with open(source, encoding="utf-8") as src, gzip.open(target, "wt", encoding="utf-8") as dst:
        dst.write(src.read())

    AIZYNTHFINDER_ASSETS.mkdir(parents=True, exist_ok=True)
    for config_name in ("config-mcts.yaml", "config-retrostar.yaml"):
        source_config = PROD_AIZYNTHFINDER_ASSETS / config_name
        if not source_config.exists():
            continue
        text = source_config.read_text(encoding="utf-8")
        text = text.replace("data/retrocast/1-benchmarks/stocks/", "data/retrocast-tests/1-benchmarks/stocks/")
        (AIZYNTHFINDER_ASSETS / config_name).write_text(text, encoding="utf-8")

    SYNPLANNER_ASSETS.mkdir(parents=True, exist_ok=True)
    for config_name in ("mcts-val-config.yaml", "mcts-rollout-config.yaml", "nmcs-config.yaml"):
        source_config = PROD_SYNPLANNER_ASSETS / config_name
        if source_config.exists():
            shutil.copy2(source_config, SYNPLANNER_ASSETS / config_name)

    STOCKS.mkdir(parents=True, exist_ok=True)
    for stock_name in ("n5-stock.csv.gz", "n5-stock.hdf5"):
        source_stock = PROD_STOCKS / stock_name
        if source_stock.exists():
            shutil.copy2(source_stock, STOCKS / stock_name)


def missing_paths(paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def run_runtime_smoke(case: RuntimeSmokeCase) -> None:
    missing = missing_paths(case.required_paths)
    if missing:
        pytest.skip("missing smoke assets: " + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing))

    shutil.rmtree(case.output_dir, ignore_errors=True)
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::SyntaxWarning")
    env["RETROCAST_DATA_DIR"] = str(DATA_DIR)

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

    assert len(results) == case.expected_targets
    assert manifest["statistics"]["total_targets"] == case.expected_targets
    assert manifest["parameters"]["effective_config_path"].startswith("2-raw/")
    assert (case.output_dir / "config.effective.yaml").exists()
    if "benchmark_total_targets" in manifest["statistics"]:
        assert manifest["statistics"]["benchmark_total_targets"] == 3
