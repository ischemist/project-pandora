from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from retrocast import read_json, write_json, write_json_gz

os.environ.setdefault("RETROCAST_DATA_DIR", str(Path(__file__).resolve().parent / "_unused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils  # noqa: E402


def test_normalize_native_results_preserves_payload_and_converts_none_failure() -> None:
    results, execution_stats = utils.normalize_native_results(
        {
            "solved": {
                "succ": True,
                "routes": "CCO>0.9>C.O",
                "time": 1.25,
                "expand_model_call": 4,
            },
            "failed": None,
        },
        expected_target_ids={"solved", "failed"},
    )

    assert results == {
        "solved": {
            "succ": True,
            "routes": "CCO>0.9>C.O",
            "time": 1.25,
            "expand_model_call": 4,
        },
        "failed": {"succ": False},
    }
    assert execution_stats == {
        "wall_time": {"solved": 1.25},
        "cpu_time": {},
    }


def test_normalize_native_results_rejects_coverage_and_bad_time() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        utils.normalize_native_results({}, expected_target_ids={"missing"})

    with pytest.raises(ValueError, match="invalid 'time'"):
        utils.normalize_native_results(
            {"target": {"succ": False, "time": -1}},
            expected_target_ids={"target"},
        )


def _task(target_id: str) -> dict:
    return {
        "name": "tiny",
        "targets": {
            target_id: {
                "id": target_id,
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            }
        },
    }


def test_gather_benchmark_writes_strict_planner_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tasks_dir = tmp_path / "1-benchmarks" / "definitions"
    raw_dir = tmp_path / "2-raw"
    run_dir = raw_dir / "dream-retroer" / "tiny"
    tasks_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    monkeypatch.setattr(utils, "TASKS_DIR", tasks_dir)
    monkeypatch.setattr(utils, "RAW_DIR", raw_dir)

    write_json_gz(_task("target"), tasks_dir / "tiny.json.gz")
    write_json(
        {
            "target": {
                "succ": True,
                "routes": "CCO>0.9>C.O",
                "time": 2.5,
            }
        },
        run_dir / "results.json",
    )
    (run_dir / "config.effective.yaml").write_text("iterations: 500\n", encoding="utf-8")

    manifest_path = utils.gather_benchmark(run_name="dream-retroer", benchmark_name="tiny")

    assert read_json(run_dir / "results.json.gz") == {
        "target": {
            "succ": True,
            "routes": "CCO>0.9>C.O",
            "time": 2.5,
        }
    }
    assert read_json(run_dir / "execution_stats.json.gz") == {
        "wall_time": {"target": 2.5},
        "cpu_time": {},
    }
    manifest = read_json(manifest_path)
    assert manifest["directives"] == {
        "adapter": "dreamretroer",
        "raw_results_filename": "results.json.gz",
    }
    assert manifest["statistics"] == {
        "solved_count": 1,
        "timed_targets": 1,
        "total_targets": 1,
        "total_wall_time": 2.5,
    }
