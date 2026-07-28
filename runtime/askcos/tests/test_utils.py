from __future__ import annotations

import json
from pathlib import Path

import pytest
import retrocast

import askcos_runtime
from askcos_runtime import (
    ExecutionTimer,
    effective_stock_name,
    find_result_file,
    gather_results,
    request_counts,
    resolve_path_within_root,
    write_effective_config,
    write_run_artifacts,
)


def task(*target_ids: str, stock: str = "buyables-stock") -> dict:
    return {
        "name": "askcos-test",
        "targets": {
            target_id: {
                "id": target_id,
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            }
            for target_id in target_ids
        },
        "default_constraints": [{"kind": "retrocast.stock_termination", "stock": stock}],
    }


def test_execution_timer_records_both_clocks() -> None:
    timer = ExecutionTimer()
    with timer.measure("target-1"):
        sum(range(10))

    assert timer.wall_time["target-1"] >= 0
    assert timer.cpu_time["target-1"] >= 0
    assert timer.to_dict() == {"wall_time": timer.wall_time, "cpu_time": timer.cpu_time}


def test_project_root_points_to_repository_root() -> None:
    expected_project_root = Path(__file__).resolve().parents[3]
    assert expected_project_root == askcos_runtime.PROJECT_ROOT


def test_resolve_path_within_root_rejects_parent_and_symlink_escapes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_root / "escape").symlink_to(outside, target_is_directory=True)

    inside = resolve_path_within_root("results/output", data_root)
    assert inside == (data_root / "results" / "output").resolve()
    with pytest.raises(ValueError, match="path must be inside"):
        resolve_path_within_root("../outside", data_root)
    with pytest.raises(ValueError, match="path must be inside"):
        resolve_path_within_root("escape/output", data_root)


def test_request_counts_separates_successes_and_failures() -> None:
    assert request_counts({"first": {"result": []}, "second": None}) == (1, 1)


def test_effective_stock_name_uses_retrocast_override_semantics() -> None:
    value = task("first", "second")
    value["constraints"] = {
        "second": [{"kind": "retrocast.stock_termination", "stock": "other-stock"}],
    }

    with pytest.raises(ValueError, match="multiple stocks"):
        effective_stock_name(value)


def test_find_result_file_prefers_position_then_target_name(tmp_path: Path) -> None:
    by_name = tmp_path / "_target_1.json"
    by_name.write_text("{}", encoding="utf-8")
    assert find_result_file(1, "target_1", tmp_path) == by_name

    by_position = tmp_path / "0001_exported-name.json"
    by_position.write_text("{}", encoding="utf-8")
    assert find_result_file(1, "target_1", tmp_path) == by_position


def test_find_result_file_rejects_ambiguous_position(tmp_path: Path) -> None:
    (tmp_path / "0001_first.json").write_text("{}", encoding="utf-8")
    (tmp_path / "0001_second.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="multiple ASKCOS files"):
        find_result_file(1, "target", tmp_path)


def test_gather_results_preserves_objects_and_reports_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "0001_first.json").write_text(json.dumps({"results": {"one": 1}}), encoding="utf-8")
    (tmp_path / "0002_second.json").write_text("not-json", encoding="utf-8")

    results, sources, missing = gather_results(task("first", "second", "third"), tmp_path)

    assert results == {"first": {"results": {"one": 1}}}
    assert sources == [tmp_path / "0001_first.json", tmp_path / "0002_second.json"]
    assert missing == ["second", "third"]


@pytest.mark.parametrize(
    ("first_target", "second_target", "filename"),
    [
        ("target_1", "other_target_1", "_other_target_1.json"),
        ("target/1", "target_1", "_target_1.json"),
    ],
)
def test_gather_results_rejects_files_matching_multiple_targets(
    tmp_path: Path,
    first_target: str,
    second_target: str,
    filename: str,
) -> None:
    (tmp_path / filename).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="match multiple task targets"):
        gather_results(task(first_target, second_target), tmp_path)


def test_write_run_artifacts_creates_verified_planner_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "retrocast"
    task_source = data_dir / "1-benchmarks" / "definitions" / "askcos-test.json.gz"
    retrocast.write_task(task("first"), task_source)
    loaded_task = retrocast.load_task(task_source)
    save_dir = data_dir / "2-raw" / "askcos" / "askcos-test"
    effective_config_path = write_effective_config({"endpoint": "http://askcos.test"}, save_dir)
    monkeypatch.setattr(askcos_runtime, "DATA_DIR", data_dir)

    manifest = write_run_artifacts(
        results={"first": {"results": {"uds": {}}}},
        execution_stats={"wall_time": {"first": 1.0}, "cpu_time": {"first": 0.5}},
        save_dir=save_dir,
        task=loaded_task,
        task_source=task_source,
        effective_config_path=effective_config_path,
        parameters={"effective_config_path": "2-raw/askcos/askcos-test/config.effective.yaml"},
    )

    assert manifest["directives"] == {
        "adapter": "askcos",
        "raw_results_filename": "results.json.gz",
    }
    assert retrocast.verify_planner_manifest(save_dir / "manifest.json", data_dir)["is_valid"] is True
    assert retrocast.read_json(save_dir / "execution_stats.json.gz") == {
        "wall_time": {"first": 1.0},
        "cpu_time": {"first": 0.5},
    }
