from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from retrocast import create_planner_manifest, read_json, write_execution_stats, write_json, write_json_gz

os.environ.setdefault("RETROCAST_DATA_DIR", str(Path(__file__).resolve().parent / "_unused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils  # noqa: E402


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "2-run-dms.py"
    spec = importlib.util.spec_from_file_location("pandora_dms_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_route_strings_rejects_non_object() -> None:
    assert utils.parse_route_strings(["{'smiles': 'CCO', 'children': []}"]) == [{"smiles": "CCO", "children": []}]

    with pytest.raises(ValueError, match="not an object"):
        utils.parse_route_strings(["['CCO']"])


def test_raw_paths_for_target_matches_directmultistep_1_1_3_interface() -> None:
    runner = _load_runner()
    dictionary_path = (
        utils.PROJECT_ROOT / "data" / "retrocast" / "0-assets" / "model-configs" / "dms" / "dms_dictionary.yaml"
    )
    route_processor = runner.RoutesProcessing(metadata_path=dictionary_path)
    raw_route = "{'smiles':'CCO','children':[{'smiles':'C','children':[]}]}"

    class FakeBeamSearch:
        def decode(self, **_kwargs):
            return [[(raw_route, 1.0)]]

    paths = runner.raw_paths_for_target(
        target_smiles="CCO",
        model_name="wide",
        route_processor=route_processor,
        beam_search=FakeBeamSearch(),
        device=torch.device("cpu"),
        use_fp16=False,
        max_steps=2,
    )

    assert utils.parse_route_strings(paths) == [{"smiles": "CCO", "children": [{"smiles": "C", "children": []}]}]


def _write_part(
    root: Path,
    *,
    run_name: str,
    task_name: str,
    target_id: str,
) -> Path:
    part_dir = root / "2-raw" / run_name / task_name
    part_dir.mkdir(parents=True)
    task_path = root / "1-benchmarks" / "definitions" / f"{task_name}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text("{}", encoding="utf-8")
    results_path = part_dir / "results.json.gz"
    stats_path = part_dir / "execution_stats.json.gz"
    manifest_path = part_dir / "manifest.json"

    write_json_gz({target_id: [{"smiles": "CCO", "children": []}]}, results_path)
    write_execution_stats(
        {"wall_time": {target_id: 1.0}, "cpu_time": {target_id: 0.5}},
        stats_path,
    )
    manifest = create_planner_manifest(
        "test-part",
        "dms",
        results_path,
        [task_path],
        root,
    )
    write_json(manifest, manifest_path)
    return part_dir


def test_gather_dms_parts_verifies_and_combines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    first = _write_part(tmp_path, run_name="dms-wide", task_name="task-pt1", target_id="target-1")
    second = _write_part(tmp_path, run_name="dms-wide", task_name="task-pt2", target_id="target-2")

    results, stats, sources = utils.gather_dms_parts(
        part_dirs=[first, second],
        expected_target_ids={"target-1", "target-2"},
    )

    assert set(results) == {"target-1", "target-2"}
    assert stats == {
        "wall_time": {"target-1": 1.0, "target-2": 1.0},
        "cpu_time": {"target-1": 0.5, "target-2": 0.5},
    }
    assert sources == [
        first / "manifest.json",
        first / "execution_stats.json.gz",
        second / "manifest.json",
        second / "execution_stats.json.gz",
    ]


def test_gather_dms_parts_rejects_duplicate_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    first = _write_part(tmp_path, run_name="dms-wide", task_name="task-pt1", target_id="duplicate")
    second = _write_part(tmp_path, run_name="dms-wide", task_name="task-pt2", target_id="duplicate")

    with pytest.raises(ValueError, match="duplicate target IDs"):
        utils.gather_dms_parts(part_dirs=[first, second], expected_target_ids={"duplicate"})


def test_write_planner_artifacts_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    output_dir = tmp_path / "2-raw" / "dms-wide" / "tiny"
    output_dir.mkdir(parents=True)
    source = tmp_path / "1-benchmarks" / "definitions" / "tiny.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")

    utils.write_planner_artifacts(
        results={"target": [{"smiles": "CCO", "children": []}]},
        execution_stats={"wall_time": {"target": 1.0}, "cpu_time": {"target": 0.5}},
        output_dir=output_dir,
        sources=[source],
        parameters={"model_name": "wide"},
        statistics={"total_targets": 1},
        action="test-run",
    )

    assert read_json(output_dir / "results.json.gz") == {"target": [{"smiles": "CCO", "children": []}]}
    assert read_json(output_dir / "manifest.json")["directives"] == {
        "adapter": "dms",
        "raw_results_filename": "results.json.gz",
    }
