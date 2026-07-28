from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import retrocast

import multistepttl_runtime
from multistepttl_runtime import (
    MultiStepTTLSerializationError,
    discover_pickle_pair,
    serialize_target,
    serialize_target_directory,
    serialize_task,
    target_id_from_directory,
    write_artifacts,
)


def task(*target_ids: str) -> dict:
    return {
        "name": "multistepttl-test",
        "targets": {
            target_id: {
                "id": target_id,
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            }
            for target_id in target_ids
        },
    }


def solved_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = pd.DataFrame(
        {
            "Solved": ["Yes", "No"],
            "Route": [[10, 20], [30]],
            "Score": [0.7, 0.1],
        }
    )
    predictions = pd.DataFrame(
        {
            "Target": ["CCO", "CC=O", "CCC"],
            "Retro": ["CC.O", "C.CO", "CC.C"],
            "Prob_Forward_Prediction_1": [0.8, 0.5, 0.1],
        },
        index=[10, 20, 30],
    )
    return tree, predictions


def write_pickle_pair(target_dir: Path) -> tuple[Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    tree, predictions = solved_frames()
    tree_path = target_dir / "run__tree.pkl"
    prediction_path = target_dir / "run__prediction.pkl"
    tree.to_pickle(tree_path)
    predictions.to_pickle(prediction_path)
    return tree_path, prediction_path


def test_serialize_target_preserves_reactions_and_scores() -> None:
    tree, predictions = solved_frames()

    routes = serialize_target(tree, predictions)

    assert routes == [
        {
            "reactions": [
                {"product": "CCO", "reactants": ["CC", "O"]},
                {"product": "CC=O", "reactants": ["C", "CO"]},
            ],
            "metadata": {
                "fwd_conf_score": pytest.approx(0.4),
                "score": pytest.approx(0.7),
                "steps": 2,
            },
        }
    ]


def test_serialize_target_accepts_reaction_id_column() -> None:
    tree, predictions = solved_frames()
    predictions = predictions.reset_index()

    assert serialize_target(tree, predictions)[0]["metadata"]["fwd_conf_score"] == pytest.approx(0.4)


def test_serialize_target_returns_empty_list_without_solved_routes() -> None:
    tree, predictions = solved_frames()
    tree["Solved"] = "No"

    assert serialize_target(tree, predictions) == []


def test_serialize_target_rejects_missing_prediction() -> None:
    tree, predictions = solved_frames()
    predictions = predictions.drop(index=20)

    with pytest.raises(MultiStepTTLSerializationError, match="reaction id 20"):
        serialize_target(tree, predictions)


def test_pickle_discovery_rejects_ambiguous_pairs(tmp_path: Path) -> None:
    write_pickle_pair(tmp_path)
    (tmp_path / "other__tree.pkl").touch()

    with pytest.raises(MultiStepTTLSerializationError, match="exactly one"):
        discover_pickle_pair(tmp_path)


def test_target_directory_without_pickle_pair_returns_none(tmp_path: Path) -> None:
    assert serialize_target_directory(tmp_path) is None


def test_task_serialization_isolates_target_failures(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    broken_tree, _prediction = write_pickle_pair(input_dir / "broken")
    broken_tree.with_name("other__tree.pkl").touch()
    write_pickle_pair(input_dir / "working")

    serialized = serialize_task(task("broken", "working"), input_dir)

    assert serialized.failed_targets == ["broken"]
    assert list(serialized.results) == ["working"]
    assert len(serialized.results["working"]) == 1


def test_uspto_directory_name_uses_legacy_target_mapping() -> None:
    assert target_id_from_directory("USPTO_190_001") == "USPTO/190/001"
    assert target_id_from_directory("another_target") == "another_target"


def test_task_serialization_and_manifest_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "retrocast"
    task_source = data_dir / "1-benchmarks" / "definitions" / "multistepttl-test.json.gz"
    retrocast.write_task(task("target-1", "target-2"), task_source)
    loaded_task = retrocast.load_task(task_source)
    input_dir = data_dir / "staging" / "multistepttl-test"
    tree_path, prediction_path = write_pickle_pair(input_dir / "target-1")
    (input_dir / "target-2").mkdir()
    output_dir = data_dir / "2-raw" / "multistepttl" / "multistepttl-test"
    monkeypatch.setattr(multistepttl_runtime, "DATA_DIR", data_dir)

    serialized = serialize_task(loaded_task, input_dir)
    manifest = write_artifacts(
        serialized=serialized,
        task=loaded_task,
        task_source=task_source,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    assert serialized.source_files == [tree_path, prediction_path]
    assert serialized.missing_pickle_targets == ["target-2"]
    assert manifest["directives"] == {
        "adapter": "multistepttl",
        "raw_results_filename": "results.json.gz",
    }
    assert retrocast.verify_planner_manifest(output_dir / "manifest.json", data_dir)["is_valid"] is True
    assert retrocast.read_json(output_dir / "results.json.gz") == serialized.results
