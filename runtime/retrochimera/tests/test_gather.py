from __future__ import annotations

import json
from pathlib import Path

import pytest
import retrocast

from gather import (
    AmbiguousResultMatchError,
    find_result_file,
    gather_results,
    normalize_target_id_for_filename,
    write_gathered_results,
)

TARGETS = {
    "target one": {
        "id": "target one",
        "smiles": "C",
        "inchikey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
    },
    "(R)-Crizotinib": {
        "id": "(R)-Crizotinib",
        "smiles": "CC",
        "inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
    },
    "missing/target": {
        "id": "missing/target",
        "smiles": "CCC",
        "inchikey": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
    },
}


def task_value() -> dict:
    return {"name": "retrochimera-test", "targets": TARGETS}


def test_filename_matching_preserves_historical_convention(tmp_path: Path) -> None:
    assert normalize_target_id_for_filename("A(B)/C D") == "A_B_C_D"

    normalized = tmp_path / "target_one.json"
    normalized.write_text("{}", encoding="utf-8")
    special = tmp_path / "Crizotinib-output.json"
    special.write_text("{}", encoding="utf-8")

    assert find_result_file("target one", tmp_path) == normalized
    assert find_result_file("(R)-Crizotinib", tmp_path) == special
    assert find_result_file("absent", tmp_path) is None


def test_filename_matching_rejects_ambiguous_prefix_fallback(tmp_path: Path) -> None:
    (tmp_path / "Crizotinib-first.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Crizotinib-second.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AmbiguousResultMatchError, match="multiple RetroChimera result files"):
        find_result_file("(R)-Crizotinib", tmp_path)


def test_gather_rejects_normalized_filename_collision(tmp_path: Path) -> None:
    colliding_task = {
        "targets": {
            "target one": {"id": "target one"},
            "target_one": {"id": "target_one"},
        }
    }
    (tmp_path / "target_one.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AmbiguousResultMatchError, match="matches both 'target one' and 'target_one'"):
        gather_results(colliding_task, tmp_path)


def test_gather_preserves_payloads_and_reports_omissions(tmp_path: Path) -> None:
    (tmp_path / "target_one.json").write_text(json.dumps({"raw": [1, 2]}), encoding="utf-8")
    (tmp_path / "Crizotinib.json").write_text("{invalid", encoding="utf-8")

    report = gather_results(task_value(), tmp_path)

    assert report.results == {"target one": {"raw": [1, 2]}}
    assert report.missing_target_ids == ("missing/target",)
    assert report.malformed_target_ids == ("(R)-Crizotinib",)
    assert {path.name for path in report.matched_files} == {"target_one.json", "Crizotinib.json"}


def test_write_gathered_results_creates_strict_planner_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "retrocast"
    task_path = data_root / "1-benchmarks" / "definitions" / "retrochimera-test.json.gz"
    eval_dir = data_root / "2-raw" / "retrochimera" / "retrochimera-test" / "parts"
    output_dir = eval_dir.parent
    eval_dir.mkdir(parents=True)
    retrocast.write_task(task_value(), task_path)
    payload = {"smiles": "C", "result": {"outputs": []}}
    (eval_dir / "target_one.json").write_text(json.dumps(payload), encoding="utf-8")

    report = write_gathered_results(
        task_path=task_path,
        eval_dir=eval_dir,
        output_dir=output_dir,
        data_root=data_root,
    )

    assert report.results == {"target one": payload}
    assert retrocast.read_json(output_dir / "results.json.gz") == report.results
    manifest = retrocast.read_json(output_dir / "manifest.json")
    assert manifest["directives"] == {
        "adapter": "retrochimera",
        "raw_results_filename": "results.json.gz",
    }
    assert manifest["statistics"] == {
        "total_targets": 3,
        "gathered_targets": 1,
        "missing_targets": 2,
        "malformed_targets": 0,
    }
    assert retrocast.verify_planner_manifest(output_dir / "manifest.json", data_root, deep=True)["is_valid"]
    assert retrocast.adapt(payload, "retrochimera", target=TARGETS["target one"]) == []


def test_write_gathered_results_rejects_external_sources(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    task_path = data_root / "task.json.gz"
    eval_dir = tmp_path / "external-eval"
    output_dir = data_root / "2-raw" / "retrochimera" / "task"
    eval_dir.mkdir()
    retrocast.write_task(task_value(), task_path)

    with pytest.raises(ValueError, match="evaluation directory must be inside data root"):
        write_gathered_results(
            task_path=task_path,
            eval_dir=eval_dir,
            output_dir=output_dir,
            data_root=data_root,
        )
