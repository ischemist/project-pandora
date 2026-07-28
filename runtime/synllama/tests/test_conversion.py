from __future__ import annotations

import csv
from pathlib import Path

import pytest
import retrocast

import synllama_runtime
from synllama_runtime import SynLlamaConversionError, convert_csv, write_artifacts


def task(*target_ids: str) -> dict:
    return {
        "name": "synllama-test",
        "targets": {
            target_id: {
                "id": target_id,
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            }
            for target_id in target_ids
        },
    }


def write_csv(path: Path, rows: list[dict[str, str]], *, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or ["Structure ID", "synthesis", "time, s"]
    with path.open("w", encoding="utf-8", newline="") as fileobj:
        writer = csv.DictWriter(fileobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_convert_csv_preserves_route_order_and_historical_summary_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "results.csv"
    write_csv(
        input_path,
        [
            {"Structure ID": "first", "synthesis": "C;R1;CC", "time, s": "2.5"},
            {"Structure ID": "second", "synthesis": "N;R2;CN", "time, s": "4"},
            {"Structure ID": "first", "synthesis": "O;R3;CO", "time, s": "2.5"},
        ],
    )

    converted = convert_csv(input_path, task("first", "second"))

    assert converted.routes_by_target == {
        "first": [
            {"synthesis_string": "C;R1;CC"},
            {"synthesis_string": "O;R3;CO"},
        ],
        "second": [{"synthesis_string": "N;R2;CN"}],
    }
    assert converted.reported_wall_times == {"first": 2.5, "second": 4.0}
    assert converted.total_reported_time_seconds == 9.0


def test_convert_csv_omits_conflicting_target_timing(tmp_path: Path) -> None:
    input_path = tmp_path / "results.csv"
    write_csv(
        input_path,
        [
            {"Structure ID": "first", "synthesis": "C;R1;CC", "time, s": "2"},
            {"Structure ID": "first", "synthesis": "O;R2;CO", "time, s": "3"},
        ],
    )

    converted = convert_csv(input_path, task("first"))

    assert converted.reported_wall_times == {}
    assert converted.conflicting_time_targets == ["first"]
    assert converted.total_reported_time_seconds == 5.0


def test_convert_csv_skips_incomplete_rows_and_invalid_times(tmp_path: Path) -> None:
    input_path = tmp_path / "results.csv"
    write_csv(
        input_path,
        [
            {"Structure ID": "", "synthesis": "C;R1;CC", "time, s": "1"},
            {"Structure ID": "first", "synthesis": "", "time, s": "2"},
            {"Structure ID": "first", "synthesis": "C;R1;CC", "time, s": "bad"},
            {"Structure ID": "first", "synthesis": "O;R2;CO", "time, s": "-1"},
            {"Structure ID": "first", "synthesis": "N;R3;CN", "time, s": "nan"},
        ],
    )

    converted = convert_csv(input_path, task("first"))

    assert converted.skipped_rows == 2
    assert converted.invalid_time_rows == 3
    assert converted.route_count == 3
    assert converted.total_reported_time_seconds == 0.0


def test_convert_csv_rejects_missing_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "results.csv"
    write_csv(input_path, [{"Structure ID": "first"}], fields=["Structure ID"])

    with pytest.raises(SynLlamaConversionError, match="missing required columns"):
        convert_csv(input_path, task("first"))


def test_convert_csv_rejects_unknown_task_target(tmp_path: Path) -> None:
    input_path = tmp_path / "results.csv"
    write_csv(input_path, [{"Structure ID": "unknown", "synthesis": "C;R1;CC", "time, s": "1"}])

    with pytest.raises(SynLlamaConversionError, match="not present in task"):
        convert_csv(input_path, task("first"))


def test_artifact_and_manifest_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "retrocast"
    task_source = data_dir / "1-benchmarks" / "definitions" / "synllama-test.json.gz"
    retrocast.write_task(task("first"), task_source)
    loaded_task = retrocast.load_task(task_source)
    input_path = data_dir / "staging" / "synllama-test" / "results.csv"
    write_csv(input_path, [{"Structure ID": "first", "synthesis": "C;R1;CCO", "time, s": "1.5"}])
    output_dir = data_dir / "2-raw" / "synllama" / "synllama-test"
    monkeypatch.setattr(synllama_runtime, "DATA_DIR", data_dir)

    converted = convert_csv(input_path, loaded_task)
    manifest = write_artifacts(
        converted=converted,
        task=loaded_task,
        task_source=task_source,
        input_path=input_path,
        output_dir=output_dir,
    )

    assert manifest["directives"] == {
        "adapter": "synllama",
        "raw_results_filename": "results.json.gz",
    }
    assert retrocast.verify_planner_manifest(output_dir / "manifest.json", data_dir)["is_valid"] is True
    assert retrocast.read_json(output_dir / "results.json.gz") == converted.routes_by_target
    assert len(
        retrocast.adapt(
            converted.routes_by_target["first"],
            "synllama",
            target=loaded_task["targets"]["first"],
        )
    ) == 1
    assert retrocast.read_json(output_dir / "execution_stats.json.gz") == {
        "wall_time": {"first": 1.5},
        "cpu_time": {},
    }
    assert retrocast.read_json(output_dir / "summary.json") == {
        "solved_count": 1,
        "time_elapsed": 1.5,
    }
