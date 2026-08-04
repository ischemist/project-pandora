from __future__ import annotations

import io
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

DOWNLOAD_PATH = Path(__file__).parents[1] / "runtime" / "synplanner" / "1-download-assets.py"
SPEC = spec_from_file_location("pandora_synplanner_download_assets", DOWNLOAD_PATH)
assert SPEC is not None and SPEC.loader is not None
DOWNLOAD = module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


def test_download_file_retries_and_atomically_promotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "assets" / "benchmark.json.gz"
    copy_attempts = 0
    sleeps: list[int] = []
    original_copyfileobj = shutil.copyfileobj

    def copyfileobj_with_transient_failure(source: io.BytesIO, target: io.BufferedWriter) -> None:
        nonlocal copy_attempts
        copy_attempts += 1
        if copy_attempts == 1:
            target.write(b"partial")
            raise OSError("transient read failure")
        original_copyfileobj(source, target)

    monkeypatch.setattr(DOWNLOAD.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"complete"))
    monkeypatch.setattr(DOWNLOAD.shutil, "copyfileobj", copyfileobj_with_transient_failure)
    monkeypatch.setattr(DOWNLOAD.time, "sleep", sleeps.append)

    DOWNLOAD.download_file("https://files.ischemist.com/benchmark.json.gz", destination)

    assert destination.read_bytes() == b"complete"
    assert not destination.with_name(f"{destination.name}.part").exists()
    assert copy_attempts == 2
    assert sleeps == [1]


def test_download_file_reraises_final_failure_and_removes_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "benchmark.json.gz"
    sleeps: list[int] = []

    def fail_download(*_args: object, **_kwargs: object) -> io.BytesIO:
        raise OSError("service unavailable")

    monkeypatch.setattr(DOWNLOAD.urllib.request, "urlopen", fail_download)
    monkeypatch.setattr(DOWNLOAD.time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="service unavailable"):
        DOWNLOAD.download_file("https://files.ischemist.com/benchmark.json.gz", destination)

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()
    assert sleeps == [1, 2]
