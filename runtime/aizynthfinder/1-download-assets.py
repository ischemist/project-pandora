"""Download public AiZynthFinder model assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import tqdm

FILES_TO_DOWNLOAD = {
    "policy_model_onnx": {
        "filename": "uspto_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_model.onnx",
    },
    "template_file": {
        "filename": "uspto_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
    },
    "ringbreaker_model_onnx": {
        "filename": "uspto_ringbreaker_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx",
    },
    "ringbreaker_templates": {
        "filename": "uspto_ringbreaker_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz",
    },
    "filter_policy_onnx": {
        "filename": "uspto_filter_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
    },
    "retrostar_value_model": {
        "filename": "retrostar_value_model.pickle",
        "url": "https://github.com/MolecularAI/PaRoutes/blob/main/publication/retrostar_value_model.pickle?raw=true",
    },
}


def _download_file(url: str, filename: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(total=total_size, desc=filename.name, unit="B", unit_scale=True)
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser("download_public_data")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "retrocast"
        / "0-assets"
        / "model-configs"
        / "aizynthfinder",
        help="the path to download the files",
    )
    path = parser.parse_args().path
    path.mkdir(parents=True, exist_ok=True)

    try:
        for filespec in FILES_TO_DOWNLOAD.values():
            _download_file(filespec["url"], path / filespec["filename"])
    except requests.RequestException as err:
        print(f"Download failed with message {str(err)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
