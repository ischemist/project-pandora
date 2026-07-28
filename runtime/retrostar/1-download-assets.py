"""Download the original Retro* model assets used by Pandora."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = PROJECT_ROOT / "data" / "retrocast" / "0-assets" / "model-configs" / "retro-star"
BASE_URL = "https://files.ischemist.com/retro-star"
ASSETS = {
    "one_step_model/saved_rollout_state_1_2048.ckpt": (f"{BASE_URL}/one_step_model/saved_rollout_state_1_2048.ckpt"),
    "one_step_model/template_rules_1.dat": f"{BASE_URL}/one_step_model/template_rules_1.dat",
    "saved_models/best_epoch_final_4.pt": f"{BASE_URL}/saved_models/best_epoch_final_4.pt",
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(destination, "wb") as fileobj,
            tqdm(total=total_size, desc=destination.name, unit="B", unit_scale=True) as progress,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fileobj.write(chunk)
                progress.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_ASSET_DIR,
        help="Asset destination directory.",
    )
    args = parser.parse_args()

    for relative_path, url in ASSETS.items():
        download_file(url, args.path / relative_path)


if __name__ == "__main__":
    main()
