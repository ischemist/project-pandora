"""Download SynPlanner model assets and URSA benchmark data.

Usage:
    uv run --directory runtime/synplanner 1-download-assets.py
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

RETROCAST_BENCHMARK_BASE_URL = "https://files.ischemist.com/retrocast/data/1-benchmarks"
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1

assets = [
    ("uspto", "uspto_reaction_rules.pickle"),
    ("uspto/weights", "filtering_policy_network.ckpt"),
    ("uspto/weights", "ranking_policy_network.ckpt"),
    ("uspto/weights", "value_network.ckpt"),
]

benchmark_files = (
    ("definitions/ursa-dca-2026.json.gz", "1-benchmarks/definitions/ursa-dca-2026.json.gz"),
    ("definitions/ursa-dca-2026.manifest.json", "1-benchmarks/definitions/ursa-dca-2026.manifest.json"),
    ("definitions/ursa-expert-100.json.gz", "1-benchmarks/definitions/ursa-expert-100.json.gz"),
    ("definitions/ursa-expert-100.manifest.json", "1-benchmarks/definitions/ursa-expert-100.manifest.json"),
    ("definitions/ursa-bridge-100.json.gz", "1-benchmarks/definitions/ursa-bridge-100.json.gz"),
    ("definitions/ursa-bridge-100.manifest.json", "1-benchmarks/definitions/ursa-bridge-100.manifest.json"),
    ("stocks/ursa-stock.csv.gz", "1-benchmarks/stocks/ursa-stock.csv.gz"),
    ("stocks/ursa-stock.txt.gz", "1-benchmarks/stocks/ursa-stock.txt.gz"),
    ("stocks/ursa-stock-meta.json.gz", "1-benchmarks/stocks/ursa-stock-meta.json.gz"),
    ("stocks/ursa-stock.manifest.json", "1-benchmarks/stocks/ursa-stock.manifest.json"),
)


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "Pandora/1.0"})
    print(f"Downloading {url} -> {destination}")
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                open(temporary, "wb") as fileobj,
            ):
                shutil.copyfileobj(response, fileobj)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * 2**attempt)


def download_benchmark_files(data_dir: Path) -> None:
    for source_path, destination_path in benchmark_files:
        download_file(f"{RETROCAST_BENCHMARK_BASE_URL}/{source_path}", data_dir / destination_path)


def main() -> None:
    from synplan.utils.loading import download_selected_files
    from utils import DATA_DIR, SYNPLANNER_DIR

    download_benchmark_files(DATA_DIR)
    download_selected_files(files_to_get=assets, save_to=SYNPLANNER_DIR, extract_zips=True)


if __name__ == "__main__":
    main()
