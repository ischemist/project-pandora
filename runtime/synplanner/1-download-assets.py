"""Download SynPlanner model assets and URSA benchmark data.

Usage:
    uv run --directory runtime/synplanner 1-download-assets.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from synplan.utils.loading import download_selected_files
from utils import DATA_DIR, SYNPLANNER_DIR

RETROCAST_BENCHMARK_BASE_URL = "https://files.ischemist.com/retrocast/data/1-benchmarks"

assets = [
    ("uspto", "uspto_reaction_rules.pickle"),
    ("uspto/weights", "filtering_policy_network.ckpt"),
    ("uspto/weights", "ranking_policy_network.ckpt"),
    ("uspto/weights", "value_network.ckpt"),
]

benchmark_files = (
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
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def download_benchmark_files() -> None:
    for source_path, destination_path in benchmark_files:
        download_file(f"{RETROCAST_BENCHMARK_BASE_URL}/{source_path}", DATA_DIR / destination_path)


download_benchmark_files()
download_selected_files(files_to_get=assets, save_to=SYNPLANNER_DIR, extract_zips=True)
