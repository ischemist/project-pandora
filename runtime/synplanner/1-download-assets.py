"""
Usage:
    uv run --directory runtime/synplanner 1-download-assets.py
"""

from synplan.utils.loading import download_selected_files
from utils import SYNPLANNER_DIR

assets = [
    ("uspto", "uspto_reaction_rules.pickle"),
    ("uspto/weights", "filtering_policy_network.ckpt"),
    ("uspto/weights", "ranking_policy_network.ckpt"),
    ("uspto/weights", "value_network.ckpt"),
]

download_selected_files(files_to_get=assets, save_to=SYNPLANNER_DIR, extract_zips=True)
