#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${RETROCAST_DATA_DIR:-${PROJECT_ROOT}/data/retrocast}"
CHECKPOINT_DIR="${DATA_DIR}/0-assets/model-configs/dms/checkpoints"

mkdir -p "${CHECKPOINT_DIR}"

# Define URLs
CKPT_URL="https://files.batistalab.com/DirectMultiStep/ckpts"

# Model checkpoint configurations
model_names=(
    "Flash"
    "Flex"
    "Wide"
    "Explorer-XL"
)
model_info=(
    "flash.ckpt|38"
    "flex.ckpt|74"
    "wide.ckpt|147"
    "explorer_xl.ckpt|192"
)

# Download model checkpoints
read -p "Do you want to download all model checkpoints? [y/N]: " all_choice
case "$all_choice" in
    y|Y )
        for i in "${!model_names[@]}"; do
            model="${model_names[$i]}"
            info="${model_info[$i]}"
            IFS="|" read -r filename size <<< "$info"
            echo "Downloading ${model} model ckpt (${size} MB)..."
            curl --fail --location --output "${CHECKPOINT_DIR}/${filename}" "${CKPT_URL}/${filename}"
        done
        ;;
    * )
        for i in "${!model_names[@]}"; do
            model="${model_names[$i]}"
            info="${model_info[$i]}"
            IFS="|" read -r filename size <<< "$info"
            read -p "Do you want to download ${model} model ckpt? (${size} MB) [y/N]: " choice
            case "$choice" in
                y|Y )
                    curl --fail --location --output "${CHECKPOINT_DIR}/${filename}" "${CKPT_URL}/${filename}"
                    ;;
                * )
                    echo "Skipping ${model} ckpt."
                    ;;
            esac
        done
        ;;
esac
