#!/bin/bash

# Run bilingual (English vs Multilingual) probing analysis for all models
# Usage: bash run_bilingual_analysis.sh

set -e  # Exit on error

PYTHON_SCRIPT="reasoning_probe/probing_analysis.py"
BASE_STORAGE="/mnt/storage/deeksha/interpsteer"
BOTTOM_K=5

MODELS=(
    "tiny-aya-base"
    # "Param-1-7B"
    "BharatGPT-3B-Indic"
    "sarvam-1"
    # "Krutrim-2-instruct"
)

echo "=========================================="
echo "Bilingual Probing Analysis - All Models"
echo "=========================================="
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "------------------------------------------"
    echo "Model: ${MODEL}"
    echo "------------------------------------------"

    ENGLISH_DIR="${BASE_STORAGE}/${MODEL}/english"
    MULTILINGUAL_DIR="${BASE_STORAGE}/${MODEL}/multilingual"
    OUTPUT_DIR="${BASE_STORAGE}/${MODEL}/results"

    # Check if at least one input dir exists before running
    if [ ! -d "${ENGLISH_DIR}" ] && [ ! -d "${MULTILINGUAL_DIR}" ]; then
        echo "  ⚠ Skipping ${MODEL}: neither english nor multilingual directory found."
        echo ""
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"

    python ${PYTHON_SCRIPT} \
        --english_dir      "${ENGLISH_DIR}" \
        --multilingual_dir "${MULTILINGUAL_DIR}" \
        --output_dir       "${OUTPUT_DIR}" \
        --bottom_k         ${BOTTOM_K}

    echo "✓ Completed: ${MODEL}"
    echo "  Results saved to: ${OUTPUT_DIR}"
    echo ""
done

echo "=========================================="
echo "All models processed!"
echo "=========================================="
echo ""
echo "Results saved under:"
for MODEL in "${MODELS[@]}"; do
    echo "  ${BASE_STORAGE}/${MODEL}/results/"
done