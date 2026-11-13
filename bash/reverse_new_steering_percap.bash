#!/bin/bash

# Run worst-layer activation steering per capability for all models
# Reversed steering vector: negative - positive
# Usage: bash run_steering_percap.sh

set -e  # Exit on error

PYTHON_SCRIPT="reasoning_probe/reverse_new_steering_percap.py"
BASE_STORAGE="/mnt/storage/deeksha/interpsteer"
BASE_MODEL_PATH="/mnt/storage/deeksha/models"
DATA_PATH="reasoning_probe/data/biggen_bench_translated.csv"
BASE_LOG_DIR="reasoning_probe/logs/reverse_steering"

# Steering settings
CONDITION="Multilingual"
TOP_N_WORST=5
ALPHA=0.5
POSITION="last"
HIGH_SCORE=5.0
LOW_SCORE=1.0
MAX_STEERING_SAMPLES=50
NUM_EVAL_SAMPLES=50
MAX_NEW_TOKENS=512
TEMPERATURE=0.7

# Column names (from the translated dataset)
INSTRUCTION_COL="translated_input"
RESPONSE_COL="translated_orig_response"
REFERENCE_COL="translated_reference_answer"
SYSTEM_PROMPT_COL="translated_system_prompt"

MODELS=(
    "sarvamai/sarvam-1"
    "CohereLabs/tiny-aya-base"
    "CoRover/BharatGPT-3B-Indic"
    # "krutrim-ai-labs/Krutrim-2-instruct"
)

CAPABILITIES=(
    "grounding"
    "instruction_following"
    "planning"
    "reasoning"
    "safety"
    "theory_of_mind"
)

echo "=========================================="
echo "Worst-Layer Activation Steering Per Capability (REVERSED SV)"
echo "Data:            ${DATA_PATH}"
echo "Condition:       ${CONDITION}"
echo "Top-N worst:     ${TOP_N_WORST}"
echo "Alpha:           ${ALPHA}"
echo "Models:          ${#MODELS[@]}"
echo "Capabilities:    ${#CAPABILITIES[@]}"
echo "SV direction:    reversed (negative - positive)"
echo "=========================================="
echo ""

for MODEL_PATH in "${MODELS[@]}"; do
    # Full path to model weights: /mnt/storage/deeksha/models/CoRover/BharatGPT-3B-Indic
    FULL_MODEL_PATH="${BASE_MODEL_PATH}/${MODEL_PATH}"
    # Short name for storage dirs: BharatGPT-3B-Indic  (matches bilingual analysis layout)
    MODEL_NAME=$(basename "${MODEL_PATH}")

    echo "=========================================="
    echo "Model: ${MODEL_PATH}"
    echo "Path:  ${FULL_MODEL_PATH}"
    echo "=========================================="

    if [ ! -d "${FULL_MODEL_PATH}" ]; then
        echo "  ⚠ Skipping ${MODEL_PATH}: model directory not found at ${FULL_MODEL_PATH}"
        echo ""
        continue
    fi

    # Analysis dir is where bilingual_metrics.csv / bilingual_report.txt live
    ANALYSIS_DIR="${BASE_STORAGE}/${MODEL_NAME}/results"

    if [ ! -d "${ANALYSIS_DIR}" ]; then
        echo "  ⚠ Skipping ${MODEL_PATH}: analysis results not found at ${ANALYSIS_DIR}"
        echo "    Run run_bilingual_analysis.sh first."
        echo ""
        continue
    fi

    LOG_DIR="${BASE_LOG_DIR}/${MODEL_NAME}"
    mkdir -p "${LOG_DIR}"

    for CAPABILITY in "${CAPABILITIES[@]}"; do
        OUTPUT_DIR="${BASE_STORAGE}/${MODEL_NAME}/reverse_steering/${CAPABILITY}"
        LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${CAPABILITY}_reverse_steering.log"

        echo "  ------------------------------------------"
        echo "  Capability: ${CAPABILITY^^}"
        echo "  Output:     ${OUTPUT_DIR}"
        echo "  Log:        ${LOG_FILE}"
        echo "  ------------------------------------------"

        mkdir -p "${OUTPUT_DIR}"

        if CUDA_VISIBLE_DEVICES=3,4 python ${PYTHON_SCRIPT} \
            --data_path              "${DATA_PATH}" \
            --model_name             "${FULL_MODEL_PATH}" \
            --analysis_dir           "${ANALYSIS_DIR}" \
            --output_dir             "${OUTPUT_DIR}" \
            --condition              "${CONDITION}" \
            --top_n_worst            ${TOP_N_WORST} \
            --alpha                  ${ALPHA} \
            --position               "${POSITION}" \
            --high_score_threshold   ${HIGH_SCORE} \
            --low_score_threshold    ${LOW_SCORE} \
            --max_steering_samples   ${MAX_STEERING_SAMPLES} \
            --num_eval_samples       ${NUM_EVAL_SAMPLES} \
            --max_new_tokens         ${MAX_NEW_TOKENS} \
            --temperature            ${TEMPERATURE} \
            --instruction_col        "${INSTRUCTION_COL}" \
            --response_col           "${RESPONSE_COL}" \
            --reference_col          "${REFERENCE_COL}" \
            --system_prompt_col      "${SYSTEM_PROMPT_COL}" \
            --filter_capability      "${CAPABILITY}" \
            --mode                   steer \
            > "${LOG_FILE}" 2>&1; then
            echo "  ✓ Completed: ${MODEL_NAME} - ${CAPABILITY^^}"
        else
            echo "  ✗ Failed:    ${MODEL_NAME} - ${CAPABILITY^^}"
            echo "    Check log: ${LOG_FILE}"
        fi
        echo ""
    done

    echo "✓ All capabilities done for: ${MODEL_NAME}"
    echo ""
done

echo "=========================================="
echo "ALL REVERSE STEERING EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Results saved under:"
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "  ${BASE_STORAGE}/${MODEL_NAME}/reverse_steering/"
done
echo ""
echo "Configuration:"
echo "  Condition:             ${CONDITION}"
echo "  Top-N worst layers:    ${TOP_N_WORST}"
echo "  Steering alpha:        ${ALPHA}"
echo "  Max eval samples:      ${NUM_EVAL_SAMPLES}"
echo "  Max steering samples:  ${MAX_STEERING_SAMPLES}"
echo "  Max new tokens:        ${MAX_NEW_TOKENS}"
echo "  SV direction:          reversed (negative - positive)"