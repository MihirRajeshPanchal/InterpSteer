#!/bin/bash

# Run unified activation steering (sum + average) for all models.
# Reversed steering vector: negative - positive
# All capability steering vectors are computed once and combined layer-wise
# before a single generation pass over the LLM.
#
# Output layout:
#   <BASE_STORAGE>/<MODEL_NAME>/reverse_steering/unified_sum/
#   <BASE_STORAGE>/<MODEL_NAME>/reverse_steering/unified_average/
#
# Usage: bash run_unified_steering.sh

set -e  # Exit on error

PYTHON_SCRIPT="reasoning_probe/reverse_unified_steering.py"
BASE_STORAGE="/mnt/storage/deeksha/interpsteer"
BASE_MODEL_PATH="/mnt/storage/deeksha/models"
DATA_PATH="reasoning_probe/data/biggen_bench_translated.csv"
BASE_LOG_DIR="reasoning_probe/logs/reverse_unified_steering"

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

# Combination modes to run
COMBINE_MODES=("sum" "average")

MODELS=(
    "sarvamai/sarvam-1"
    "CohereLabs/tiny-aya-base"
    "CoRover/BharatGPT-3B-Indic"
    # "krutrim-ai-labs/Krutrim-2-instruct"
)

echo "=========================================================="
echo "Unified Activation Steering (sum + average) (REVERSED SV)"
echo "Data:           ${DATA_PATH}"
echo "Condition:      ${CONDITION}"
echo "Top-N worst:    ${TOP_N_WORST}"
echo "Alpha:          ${ALPHA}"
echo "Modes:          ${COMBINE_MODES[*]}"
echo "Models:         ${#MODELS[@]}"
echo "SV direction:   reversed (negative - positive)"
echo "=========================================================="
echo ""

for MODEL_PATH in "${MODELS[@]}"; do
    FULL_MODEL_PATH="${BASE_MODEL_PATH}/${MODEL_PATH}"
    MODEL_NAME=$(basename "${MODEL_PATH}")

    echo "=========================================================="
    echo "Model: ${MODEL_PATH}"
    echo "Path:  ${FULL_MODEL_PATH}"
    echo "=========================================================="

    if [ ! -d "${FULL_MODEL_PATH}" ]; then
        echo "  ⚠ Skipping ${MODEL_PATH}: model directory not found at ${FULL_MODEL_PATH}"
        echo ""
        continue
    fi

    ANALYSIS_DIR="${BASE_STORAGE}/${MODEL_NAME}/results"

    if [ ! -d "${ANALYSIS_DIR}" ]; then
        echo "  ⚠ Skipping ${MODEL_PATH}: analysis results not found at ${ANALYSIS_DIR}"
        echo "    Run run_bilingual_analysis.sh first."
        echo ""
        continue
    fi

    LOG_DIR="${BASE_LOG_DIR}/${MODEL_NAME}"
    mkdir -p "${LOG_DIR}"

    # Base output dir — the python script appends /unified_sum or /unified_average
    BASE_OUTPUT_DIR="${BASE_STORAGE}/${MODEL_NAME}/reverse_steering"

    for MODE in "${COMBINE_MODES[@]}"; do
        LOG_FILE="${LOG_DIR}/${MODEL_NAME}_unified_${MODE}_reverse.log"

        echo "  --------------------------------------------------"
        echo "  Combine mode : ${MODE^^}"
        echo "  Output       : ${BASE_OUTPUT_DIR}/unified_${MODE}/"
        echo "  Log          : ${LOG_FILE}"
        echo "  --------------------------------------------------"

        mkdir -p "${BASE_OUTPUT_DIR}/unified_${MODE}"

        if CUDA_VISIBLE_DEVICES=3,4 python ${PYTHON_SCRIPT} \
            --data_path              "${DATA_PATH}" \
            --model_name             "${FULL_MODEL_PATH}" \
            --analysis_dir           "${ANALYSIS_DIR}" \
            --output_dir             "${BASE_OUTPUT_DIR}" \
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
            --combine_mode           "${MODE}" \
            > "${LOG_FILE}" 2>&1; then
            echo "  ✓ Completed: ${MODEL_NAME} — unified_${MODE} (reverse)"
        else
            echo "  ✗ Failed:    ${MODEL_NAME} — unified_${MODE} (reverse)"
            echo "    Check log: ${LOG_FILE}"
        fi
        echo ""
    done

    echo "✓ Both modes done for: ${MODEL_NAME}"
    echo ""
done

echo "=========================================================="
echo "ALL UNIFIED REVERSE STEERING EXPERIMENTS COMPLETED!"
echo "=========================================================="
echo ""
echo "Results saved under:"
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "  ${BASE_STORAGE}/${MODEL_NAME}/reverse_steering/unified_sum/"
    echo "  ${BASE_STORAGE}/${MODEL_NAME}/reverse_steering/unified_average/"
done
echo ""
echo "Configuration:"
echo "  Condition:            ${CONDITION}"
echo "  Top-N worst layers:   ${TOP_N_WORST}"
echo "  Steering alpha:       ${ALPHA}"
echo "  Max eval samples:     ${NUM_EVAL_SAMPLES}"
echo "  Max steering samples: ${MAX_STEERING_SAMPLES}"
echo "  Max new tokens:       ${MAX_NEW_TOKENS}"
echo "  SV direction:         reversed (negative - positive)"