#!/bin/bash

# Bash script to run multiclass CoT probing experiments for all capabilities and models
# Usage: bash run_all_capability_experiments.sh

set -e  # Exit on error

# Configuration
DATA_PATH="reasoning_probe/data/biggen_bench_translated.csv"
BASE_MODEL_PATH="/mnt/storage/deeksha/models"
BASE_OUTPUT_DIR="/mnt/storage/deeksha/interpsteer"
BASE_LOG_DIR="reasoning_probe/logs"
PYTHON_SCRIPT="reasoning_probe/multilingual_probing.py"

# Experiment settings
NUM_EPOCHS=5
BATCH_SIZE=2
NUM_CLASSES=4
SCORE_RANGE="1.0 4.0"

# Array of models to test
MODELS=(
    "sarvamai/sarvam-1"
    "CohereLabs/tiny-aya-base"
    "CoRover/BharatGPT-3B-Indic"
    # "krutrim-ai-labs/Krutrim-2-instruct"
)

# Array of capabilities to test
CAPABILITIES=(
    "grounding"
    "instruction_following"
    "planning"
    "reasoning"
    "safety"
    "theory_of_mind"
)

echo "=========================================="
echo "Starting Multiclass CoT Probing Experiments"
echo "Data: ${DATA_PATH}"
echo "Number of Classes: ${NUM_CLASSES}"
echo "Score Range: ${SCORE_RANGE}"
echo "Models: ${#MODELS[@]}"
echo "Capabilities: ${#CAPABILITIES[@]}"
echo "=========================================="
echo ""

# Run experiments for each model
for model_path in "${MODELS[@]}"; do
    # Extract model name for directory structure
    MODEL_NAME=$(basename ${model_path})
    FULL_MODEL_PATH="${BASE_MODEL_PATH}/${model_path}"
    
    echo "=========================================="
    echo "Processing Model: ${MODEL_NAME}"
    echo "Full Path: ${FULL_MODEL_PATH}"
    echo "=========================================="
    echo ""
    
    # Check if model path exists
    if [ ! -d "${FULL_MODEL_PATH}" ]; then
        echo "WARNING: Model path does not exist: ${FULL_MODEL_PATH}"
        echo "Skipping ${MODEL_NAME}..."
        echo ""
        continue
    fi
    
    # Create log directory for this model
    LOG_DIR="${BASE_LOG_DIR}/${MODEL_NAME}/multilingual"
    mkdir -p ${LOG_DIR}
    
    # Run experiments for each capability
    for capability in "${CAPABILITIES[@]}"; do
        # Create task-specific CoT prompt
        COT_PROMPT="Let's think step by step for the task ${capability}:"
        
        echo "----------------------------------------"
        echo "Running: ${MODEL_NAME} - ${capability^^}"
        echo "Prompt: ${COT_PROMPT}"
        echo "----------------------------------------"
        
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}/multilingual/tasks/${capability}"
        LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${capability}.log"
        
        # Create output directory
        mkdir -p ${OUTPUT_DIR}
        
        # Run the Python script
        if CUDA_VISIBLE_DEVICES=3,4 python ${PYTHON_SCRIPT} \
            --data_path ${DATA_PATH} \
            --model_name ${FULL_MODEL_PATH} \
            --all_layers \
            --balance_data \
            --use_class_weights \
            --num_epochs ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --num_classes ${NUM_CLASSES} \
            --score_range ${SCORE_RANGE} \
            --filter_capability ${capability} \
            --output_dir ${OUTPUT_DIR} \
            > ${LOG_FILE} 2>&1; then
            echo "✓ Completed: ${MODEL_NAME} - ${capability^^}"
        else
            echo "✗ Failed: ${MODEL_NAME} - ${capability^^}"
            echo "  Check log: ${LOG_FILE}"
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "Completed all capabilities for: ${MODEL_NAME}"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Results saved in: ${BASE_OUTPUT_DIR}/"
echo "Logs saved in: ${BASE_LOG_DIR}/"
echo ""
echo "Summary:"
echo "  Models tested: ${#MODELS[@]}"
for model_path in "${MODELS[@]}"; do
    MODEL_NAME=$(basename ${model_path})
    echo "    - ${MODEL_NAME}"
done
echo ""
echo "  Capabilities per model: ${#CAPABILITIES[@]}"
for capability in "${CAPABILITIES[@]}"; do
    echo "    - ${capability}"
done
echo ""
echo "  Total experiments: $((${#MODELS[@]} * ${#CAPABILITIES[@]}))"
echo ""
echo "Configuration:"
echo "  - Number of classes: ${NUM_CLASSES}"
echo "  - Score range: ${SCORE_RANGE}"
echo "  - Epochs: ${NUM_EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Balanced data: Yes"
echo "  - Class weights: Yes"