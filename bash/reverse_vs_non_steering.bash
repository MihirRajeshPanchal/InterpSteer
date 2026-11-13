#!/bin/bash
# ============================================================================
# run_nlp_metrics.bash
#
# Driver script: install dependencies and run NLP metric comparison
# (ROUGE, BLEU, BLEURT, BERTScore, TTR) for forward vs reverse steering,
# covering all four experiment types:
#   1. per-capability steering          (steering/<cap>/)
#   2. per-capability reverse steering  (reverse_steering/<cap>/)
#   3. unified sum/average steering     (steering/unified_sum|unified_average/)
#   4. unified reverse sum/average      (reverse_steering/unified_sum|unified_average/)
#
# BERTScore models loaded from HuggingFace Hub (requires network):
#   distilbert/distilbert-base-uncased  (tried first — fastest)
#   FacebookAI/roberta-large
#   google-bert/bert-base-uncased
#
# Outputs per model (under <MODEL_BASE>/analysis/nlp_metrics/):
#   merged_for_nlp.csv
#   nlp_metrics_report.txt          ← human-readable verdict
#   nlp_summary.json
#   metrics_by_capability.<fmt>
#   metric_heatmap.<fmt>
#   fwd_vs_rev_direct.<fmt>
#   verdict_summary.<fmt>
#   ttr_comparison.<fmt>
#   metrics_by_exp_type.<fmt>
#
# Cross-model output (under <BASE_STORAGE>/cross_model_analysis/nlp_metrics/):
#   cross_model_rouge1_f.<fmt>
#   cross_model_nlp_summary.json
#
# Usage:
#   bash run_nlp_metrics.bash
# ============================================================================

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON_SCRIPT="reasoning_probe/reverse_vs_non_steering.py"
BASE_STORAGE="/mnt/storage/deeksha/interpsteer"
BASE_LOG_DIR="reasoning_probe/logs/nlp_metrics_analysis"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS=(
    "sarvamai/sarvam-1"
    "CohereLabs/tiny-aya-base"
    "CoRover/BharatGPT-3B-Indic"
    # "krutrim-ai-labs/Krutrim-2-instruct"
)

# ── Capabilities ──────────────────────────────────────────────────────────────
CAPABILITIES=(
    "grounding"
    "instruction_following"
    "planning"
    "reasoning"
    "safety"
    "theory_of_mind"
)

# ── Unified modes ─────────────────────────────────────────────────────────────
UNIFIED_MODES=("sum" "average")

# ── Plot format ───────────────────────────────────────────────────────────────
PLOT_FORMAT="png"
DPI=150

# ── BERTScore batch size per GPU ──────────────────────────────────────────────
# Input is split evenly across all GPUs; each GPU scores this many sentences
# per forward pass.  Lower this if you hit CUDA OOM (e.g. 8 or 16).
BERTSCORE_BATCH_SIZE=${BERTSCORE_BATCH_SIZE:-32}

# ── Network probe: detect whether HuggingFace is reachable ───────────────────
# Set NO_NETWORK=1 to skip BLEURT/BERTScore without any probe attempt.
NO_NETWORK=${NO_NETWORK:-""}

if [ -z "${NO_NETWORK}" ]; then
    if python -c "
import socket, sys
try:
    socket.setdefaulttimeout(3)
    socket.getaddrinfo('huggingface.co', 443)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "  [info] HuggingFace reachable — BLEURT/BERTScore enabled (if installed)."
        NO_NETWORK=""
    else
        echo "  [info] HuggingFace NOT reachable — setting --no_network flag."
        NO_NETWORK="1"
    fi
fi

_NO_NETWORK_ARG=""
[ -n "${NO_NETWORK}" ] && _NO_NETWORK_ARG="--no_network"

# ── Install dependencies (once) ───────────────────────────────────────────────
echo "============================================================"
echo " Checking / installing NLP metric dependencies"
echo "============================================================"

pip install -q rouge-score nltk 2>/dev/null && echo "  ✓ rouge-score, nltk"

# BERTScore — downloads model weights from HuggingFace Hub on first use
pip install -q bert-score 2>/dev/null && echo "  ✓ bert-score" \
    || echo "  ⚠ bert-score install failed — BERTScore will be skipped"

# BLEURT via HuggingFace evaluate
pip install -q evaluate 2>/dev/null && echo "  ✓ evaluate (for BLEURT)" \
    || echo "  ⚠ evaluate install failed — BLEURT will be skipped"

echo ""
echo "============================================================"
echo " NLP Metrics: Forward vs Reverse Steering Analysis"
echo " Models:        ${#MODELS[@]}"
echo " Capabilities:  ${#CAPABILITIES[@]}"
echo " Unified modes: ${UNIFIED_MODES[*]}"
echo " BERTScore Hub models (priority order):"
echo "   1. distilbert/distilbert-base-uncased"
echo "   2. FacebookAI/roberta-large"
echo "   3. google-bert/bert-base-uncased"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Per-model loop
# ─────────────────────────────────────────────────────────────────────────────

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    MODEL_BASE="${BASE_STORAGE}/${MODEL_NAME}"

    echo "============================================================"
    echo " Model: ${MODEL_NAME}"
    echo "============================================================"

    if [ ! -d "${MODEL_BASE}" ]; then
        echo "  ⚠ No storage directory found at ${MODEL_BASE} — skipping."
        echo ""
        continue
    fi

    OUTPUT_DIR="${MODEL_BASE}/analysis/nlp_metrics"
    mkdir -p "${OUTPUT_DIR}"

    LOG_DIR="${BASE_LOG_DIR}/${MODEL_NAME}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_nlp_metrics.log"

    # ── Collect forward CSVs ──────────────────────────────────────────
    FORWARD_CSVS=()
    for CAP in "${CAPABILITIES[@]}"; do
        CAP_DIR="${MODEL_BASE}/steering/${CAP}"
        CSV="${CAP_DIR}/worst_layer_steering_results_${CAP}.csv"
        if [ -f "${CSV}" ]; then
            FORWARD_CSVS+=("${CSV}")
        else
            FOUND=$(find "${CAP_DIR}" -maxdepth 1 -name "*.csv" 2>/dev/null | head -1)
            [ -n "${FOUND}" ] && FORWARD_CSVS+=("${FOUND}")
        fi
    done
    for MODE in "${UNIFIED_MODES[@]}"; do
        UNIFIED_DIR="${MODEL_BASE}/steering/unified_${MODE}"
        CSV="${UNIFIED_DIR}/unified_${MODE}_results.csv"
        if [ -f "${CSV}" ]; then
            FORWARD_CSVS+=("${CSV}")
        else
            FOUND=$(find "${UNIFIED_DIR}" -maxdepth 1 -name "*.csv" 2>/dev/null | head -1)
            [ -n "${FOUND}" ] && FORWARD_CSVS+=("${FOUND}")
        fi
    done

    # ── Collect reverse CSVs ──────────────────────────────────────────
    REVERSE_CSVS=()
    for CAP in "${CAPABILITIES[@]}"; do
        CAP_DIR="${MODEL_BASE}/reverse_steering/${CAP}"
        CSV="${CAP_DIR}/worst_layer_steering_results_${CAP}.csv"
        if [ -f "${CSV}" ]; then
            REVERSE_CSVS+=("${CSV}")
        else
            FOUND=$(find "${CAP_DIR}" -maxdepth 1 -name "*.csv" 2>/dev/null | head -1)
            [ -n "${FOUND}" ] && REVERSE_CSVS+=("${FOUND}")
        fi
    done
    for MODE in "${UNIFIED_MODES[@]}"; do
        UNIFIED_DIR="${MODEL_BASE}/reverse_steering/unified_${MODE}"
        CSV="${UNIFIED_DIR}/unified_${MODE}_results.csv"
        if [ -f "${CSV}" ]; then
            REVERSE_CSVS+=("${CSV}")
        else
            FOUND=$(find "${UNIFIED_DIR}" -maxdepth 1 -name "*.csv" 2>/dev/null | head -1)
            [ -n "${FOUND}" ] && REVERSE_CSVS+=("${FOUND}")
        fi
    done

    echo "  Forward CSVs found:  ${#FORWARD_CSVS[@]}"
    echo "  Reverse CSVs found:  ${#REVERSE_CSVS[@]}"

    if [ ${#FORWARD_CSVS[@]} -eq 0 ] && [ ${#REVERSE_CSVS[@]} -eq 0 ]; then
        echo "  ⚠ No CSVs found — skipping."
        echo ""
        continue
    fi

    FWD_ARGS="${FORWARD_CSVS[*]}"
    REV_ARGS="${REVERSE_CSVS[*]}"

    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Log:        ${LOG_FILE}"
    echo ""

    if python ${PYTHON_SCRIPT} \
        --model_name        "${MODEL_NAME}" \
        --forward_csvs      ${FWD_ARGS} \
        --reverse_csvs      ${REV_ARGS} \
        --output_dir        "${OUTPUT_DIR}" \
        --plot_format       "${PLOT_FORMAT}" \
        --dpi               ${DPI} \
        --capabilities      "${CAPABILITIES[@]}" \
        --unified_modes     "${UNIFIED_MODES[@]}" \
        --bertscore_batch_size ${BERTSCORE_BATCH_SIZE} \
        ${_NO_NETWORK_ARG} \
        > "${LOG_FILE}" 2>&1; then
        echo "  ✓ NLP analysis complete: ${MODEL_NAME}"
        echo "  Report: ${OUTPUT_DIR}/nlp_metrics_report.txt"
    else
        echo "  ✗ NLP analysis failed: ${MODEL_NAME}"
        echo "    Check log: ${LOG_FILE}"
    fi
    echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Cross-model summary
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Cross-Model NLP Summary"
echo "============================================================"
echo ""

SUMMARY_JSONS=()
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    SUMMARY="${BASE_STORAGE}/${MODEL_NAME}/analysis/nlp_metrics/nlp_summary.json"
    [ -f "${SUMMARY}" ] && SUMMARY_JSONS+=("${SUMMARY}")
done

CROSS_OUTPUT_DIR="${BASE_STORAGE}/cross_model_analysis/nlp_metrics"
mkdir -p "${CROSS_OUTPUT_DIR}"
CROSS_LOG="${BASE_LOG_DIR}/cross_model_nlp.log"

if [ ${#SUMMARY_JSONS[@]} -gt 0 ]; then
    if python ${PYTHON_SCRIPT} \
        --mode          cross_model \
        --summary_jsons "${SUMMARY_JSONS[@]}" \
        --output_dir    "${CROSS_OUTPUT_DIR}" \
        --plot_format   "${PLOT_FORMAT}" \
        --dpi           ${DPI} \
        --bertscore_batch_size ${BERTSCORE_BATCH_SIZE} \
        ${_NO_NETWORK_ARG} \
        > "${CROSS_LOG}" 2>&1; then
        echo "  ✓ Cross-model NLP analysis complete → ${CROSS_OUTPUT_DIR}"
    else
        echo "  ✗ Cross-model analysis failed — check ${CROSS_LOG}"
    fi
else
    echo "  ⚠ No per-model summaries found; skipping cross-model analysis."
fi

echo ""
echo "============================================================"
echo " ALL NLP ANALYSIS COMPLETE"
echo "============================================================"
echo ""
echo "Per-model outputs:"
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "  ${BASE_STORAGE}/${MODEL_NAME}/analysis/nlp_metrics/"
done
echo ""
echo "Cross-model output:"
echo "  ${CROSS_OUTPUT_DIR}"