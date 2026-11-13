#!/usr/bin/env bash
# run_generation.sh — generates orig_score + orig_response via OpenRouter GPT-OSS
#
# Usage:
#   bash run_generation.sh [options]
#
# Options:
#   --input PATH       Input CSV               (default: reasoning_probe/data/combined_dataset.csv)
#   --output PATH      Output CSV              (default: reasoning_probe/data/combined_dataset_with_responses.csv)
#   --fewshot PATH     Few-shot examples CSV   (default: /Data/deeksha/probing/combined_cleaned_xx.csv)
#   --temperature F    Sampling temperature    (default: 0.7)
#
# Few-shot CSV must have columns:
#   capability, task, system_prompt, orig_instruction, orig_response, orig_score
#
# Input CSV must have column:
#   reference_answer  (gold score-5 answer shown in prompt)
#
# Self-resuming: re-running skips already-completed rows (tracked in output CSV itself).
# Logs to reasoning_probe/logs/generate_responses.log

set -euo pipefail

INPUT="reasoning_probe/data/combined_dataset.csv"
OUTPUT="reasoning_probe/data/combined_dataset_with_responses.csv"
FEWSHOT="/Data/deeksha/probing/combined_cleaned_xx.csv"
TEMPERATURE=0.7
LOG_FILE="reasoning_probe/logs/generate_responses.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)       INPUT="$2";       shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --fewshot)     FEWSHOT="$2";     shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

GENERATE_PY="reasoning_probe/generate_responses.py"

[[ -f "$INPUT" ]]       || { echo "Input not found: $INPUT";         exit 1; }
[[ -f "$GENERATE_PY" ]] || { echo "Script not found: $GENERATE_PY"; exit 1; }

# Warn (but don't abort) if optional files are missing
[[ -f "$FEWSHOT" ]] || echo "⚠  Few-shot file not found: $FEWSHOT — generation will run without few-shot examples."

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG_FILE")"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "  Response Generation  (OpenRouter GPT-OSS)"
echo "  Started : $(date)"
echo "  Input   : $INPUT"
echo "  Output  : $OUTPUT"
echo "  Few-shot: $FEWSHOT"
echo "  Temp    : $TEMPERATURE"
echo "========================================================"

pip install openai pandas tqdm -q

python "$GENERATE_PY" \
    --input       "$INPUT" \
    --output      "$OUTPUT" \
    --fewshot     "$FEWSHOT" \
    --temperature "$TEMPERATURE"

echo ""
echo "========================================================"
echo "Done — $(date)"
echo "Output : $OUTPUT"
echo "Log    : $LOG_FILE"
echo "========================================================"