#!/bin/bash
#
# Compute per-prompt GRPO gradients on rollouts.
#
# Usage:
#   bash run_gradient.sh <start_idx> <end_idx> [extra args forwarded to checkpoint_gradient_analysis.py]
#
# Common extra args:
#   --lora                       Wrap the model in a LoRA adapter
#   --lora_r 16 --lora_alpha 32  LoRA rank / alpha
#   --pos_only                   Use only correct rollouts
#   --prompt_type qwen|llama     Prompt template (auto-detected from model path otherwise)
#   --id_file PATH               Restrict to question_ids listed in this JSON
#   --batch_size 2               Per-batch rollouts during gradient computation
#   --type mean|squared|both     Output type
#
# Configure via environment variables (defaults in parens):
#   ROLLOUTS_FILE  glob path(s) to rollout JSON files produced by inference/test.sh
#                  (default: ./results/rollouts*.json)
#   MODEL_PATH     HF model dir or hub id
#                  (default: Qwen/Qwen2.5-0.5B)
#   OUTPUT_DIR     output dir for per-prompt gradient .npz files
#                  (default: ./grpo_gradient_analysis_output/qwen_0.5b_demo)

set -euo pipefail

ROLLOUTS_FILE="${ROLLOUTS_FILE:-./results/rollouts*.json}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_gradient_analysis_output/qwen_0.5b_demo}"

start_idx="${1:-}"
end_idx="${2:-}"
shift 2 || true

if [ -z "$start_idx" ] || [ -z "$end_idx" ]; then
    echo "Usage: $0 <start_idx> <end_idx> [extra args]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "ROLLOUTS_FILE: $ROLLOUTS_FILE"
echo "MODEL_PATH:    $MODEL_PATH"
echo "OUTPUT_DIR:    $OUTPUT_DIR"

python -u "$SCRIPT_DIR/checkpoint_gradient_analysis.py" \
    --rollouts_file $ROLLOUTS_FILE \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device auto \
    --type mean \
    --start_idx "$start_idx" \
    --end_idx "$end_idx" \
    --batch_size 2 \
    "$@"

echo "Done. Output: $OUTPUT_DIR"
